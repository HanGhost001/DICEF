import gc
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from dicef.clinical import ClinicalClassifier
from dicef.data import DOMAIN_NAMES, ImageDataset, clinical_matrix, read_manifest, validate_five_folds
from dicef.losses import MODALITIES, ramp_weight, stage1_loss
from dicef.metrics import classification_metrics, paired_bootstrap
from dicef.mocef import MOCEF, softmax
from dicef.models import FusionMlp, T1PetDtiThreeEncoderSharedHead
from dicef.normalization import normalize_t1,normalize_pet,normalize_dti
from dicef.synthetic import make_synthetic_manifest

torch.set_num_threads(2)


class FusionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.pi = rng.dirichlet([2,3,4],size=18)
        self.pc = rng.dirichlet([3,4,2],size=18)
        self.y = np.tile([0,1,2],6)
        self.model = MOCEF().fit(self.pi,self.y)

    def test_kl_and_logit_equivalence(self):
        q = self.model.predict_proba(self.pc,self.pi)
        expected = softmax(np.log(self.pc)+self.model.compatibility(self.pi))
        np.testing.assert_allclose(q,expected,rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(q.sum(1),1)

    def test_beta_zero(self):
        model = MOCEF(beta=0,prototypes=self.model.prototypes)
        np.testing.assert_allclose(model.predict_proba(self.pc,self.pi),self.pc)

    def test_zero_components_and_serialization(self):
        model = MOCEF.from_dict(json.loads(json.dumps(self.model.to_dict())))
        q = model.predict_proba(np.eye(3),np.eye(3))
        self.assertTrue(np.isfinite(q).all())
        np.testing.assert_allclose(q.sum(1),1)

    def test_prototype_fit_requires_class_coverage(self):
        with self.assertRaises(ValueError):
            MOCEF().fit(self.pi,np.zeros(18))
        with self.assertRaises(ValueError):
            MOCEF(beta=-1)
        with self.assertRaises(ValueError):
            MOCEF(beta=float("nan"))
        with self.assertRaises(ValueError):
            self.model.predict_proba(self.pc[:2],self.pi)

    def test_identical_paired_predictions(self):
        result = paired_bootstrap(self.y,self.pi,self.pi,samples=60)
        self.assertEqual(result["bacc"]["delta"],0)
        self.assertEqual(result["bacc"]["ci95"],[0,0])
        cluster = paired_bootstrap(self.y,self.pi,self.pi,np.repeat(np.arange(6),3),samples=60)
        self.assertEqual(cluster["macro_f1"]["ci95"],[0,0])

    def test_perfect_metrics(self):
        m = classification_metrics(self.y,np.eye(3)[self.y])
        self.assertEqual(m["bacc"],1)
        self.assertEqual(m["macro_f1"],1)


class ModelTests(unittest.TestCase):
    def test_ramp_endpoints(self):
        self.assertEqual(ramp_weight(1),0)
        self.assertEqual(ramp_weight(30),2.2)
        self.assertEqual(ramp_weight(80),2.2)
        self.assertAlmostEqual(ramp_weight(15),1.0404472005560408)

    def test_shared_head_is_one_parameter_set(self):
        model = T1PetDtiThreeEncoderSharedHead(dropout=.2).eval()
        self.assertEqual(sum(name == "shared_head" for name,_ in model.named_modules()),1)
        with torch.no_grad():
            outputs = model(torch.zeros(1,1,32,32,32),torch.zeros(1,1,32,32,32),torch.zeros(1,3,32,32,32))
        for m in MODALITIES:
            self.assertEqual(outputs[f"{m}_feature"].shape,(1,512))
            self.assertEqual(outputs[f"shared_{m}_logits"].shape,(1,3))
        del model
        gc.collect()

    def test_teacher_features_are_detached(self):
        torch.manual_seed(8)
        outputs,teachers = {},{}
        for m in MODALITIES:
            outputs[f"{m}_feature"] = torch.randn(3,512,requires_grad=True)
            outputs[f"{m}_logits"] = torch.randn(3,3,requires_grad=True)
            outputs[f"shared_{m}_logits"] = torch.randn(3,3,requires_grad=True)
            teachers[m] = torch.randn(3,512,requires_grad=True)
        loss,parts = stage1_loss(outputs,teachers,torch.arange(3),torch.ones(3),.5)
        loss.backward()
        self.assertTrue(all(teachers[m].grad is None for m in MODALITIES))
        self.assertTrue(all(outputs[f"{m}_feature"].grad is not None for m in MODALITIES))
        self.assertEqual(set(parts),{"private_ce","shared_ce","umt_mse"})

    def test_stage2_backward(self):
        model = FusionMlp(1536,256,.2)
        logits = model(torch.randn(6,1536))
        torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,2,0,1,2])).backward()
        self.assertEqual(logits.shape,(6,3))
        self.assertTrue(all(p.grad is not None for p in model.parameters()))


class DataTests(unittest.TestCase):
    def test_cache_transforms(self):
        np.testing.assert_allclose(normalize_t1(np.array([-6,0,6])),[-1,0,1])
        np.testing.assert_allclose(normalize_pet(np.array([0,.5,1,2.5,8.])),[0,-1,0,3,3])
        np.testing.assert_allclose(normalize_dti(np.array([2.]),np.array([.008]),np.array([.002])),[[1],[1],[.5]])

    def test_manifest_and_missingness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = make_synthetic_manifest(directory)
            frame = read_manifest(path)
            x = clinical_matrix(frame)
            self.assertEqual(x.shape,(36,17))
            self.assertEqual(x[0,10],1)
            sample = ImageDataset(frame,shape=(32,32,32))[0]
            self.assertEqual(sample["dti"].shape,(3,32,32,32))
            original = np.load(frame.iloc[0].cache_path,allow_pickle=False)
            np.testing.assert_array_equal(sample["t1"].numpy(),original[0:1].astype(np.float32)*3)
            frame.loc[frame.split.eq("test"),"center_id"] = "synthetic_train_center"
            frame.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,"crosses"):
                read_manifest(path)

    def test_clinical_numeric_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = read_manifest(make_synthetic_manifest(directory))
            train = frame.loc[frame.split.eq("train")]
            x = clinical_matrix(train)
            model = ClinicalClassifier().fit(x,train.label,c_grid=[.1,1],seed=42)
            copied = ClinicalClassifier.from_dict(json.loads(json.dumps(model.state)))
            np.testing.assert_allclose(model.predict_proba(x),copied.predict_proba(x),atol=1e-14)
            self.assertEqual(len(model.state["features"]),17)
            state = dict(model.state,classes=["AD","MCI","CN"])
            with self.assertRaises(ValueError):
                ClinicalClassifier.from_dict(state)

    def test_five_fold_coverage(self):
        frames = []
        for fold in range(5):
            rows = [{"sample_id":f"synthetic_{c}_{i}","center_id":str(c),"label":i%3,
                "split":"test" if c==fold else "development" if c==(fold+1)%5 else "train"}
                for c in range(5) for i in range(6)]
            frames.append(pd.DataFrame(rows))
        self.assertEqual(validate_five_folds(frames)["participants"],30)
        frames[-1] = frames[0].copy()
        with self.assertRaises(ValueError):
            validate_five_folds(frames)


if __name__ == "__main__":
    unittest.main()
