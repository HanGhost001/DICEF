"""Portable training harness for the formal DICEF pathway only."""
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .augmentation import augment_batch, augmentation_config
from .backbone import standard_resnet18_dual_branch
from .clinical import ClinicalClassifier
from .data import ImageDataset, clinical_matrix, read_manifest
from .losses import MODALITIES, ramp_weight, stage1_loss
from .metrics import classification_metrics
from .models import FusionMlp, T1PetDtiThreeEncoderSharedHead
from .mocef import MOCEF


def save_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def validate_config(cfg):
    positive = ("batch_size","accumulation_steps","imaging_epochs","imaging_patience",
                "imaging_lr","stage2_epochs","stage2_patience","stage2_batch_size",
                "stage2_lr","stage2_hidden")
    for name in positive:
        if not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if len(cfg["shape"]) != 3 or any(int(v) != v or v < 32 for v in cfg["shape"]):
        raise ValueError("Each spatial dimension must be an integer >= 32")
    if cfg["workers"] < 0 or not 0 <= cfg["dropout"] < 1 or cfg["weight_decay"] < 0:
        raise ValueError("Invalid worker count, dropout or weight decay")
    ramp_weight(1,cfg["umt_maximum"],cfg["umt_start"],cfg["umt_end"])
    MOCEF(cfg["beta"])
    return cfg


def loader(frame,cfg,training=False,modality=None):
    kwargs = {"prefetch_factor":1,"persistent_workers":True} if cfg["workers"] else {}
    return DataLoader(ImageDataset(frame,cfg["shape"],modality),batch_size=cfg["batch_size"],
        shuffle=training,num_workers=cfg["workers"],pin_memory=torch.cuda.is_available(),drop_last=False,**kwargs)


def class_weights(y,device):
    counts = np.bincount(np.asarray(y,dtype=int),minlength=3)
    if (counts == 0).any():
        raise ValueError("All three training classes are required")
    return torch.tensor(counts.sum()/(3.0*counts),dtype=torch.float32,device=device)


def autocast(device,enabled):
    return torch.amp.autocast("cuda",enabled=True) if device.type == "cuda" and enabled else nullcontext()


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def image_epoch(model,batches,device,cfg,weights,optimizer=None,teachers=None,epoch=1,modality=None,scaler=None):
    training = optimizer is not None
    model.train(training)
    if teachers is not None:
        for teacher in teachers.values():
            teacher.eval()
    y_all, predictions, losses = [],{m:[] for m in ((modality,) if modality else MODALITIES)},[]
    optimizer_steps,amp_skips = 0,0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step,batch in enumerate(batches,start=1):
        inputs = {m:batch[m].to(device,non_blocking=True) for m in predictions}
        if training and cfg["augmentation"]:
            # Source behavior: per-modality affine draws, but identical teacher/student input tensors.
            inputs = {m:augment_batch(x,augmentation_config("strong" if m == "t1" else "spatial"))
                      for m,x in inputs.items()}
        y = batch["label"].to(device)
        with torch.set_grad_enabled(training),autocast(device,cfg["amp"]):
            if modality:
                logits = model(inputs[modality])
                loss = F.cross_entropy(logits,y,weight=weights)
                outputs = {f"{modality}_logits":logits}
            else:
                outputs = model(inputs["t1"],inputs["pet"],inputs["dti"])
                if training:
                    with torch.no_grad():
                        target = {m:teachers[m].global_feature(teachers[m].forward_features(inputs[m])) for m in MODALITIES}
                    weight = ramp_weight(epoch,cfg["umt_maximum"],cfg["umt_start"],cfg["umt_end"])
                    loss,_ = stage1_loss(outputs,target,y,weights,weight)
                else:
                    loss = torch.stack([F.cross_entropy(outputs[f"{m}_logits"],y,weight=weights)
                                        for m in MODALITIES]).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite imaging loss")
        if training:
            # Fixed divisor matches the archived final partial accumulation group.
            scaler.scale(loss/cfg["accumulation_steps"]).backward()
            if step % cfg["accumulation_steps"] == 0 or step == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < previous_scale
                amp_skips += int(skipped)
                optimizer_steps += int(not skipped)
                optimizer.zero_grad(set_to_none=True)
        y_all.extend(y.detach().cpu().tolist())
        for m in predictions:
            predictions[m].extend(outputs[f"{m}_logits"].detach().float().softmax(1).cpu().tolist())
        losses.append(float(loss.detach()))
    scores = {m:classification_metrics(y_all,p) for m,p in predictions.items()}
    return {"loss":float(np.mean(losses)),"selection_bacc":float(np.mean([s["bacc"] for s in scores.values()])),
            "optimizer_steps":optimizer_steps,"amp_skipped_steps":amp_skips}


def fit_image_model(model,train,development,cfg,device,path,modality=None,teachers=None):
    weights = class_weights(train.label,device)
    train_loader = loader(train,cfg,True,modality)
    dev_loader = loader(development,cfg,False,modality)
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg["imaging_lr"],weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg["imaging_epochs"])
    scaler = torch.amp.GradScaler("cuda",enabled=device.type == "cuda" and cfg["amp"])
    best,stale,history = -np.inf,0,[]
    for epoch in range(1,cfg["imaging_epochs"]+1):
        tr = image_epoch(model,train_loader,device,cfg,weights,optimizer,teachers,epoch,modality,scaler)
        if tr["optimizer_steps"] == 0:
            raise RuntimeError("All optimizer steps were skipped; inspect input scaling or retry with amp=false")
        dev = image_epoch(model,dev_loader,device,cfg,weights,modality=modality)
        scheduler.step()
        history.append({"epoch":epoch,"train":tr,"development":dev})
        print(f"{path.stem}: epoch={epoch} development_BAcc={dev['selection_bacc']:.4f}",flush=True)
        if dev["selection_bacc"] > best:
            best,stale = dev["selection_bacc"],0
            torch.save({"model_state":cpu_state(model),"epoch":epoch},path)
        else:
            stale += 1
        if stale >= cfg["imaging_patience"]:
            break
    state = torch.load(path,map_location=device,weights_only=True)
    model.load_state_dict(state["model_state"],strict=True)
    save_json(path.with_suffix(".history.json"),history)
    return model.eval()


@torch.no_grad()
def extract_features(model,frame,cfg,device):
    model.eval()
    chunks,ids = [],[]
    for batch in loader(frame,cfg):
        features = []
        for m in MODALITIES:
            encoder = getattr(model,f"{m}_encoder")
            x = batch[m].to(device,non_blocking=True)
            with autocast(device,cfg["amp"]):
                f = encoder.global_feature(encoder.forward_features(x))
            features.append(f.float())
        chunks.append(torch.cat(features,dim=1).cpu().numpy())
        ids.extend(batch["sample_id"])
    if ids != frame.sample_id.tolist():
        raise RuntimeError("Feature extraction changed participant order")
    return np.concatenate(chunks).astype(np.float32)


@torch.no_grad()
def predict_head(head,x,device,batch_size):
    head.eval()
    batches = DataLoader(TensorDataset(torch.from_numpy(x)),batch_size=batch_size,shuffle=False)
    return np.concatenate([head(batch.to(device)).softmax(1).cpu().numpy() for (batch,) in batches]).astype(np.float64)


def fit_feature_head(train_x,train_y,dev_x,dev_y,cfg,device,path):
    set_seed(cfg["seed"])
    model = FusionMlp(1536,cfg["stage2_hidden"],cfg["dropout"]).to(device)
    weights = class_weights(train_y,device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg["stage2_lr"],weight_decay=cfg["weight_decay"])
    batches = DataLoader(TensorDataset(torch.from_numpy(train_x),torch.as_tensor(np.asarray(train_y),dtype=torch.long)),
                         batch_size=cfg["stage2_batch_size"],shuffle=True)
    best,stale,history = -np.inf,0,[]
    for epoch in range(1,cfg["stage2_epochs"]+1):
        model.train()
        for x,y in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(device)),y.to(device),weight=weights)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite Stage 2 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
        score = classification_metrics(dev_y,predict_head(model,dev_x,device,cfg["stage2_batch_size"]))["bacc"]
        history.append({"epoch":epoch,"development_bacc":score})
        print(f"stage2: epoch={epoch} development_BAcc={score:.4f}",flush=True)
        if score > best:
            best,stale = score,0
            torch.save({"model_state":cpu_state(model),"epoch":epoch},path)
        else:
            stale += 1
        if stale >= cfg["stage2_patience"]:
            break
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True)["model_state"],strict=True)
    save_json(path.with_suffix(".history.json"),history)
    return model.eval()


def fit_fold(manifest,output,cfg,fold=1,device="cuda"):
    cfg = validate_config(dict(cfg))
    if fold not in range(1,6):
        raise ValueError("Outer fold must be 1..5")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu explicitly")
    frame = read_manifest(manifest)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; use a new private run directory")
    output.mkdir(parents=True,exist_ok=True)
    save_json(output / "config.json",cfg)
    train = frame.loc[frame.split.eq("train")].reset_index(drop=True)
    dev = frame.loc[frame.split.eq("development")].reset_index(drop=True)
    teachers = {}
    for m,channels in (("t1",1),("pet",1),("dti",3)):
        set_seed(cfg["seed"])
        model = standard_resnet18_dual_branch(channels,3).to(device)
        fit_image_model(model,train,dev,cfg,device,output/f"teacher_{m}.pt",modality=m)
        teachers[m] = model.cpu().requires_grad_(False).eval()
        gc.collect()
    teachers = {m:model.to(device) for m,model in teachers.items()}
    set_seed(cfg["seed"]+fold)
    encoders = T1PetDtiThreeEncoderSharedHead(cfg["dropout"]).to(device)
    fit_image_model(encoders,train,dev,cfg,device,output/"stage1.pt",teachers=teachers)
    del teachers,model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    encoders.requires_grad_(False).eval()
    # Stable ordering is part of the frozen-feature Stage 2 training contract.
    train = train.sort_values("sample_id",kind="stable").reset_index(drop=True)
    dev = dev.sort_values("sample_id",kind="stable").reset_index(drop=True)
    train_x = extract_features(encoders,train,cfg,device)
    dev_x = extract_features(encoders,dev,cfg,device)
    encoders.cpu()
    head = fit_feature_head(train_x,train.label,dev_x,dev.label,cfg,device,output/"stage2.pt")
    pi = predict_head(head,dev_x,device,cfg["stage2_batch_size"])
    clinical = ClinicalClassifier().fit(clinical_matrix(train),train.label,
        cfg["clinical_c_grid"],cfg["seed"]+fold*cfg["clinical_cv_seed_offset_per_fold"])
    mocef = MOCEF(cfg["beta"]).fit(pi,dev.label)
    save_json(output/"clinical.json",clinical.state)
    save_json(output/"mocef.json",mocef.to_dict())
    save_json(output/"fit_complete.json",{"fold":fold,"classes":["CN","MCI","AD"],
        "fit_splits":["train","development"],"outer_test_used_for_fitting":False,
        "training_subjects":len(train),"development_subjects":len(dev)})
    return output


def predict_fold(manifest,model_dir,split="test",device="cuda"):
    model_dir,device = Path(model_dir),torch.device(device)
    if not (model_dir/"fit_complete.json").is_file():
        raise ValueError("Model directory has no successful fit completion marker")
    cfg = validate_config(json.loads((model_dir/"config.json").read_text()))
    frame = read_manifest(manifest,training=False)
    frame = frame.loc[frame.split.eq(split)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("Selected inference partition is empty")
    model = T1PetDtiThreeEncoderSharedHead(cfg["dropout"])
    model.load_state_dict(torch.load(model_dir/"stage1.pt",map_location="cpu",weights_only=True)["model_state"],strict=True)
    model.to(device).requires_grad_(False).eval()
    features = extract_features(model,frame,cfg,device)
    model.cpu()
    head = FusionMlp(1536,cfg["stage2_hidden"],cfg["dropout"])
    head.load_state_dict(torch.load(model_dir/"stage2.pt",map_location="cpu",weights_only=True)["model_state"],strict=True)
    pi = predict_head(head.to(device),features,device,cfg["stage2_batch_size"])
    clinical = ClinicalClassifier.from_dict(json.loads((model_dir/"clinical.json").read_text()))
    pc = clinical.predict_proba(clinical_matrix(frame))
    mocef = MOCEF.from_dict(json.loads((model_dir/"mocef.json").read_text()))
    q = mocef.predict_proba(pc,pi)
    result = frame[["sample_id"]].copy()
    for prefix,p in (("pI",pi),("pC",pc),("q",q)):
        for index,name in enumerate(("CN","MCI","AD")):
            result[f"{prefix}_{name}"] = p[:,index]
        result[f"{prefix}_prediction"] = p.argmax(1)
    metrics = None if "label" not in frame else {name:classification_metrics(frame.label,p)
        for name,p in (("image",pi),("clinical",pc),("MOCEF",q))}
    return result,metrics
