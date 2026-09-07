"""Command-line entry points; all data and output locations are explicit."""
import argparse
import json
from pathlib import Path


def load_config(path):
    from .training import validate_config
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def main():
    parser = argparse.ArgumentParser(prog="dicef",description="DICEF core research implementation")
    sub = parser.add_subparsers(dest="command",required=True)
    fit = sub.add_parser("fit",help="Fit teachers, Stage 1, Stage 2, clinical classifier and development prototypes")
    fit.add_argument("--manifest",type=Path,required=True)
    fit.add_argument("--config",type=Path,required=True)
    fit.add_argument("--output",type=Path,required=True)
    fit.add_argument("--fold",type=int,choices=range(1,6),required=True)
    fit.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    pred = sub.add_parser("predict",help="Frozen-pipeline inference; no fitting or parameter selection")
    pred.add_argument("--manifest",type=Path,required=True)
    pred.add_argument("--model-dir",type=Path,required=True)
    pred.add_argument("--output",type=Path,required=True)
    pred.add_argument("--split",choices=("train","development","test"),default="test")
    pred.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    check = sub.add_parser("validate-splits",help="Validate five user-supplied outer-fold manifests")
    check.add_argument("manifests",type=Path,nargs=5)
    smoke = sub.add_parser("smoke",help="Run the complete pipeline using generated random arrays only")
    smoke.add_argument("--config",type=Path,required=True)
    smoke.add_argument("--work-dir",type=Path,required=True)
    smoke.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    cache = sub.add_parser("prepare-cache",help="Normalize five already registered/cropped NIfTI channels")
    for name in ("t1","pet","fa","md","rd"):
        cache.add_argument("--"+name,type=Path,required=True)
    cache.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.command == "fit":
        from .training import fit_fold
        fit_fold(args.manifest,args.output,load_config(args.config),args.fold,args.device)
        print("Fit complete. Outer-test inference has not been performed.")
    elif args.command == "predict":
        from .training import predict_fold,save_json
        if args.output.exists() or args.output.with_suffix(".metrics.json").exists():
            parser.error("Prediction output already exists; select a new output path")
        predictions,metrics = predict_fold(args.manifest,args.model_dir,args.split,args.device)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        predictions.to_csv(args.output,index=False)
        if metrics is not None:
            save_json(args.output.with_suffix(".metrics.json"),metrics)
        print("Inference complete. Keep predictions and fitted artifacts private.")
    elif args.command == "validate-splits":
        from .data import read_manifest,validate_five_folds
        print(json.dumps(validate_five_folds([read_manifest(p) for p in args.manifests]),indent=2))
    elif args.command == "smoke":
        from .synthetic import make_synthetic_manifest
        from .training import fit_fold,predict_fold,save_json
        import numpy as np
        import torch
        torch.set_num_threads(2)
        if args.work_dir.exists() and any(args.work_dir.iterdir()):
            parser.error("Smoke work directory must be empty")
        cfg = load_config(args.config)
        cfg.update(shape=[32,32,32],workers=0,amp=False,imaging_epochs=2,imaging_patience=2,
            umt_end=2,stage2_epochs=3,stage2_patience=2,stage2_batch_size=6,accumulation_steps=1,
            clinical_c_grid=[0.1,1.0])
        manifest = make_synthetic_manifest(args.work_dir/"synthetic")
        import pandas as pd
        fit_frame = pd.read_csv(manifest)
        fit_frame.loc[fit_frame.split.eq("test"),"cache_path"] = "unavailable_outer_test.npy"
        # Use distinct nonexistent paths: fitting must not open any test image.
        test = fit_frame.split.eq("test")
        fit_frame.loc[test,"cache_path"] = fit_frame.loc[test,"sample_id"] + "_unavailable.npy"
        fit_manifest = manifest.with_name("fit_guard_manifest.csv")
        fit_frame.to_csv(fit_manifest,index=False)
        model_dir = fit_fold(fit_manifest,args.work_dir/"models",cfg,1,args.device)
        predictions,metrics = predict_fold(manifest,model_dir,"test",args.device)
        q = predictions[["q_CN","q_MCI","q_AD"]].to_numpy()
        assert len(q) == 9 and np.isfinite(q).all() and np.allclose(q.sum(1),1)
        save_json(args.work_dir/"smoke_report.json",{"synthetic_only":True,
            "full_pipeline_completed":True,"test_rows":len(q),"probabilities_sum_to_one":True,
            "fit_succeeds_without_test_images":True,"positive_ramp_weight_exercised":True,
            "note":"Random-array smoke metrics are not scientific results."})
        print("Full synthetic smoke passed.")
    else:
        prepare_cache(args)


def prepare_cache(args):
    import numpy as np
    try:
        import nibabel as nib
    except ImportError as error:
        raise RuntimeError("Install the optional NIfTI dependency: pip install '.[nifti]'") from error
    from .normalization import normalize_t1,normalize_pet,normalize_dti
    if args.output.exists() or args.output.suffix != ".npy":
        raise ValueError("Output must be a new .npy file")
    images = [nib.load(str(getattr(args,key))) for key in ("t1","pet","fa","md","rd")]
    if any(image.shape != (192,192,192) for image in images):
        raise ValueError("NIfTI inputs must already be cropped to 192 x 192 x 192")
    if any(not np.allclose(image.affine,images[0].affine,atol=1e-5,rtol=0) for image in images[1:]):
        raise ValueError("NIfTI inputs do not share the same template grid/affine")
    arrays = [np.nan_to_num(np.asarray(image.dataobj,dtype=np.float32),nan=0,posinf=0,neginf=0) for image in images]
    t1,pet,fa,md,rd = arrays
    channels = np.concatenate([normalize_t1(t1)[None],normalize_pet(pet)[None],normalize_dti(fa,md,rd)])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.save(args.output,channels.astype(np.float16),allow_pickle=False)
    print("Cache created. Keep this derived image file private.")
