"""Explicit input contracts. No cohort files, identifiers or default data paths."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DOMAIN_NAMES = (
    "moca_visuospatial_executive", "moca_naming", "moca_attention", "moca_language",
    "moca_abstraction", "moca_delayed_free_recall", "moca_orientation",
)
RAW_CLINICAL_NAMES = ("age", "sex_male", "moca_total", *DOMAIN_NAMES)
CLINICAL_NAMES = (*RAW_CLINICAL_NAMES, *(f"{name}_missing" for name in DOMAIN_NAMES))
SPLITS = ("train", "development", "test")


def read_manifest(path, training=True):
    path = Path(path).resolve()
    frame = pd.read_csv(path, dtype={"sample_id":str, "center_id":str})
    required = {"sample_id", "center_id", "split", "cache_path", *RAW_CLINICAL_NAMES}
    if training:
        required.add("label")
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing manifest columns: {sorted(required-set(frame.columns))}")
    if frame.empty or frame[list(required- set(RAW_CLINICAL_NAMES))].isna().any().any():
        raise ValueError("Manifest is empty or has missing metadata")
    if frame.sample_id.duplicated().any() or frame.sample_id.str.strip().eq("").any():
        raise ValueError("sample_id must be nonempty and unique within each outer fold")
    if not set(frame.split).issubset(SPLITS):
        raise ValueError(f"split must be one of {SPLITS}")
    if training and set(frame.split) != set(SPLITS):
        raise ValueError("Training manifest must define train, development and test partitions")
    if "label" in frame and (frame.label.isna().any() or not frame.label.isin([0,1,2]).all()):
        raise ValueError("label must use CN=0, MCI=1, AD=2")
    if frame.groupby("center_id").split.nunique().gt(1).any():
        raise ValueError("An acquisition center crosses partitions")
    frame["cache_path"] = frame.cache_path.map(lambda p:str((path.parent / str(p)).resolve()))
    if frame.cache_path.duplicated().any():
        raise ValueError("The same image cache occurs more than once")
    # Validate values without fitting imputation or scaling on held-out partitions.
    clinical_matrix(frame)
    if training:
        for split, minimum in (("train",3),("development",2),("test",1)):
            counts = frame.loc[frame.split.eq(split),"label"].value_counts().reindex([0,1,2],fill_value=0)
            if (counts < minimum).any():
                raise ValueError(f"{split} requires at least {minimum} samples per class")
    return frame


def clinical_matrix(frame):
    base = frame[list(RAW_CLINICAL_NAMES)].to_numpy(dtype=np.float64)
    if np.isinf(base).any() or np.isnan(base[:,:3]).any():
        raise ValueError("Age, sex and MoCA total must be present; infinity is not allowed")
    if not np.isin(base[:,1], [0,1]).all():
        raise ValueError("sex_male must be 0 or 1")
    missing = np.isnan(base[:,3:]).astype(np.float64)
    return np.concatenate([base,missing],axis=1)


def validate_five_folds(frames):
    if len(frames) != 5:
        raise ValueError("Expected five outer-fold manifests")
    reference = frames[0].set_index("sample_id")[["center_id","label"]].sort_index()
    tests = []
    for frame in frames:
        candidate = frame.set_index("sample_id")[["center_id","label"]].sort_index()
        if not candidate.equals(reference):
            raise ValueError("Cohort membership, labels or centers differ between outer folds")
        tests.append(frame.loc[frame.split.eq("test"),["sample_id","center_id"]])
    test = pd.concat(tests,ignore_index=True)
    if test.sample_id.duplicated().any() or set(test.sample_id) != set(reference.index):
        raise ValueError("Each participant must occur in outer test exactly once")
    return {"outer_folds":5,"participants":len(reference),"centers":reference.center_id.nunique()}


class ImageDataset(Dataset):
    def __init__(self, frame, shape=(192,192,192), modality=None):
        self.frame = frame.reset_index(drop=True)
        self.shape = tuple(shape)
        self.modality = modality

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        array = np.load(row.cache_path, mmap_mode="r", allow_pickle=False)
        if array.shape != (5,*self.shape) or array.dtype.kind != "f":
            raise ValueError("Expected a floating point cache with shape (5, *configured_shape)")
        slices = {"t1":slice(0,1), "pet":slice(1,2), "dti":slice(2,5)}
        inputs = {}
        for m in ((self.modality,) if self.modality else slices):
            values = np.array(array[slices[m]], dtype=np.float32, copy=True)
            if not np.isfinite(values).all():
                raise ValueError("Image cache contains nonfinite values")
            if m == "t1":
                values *= 3.0
            inputs[m] = torch.from_numpy(values)
        return {**inputs,"label":int(row.get("label",-1)),"sample_id":row.sample_id}
