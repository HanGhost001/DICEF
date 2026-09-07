"""Generate independent random arrays for software smoke tests, not medical evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import DOMAIN_NAMES


def make_synthetic_manifest(directory, shape=(32,32,32), seed=7):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    maxima = [5,3,6,3,2,5,6]
    for split,n in (("train",18),("development",9),("test",9)):
        for i in range(n):
            ident = f"synthetic_{split}_{i:03d}"
            cache = directory / (ident+".npy")
            array = rng.normal(0,0.3,size=(5,*shape)).astype(np.float16)
            array[2:] = np.clip(array[2:],0,1)
            np.save(cache,array,allow_pickle=False)
            row = {"sample_id":ident,"center_id":f"synthetic_{split}_center", "split":split,
                "label":i%3,"cache_path":cache.name,"age":float(rng.uniform(55,85)),
                "sex_male":int(rng.integers(0,2)),"moca_total":float(rng.integers(12,31))}
            for domain,maximum in zip(DOMAIN_NAMES,maxima):
                row[domain] = float(rng.integers(0,maximum+1))
            if i == 0:
                row[DOMAIN_NAMES[0]] = np.nan
            rows.append(row)
    path = directory/"synthetic_manifest.csv"
    pd.DataFrame(rows).to_csv(path,index=False)
    return path
