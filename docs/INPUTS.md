# Input specification

## Privacy boundary

Supply your own authorized data locally. This repository does not distribute
ADNI data or permission to use it. Keep original identifiers, image headers,
clinical records, split membership, paths, predictions and model artifacts outside
the source repository. Even pseudonymous sample identifiers are not automatically
anonymous.

## One manifest per outer fold

A CSV has one row per participant and these columns:

| Column | Meaning |
| --- | --- |
| `sample_id` | Your local unique sample key, treated only as an alignment key |
| `center_id` | Acquisition-center grouping key, never a model input |
| `split` | `train`, `development`, or `test` |
| `label` | CN=0, MCI=1, AD=2; required for fitting and optional for inference |
| `cache_path` | Path to a five-channel `.npy` array, relative to this CSV or absolute |
| `age` | Age in years |
| `sex_male` | Binary encoding: 1=male, 0=female |
| `moca_total` | MoCA total score |
| `moca_visuospatial_executive` | MoCA visuospatial/executive domain |
| `moca_naming` | Naming domain |
| `moca_attention` | Attention domain |
| `moca_language` | Language domain |
| `moca_abstraction` | Abstraction domain |
| `moca_delayed_free_recall` | Delayed free-recall domain |
| `moca_orientation` | Orientation domain |

The feature order is age, sex, total score, the seven domains in the table order,
then seven domain-missingness indicators in that same order. Indicators are
computed from missing domain values **before** median imputation. There are 17
features, not 17 independent clinical measurements. Age, sex and total score must
be available; missing domain values are represented as empty CSV cells.

Do not include labels, dates, subject keys or acquisition-center information in
the feature vector. Model preprocessing uses only explicitly named clinical columns.
The release rejects a clinical fitting partition with a completely missing feature.

Within a fold, participants and centers must be disjoint across all three splits.
Across five manifests, cohort membership, labels and center IDs must agree, and
each participant must appear in outer test exactly once. Training needs at least
three samples per class for internal three-fold clinical selection; development
needs at least two per class for prototypes. These are software minimums, not
recommendations for adequate research sample size.

## Five-channel imaging cache

The default array shape is `(5, 192, 192, 192)` and dtype is float16. Channel order
is T1, tau PET, FA, MD, RD. The last three channels are parameters from the same
diffusion tensor, not three independent imaging modalities.

Arrays must already share the registered, cropped template grid. This package
does not perform skull stripping, registration, susceptibility correction, tensor
fitting, PET reference-region estimation or participant/visit matching.

Before cache construction, T1 must already be normalized in template space and
PET must already be the intended SUVR-based input. Numeric cache transforms are:

| Channel | Cache transform |
| --- | --- |
| T1 | `clip(T1, -3, 3) / 3` |
| tau PET | Preserve exact zero background; elsewhere `(clip(PET, 0, 2.5) - 1) / 0.5` |
| FA | `clip(FA, 0, 1)` |
| MD, RD | `clip(value, 0, 0.004) / 0.004` |

The imaging loader multiplies the T1 cache channel by 3 after conversion to
float32, preserving the experimental encoder-input convention. **Do not apply
that factor twice** when creating a cache. MD and RD must use the physical units
corresponding to the stated clipping limit, not a vendor-scaled integer map.

For five already prepared NIfTI files:

```bash
python -m dicef prepare-cache --t1 ../private/t1.nii.gz --pet ../private/pet.nii.gz --fa ../private/fa.nii.gz --md ../private/md.nii.gz --rd ../private/rd.nii.gz --output ../private/cache/sample.npy
```

The command validates shape and shared affine, sanitizes nonfinite values to zero
as in the source cache builder, applies numeric transforms and writes float16.
Equal shape/affine does not prove anatomical alignment; that remains an upstream
quality-control requirement. A custom array adapter must preserve channel order,
orientation, physical coordinates and numeric transformations.
