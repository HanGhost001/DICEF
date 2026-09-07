# DICEF

**Dual-stage imaging learning and clinical evidence fusion** for CN/MCI/AD
classification. Code author and maintainer: **Liangliang Han** (GitHub: **HanGhost001**).

DICEF combines T1-weighted MRI, tau PET and three-channel DTI (FA/MD/RD) with
17 clinical features. Three independently trained teachers supervise same-modality
student features during Stage 1, alongside private classifiers and one shared
classification head. Stage 2 freezes the encoders and learns a concatenation MLP.
A class-balanced clinical logistic regression supplies the probability anchor;
MOCEF updates it using development-set imaging class prototypes.

This is a **data-free, portable core release**, not a copy of the experiment archive.
It includes training and inference but no participant records, images, folds,
predictions, checkpoints, fitted clinical parameters or class prototypes.

## Installation

Python 3.11 is the tested interpreter. Install a CPU or CUDA build of PyTorch
appropriate for your machine using the [official installation guide](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m dicef --help
```

`requirements-tested.txt` records the package versions used for release testing;
it does not choose your CUDA driver or PyTorch wheel index. NIfTI cache preparation
is optional: `python -m pip install -e ".[nifti]"`.

## Quick smoke test

```bash
python -m dicef smoke --config configs/dicef.json --work-dir ../dicef-smoke --device cpu
```

Use `--device cuda` for a faster GPU check. This generates random 32-cube arrays
outside the repository and runs all stages with short training schedules. The
network remains ResNet-18; the smoke test uses full precision and a two-epoch ramp.
The small grid and brief schedule are **only a software
test**, not a research experiment. The command uses a fresh work directory and
checks finite, normalized output probabilities after saving and reloading models.

## Train and infer

Prepare your own center-disjoint manifests and registered image caches as described
in [the input specification](docs/INPUTS.md). No cohort is downloaded automatically.

```bash
python -m dicef validate-splits ../private/fold1.csv ../private/fold2.csv ../private/fold3.csv ../private/fold4.csv ../private/fold5.csv
python -m dicef fit --manifest ../private/fold1.csv --config configs/dicef.json --fold 1 --output ../private/run-fold1 --device cuda
python -m dicef predict --manifest ../private/fold1.csv --model-dir ../private/run-fold1 --split test --output ../private/fold1-predictions.csv --device cuda
```

Repeat fitting and final inference independently for folds 2-5. `fit` uses only
training and development images; it does not perform outer-test inference. It trains
the three teachers, selects Stage 1, freezes the student encoders, trains Stage 2,
fits the clinical model on training data, and estimates MOCEF prototypes from
development predictions. `predict` does not fit or select anything.

The fitted directory is **private output**, not part of this source release. Use
only checkpoints you trust, even though loading uses PyTorch's restricted
`weights_only=True` mode. Do not upload fitted artifacts without a separate privacy
and data-use review.

## MOCEF API

```python
from dicef import MOCEF

# Inputs are aligned (N, 3) arrays in CN, MCI, AD order.
model = MOCEF(beta=1.0).fit(development_image_probabilities, development_labels)
q = model.predict_proba(test_clinical_probabilities, test_image_probabilities)
```

Each prototype is a development-class mean. If row `y` of `C` is the prototype for
class `y`, the update is

```text
e_y = -KL(p_I || C_y)
q   = softmax(log(p_C) + beta * p_I @ log(C).T)
```

The second expression is exactly the normalized evidence update: the sample-only
entropy term cancels. This is a prototype-constrained linear logit correction,
not an additional neural classifier. `beta=1` is fixed in the default configuration.

## Repository contents

| Location | Purpose |
| --- | --- |
| `src/dicef/backbone.py`, `models.py` | 3D encoders, one shared head, Stage 2 MLP |
| `src/dicef/losses.py`, `augmentation.py` | Shared/private supervision and Ramp UMT |
| `src/dicef/clinical.py`, `mocef.py` | Clinical classifier and prototype update |
| `src/dicef/data.py`, `normalization.py` | Explicit input contract and cache transforms |
| `src/dicef/training.py`, `cli.py` | Portable training and inference |
| `src/dicef/metrics.py` | BAcc, Macro-F1, recall and paired bootstrap |
| `configs/dicef.json` | Default experiment configuration |
| `tests/`, `tools/release_check.py` | Synthetic tests and release privacy checks |

See [reproducibility notes](docs/REPRODUCIBILITY.md) for the exact included scope,
selection boundaries and differences between a portable harness and the original
experiment environment. External comparison implementations, historical exploratory
branches and visualization tools are intentionally not bundled.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite this software. A manuscript citation and
repository URL can be added after their publication details are confirmed.

MIT License, copyright 2026 Liangliang Han. Dependencies retain their own licenses;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This research software is not
a clinically validated medical device and must not be used as a standalone basis
for diagnosis or treatment.
