# Release validation

Validation performed on 2026-09-08 for the prepared 0.1.0 source release:

| Check | Result |
| --- | --- |
| Synthetic unit tests | 14 passed |
| Source-operator parity | Passed using random inputs only |
| ResNet-18 state initialization and forward output | Identical to the extracted source definition |
| Stage 2 MLP forward output | Identical under the same initialized state |
| Stage 1 loss and 100-epoch cosine-ramp values | Identical to source operations |
| Augmentation under the same RNG state | Identical |
| Numeric cache transforms | Identical |
| Clinical C selection and probability output | Identical on the synthetic parity case |
| MOCEF probability difference | Maximum absolute difference below 1e-14 |
| Complete synthetic GPU fit and inference | Passed, including save/load of all fitted components |
| No-test-image fitting guard | Fit succeeded with nonexistent outer-test image paths |
| Nonzero teacher regularization | Exercised using a two-epoch synthetic ramp |
| Source privacy scan | No included data arrays, tables, checkpoints or detected credentials |

Environment: Python 3.11, PyTorch 2.10.0, NumPy 2.3.5, pandas 2.3.3,
SciPy 1.15.3, scikit-learn 1.6.1. The end-to-end smoke used CUDA and full-precision
training on random 32-cube arrays. The default research configuration enables AMP
and uses 192-cube inputs; no full-size cohort retraining was performed for this
release. The smoke's metrics are intentionally not presented as research results.

Operator parity is not a claim of identical long-run optimization across machines
or a substitute for a complete reproduction on the original cohort. The portable
harness adds input checks, restricted loading, explicit paths and training/test
separation; historical experiment orchestration is not included.

Raw audit logs, synthetic arrays and smoke checkpoints are not distributed.
