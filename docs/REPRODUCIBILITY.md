# Reproducibility and scope

## Included pathway

1. Independently train same-modality 3D ResNet-18 teachers with weighted cross-entropy.
2. Initialize three student encoders. Optimize the mean private cross-entropy plus
   mean shared-head cross-entropy and cosine-ramped same-modality feature MSE.
3. Select one common Stage 1 epoch by mean development BAcc of the three private
   heads. Freeze those student encoders; do not combine independently selected epochs.
4. Concatenate the three 512-dimensional frozen features. Train a
   LayerNorm/dropout/1536-to-256/GELU/dropout/256-to-3 MLP.
5. Fit median imputation, standardization and class-balanced multinomial logistic
   regression on the training partition only. Choose C with training-internal
   stratified three-fold CV, not the outer test set.
6. Fit three imaging probability prototypes on the development partition and
   apply MOCEF with fixed beta=1 at inference.

Teachers and classification heads used only in Stage 1 do not supply the final
image probability. One shared head is called on all three modality features.
Its parameters are not duplicated across modalities.

The backbone preserves the source model's unused token parameters to retain its
state-dict layout and initialization behavior. DICEF never uses token features
for fusion, and no token loss is applied. This compatibility detail is not an
additional branch of the proposed method.

## Default optimization

| Component | Settings |
| --- | --- |
| Teachers / Stage 1 | Adam, lr=1e-4, weight decay=1e-5, cosine learning-rate decay |
| Image training | 100 epochs maximum, patience=20, batch=2, accumulation=4, gradient clipping=1 |
| Ramp UMT | 0 through epoch 1; cosine increase to 2.2 at epoch 30; then fixed |
| UMT target | Raw 512-D same-modality features; unweighted elementwise MSE |
| Dropout | Shared head and Stage 2: 0.2 |
| Stage 2 | AdamW, lr=3e-4, weight decay=1e-5, batch=64, 200 epochs maximum, patience=25 |
| Clinical model | L-BFGS, maximum 5000 iterations, inverse-frequency class weights |
| Clinical C grid | 0.01, 0.03, 0.1, 0.3, 1, 3, 10; smaller C wins exact ties |

The source augmentation draws spatial transforms **separately for each modality**.
Within each modality, the teacher and student receive exactly the same augmented
tensor. Spatial parameters are probability 0.8, rotation up to 7 degrees,
translation up to 5 voxels and scale 0.95-1.05. Only T1 additionally receives
intensity/noise perturbation. This describes implementation behavior, not a claim
that the three modalities share the same sampled affine transformation.

Stage 2 uses unaugmented frozen features, stable sample-key ordering and a fresh
seed-42 reset. The training harness uses teacher seed 42, Stage 1 seed `42 + fold`,
and clinical CV seed `42 + 17 * fold`. These defaults make randomization explicit;
they do not replace the original inputs, order, checkpoints and experiment records.

## What this release establishes

The core model definitions and augmentation/cache operations were isolated from
the research implementation. MOCEF and the clinical pipeline use the same
mathematical operations. The public harness replaces machine-specific I/O and
experiment orchestration with explicit input paths and validation boundaries.

Unit tests and an end-to-end synthetic smoke test exercise the released software.
They do **not** reproduce the reported cohort metrics. A full five-fold rerun on
the original research data has not been performed as part of packaging. Exact
replication requires authorized source data, the original preprocessing, matched
participants/visits, folds, training order and compatible software/hardware.
Dataset memberships and trained parameters are deliberately not distributed.

All checkpoint selection uses development performance; outer test is read for
inference only after the models/prototypes are frozen. BAcc is mean class recall;
Macro-F1 is the equally weighted class F1 average. Fold summaries and pooled
prediction metrics are different estimands. `metrics.paired_bootstrap` expects
already aligned probabilities and can resample subjects within class or entire
centers. It does not refit a model.

## Exclusions

No external comparison code, RACF, ordinal hard-rule correction, exploratory
losses, training logs, paper figures, clinical data, pretrained weights or fold
assignments are included. MOCEF is the final probability update; there is no
subsequent MoCA threshold rule in this release.

## Conceptual references

- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- UMT: [Improving Multi-modal Learning with Uni-modal Teachers](https://arxiv.org/abs/2106.11059).
- Shared-head/frozen-fusion context: [UniCross](https://doi.org/10.1007/978-3-032-05182-0_62).

These are scientific references, not additional DICEF code authors. The source
release does not vendor their official repositories.
