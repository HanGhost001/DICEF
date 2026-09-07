# Contributing

This release is maintained by Liangliang Han (GitHub: HanGhost001). Use issues for concise, reproducible
bug reports based on synthetic data. Do not include research data or credentials.

Before proposing a change, run:

```bash
python -m unittest discover -s tests -v
python tools/release_check.py
```

Keep changes limited to the core pathway and preserve the distinction between
training, development and outer test. Changes to augmentation, normalization,
sample ordering, checkpoint selection or randomization can affect reproducibility
and should be documented explicitly.

Code contribution attribution must remain truthful. If external contributions
are accepted later, the maintainer must review authorship and license notices
instead of deleting another contributor's provenance.
