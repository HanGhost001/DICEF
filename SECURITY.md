# Security and privacy

Do not attach patient/participant records, image headers, medical images, real
manifests, credentials, model checkpoints or prediction files to public issues.
Use generated synthetic inputs and remove machine-specific paths from logs.

Report sensitive problems privately to Liangliang Han (GitHub: HanGhost001) through GitHub's private
vulnerability reporting feature once enabled for the repository. Do not use a
public issue for information that could expose participants or credentials.

Only load trusted local model files. Clinical and prototype states use JSON;
PyTorch checkpoints are loaded with `weights_only=True`. These choices reduce
executable-deserialization exposure but do not make arbitrary artifacts safe.

The release check is a defense-in-depth scan, not a proof of de-identification.
Review the exact files selected for upload. `.gitignore` does not protect files
that were previously tracked, and browser uploads can bypass it.
