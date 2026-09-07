# Manual GitHub publishing

## Prepared author metadata

The sole code author and maintainer is **Liangliang Han**, as recorded in
`pyproject.toml`, `AUTHORS.md`, `CITATION.cff` and `LICENSE`. The GitHub account
is **HanGhost001**; `CODEOWNERS` and issue assignments use this account handle.
There is no imported Git history and no local commit. No remote repository has
been created and nothing has been uploaded automatically.

## Before upload

1. Run `python -m unittest discover -s tests -v` and `python tools/release_check.py`.
2. Confirm the directory contains source, tests, documentation and configuration
   only. Keep synthetic smoke outputs outside it as well.
3. Create an empty repository using your own GitHub account. Choose a public or
   private visibility deliberately. Do not initialize another conflicting license.
4. Upload the prepared contents. Browser upload may omit hidden dotfiles; verify
   `.github`, `.gitignore` and `.gitattributes` are included.
5. Confirm the displayed commit belongs to HanGhost001. Avoid importing an old
   repository's `.git` folder, adding co-author trailers, or using a bot account.
6. Enable private vulnerability reporting if desired. The included CI workflow
   runs tests; it has read-only repository permissions and creates no commits.
7. Add the actual repository URL to `CITATION.cff` after it exists. Add a paper
   citation only after the title, paper author list and publication details are final.
8. Review the first CI run before creating a `v0.1.0` release.

If using Git instead of browser upload, configure the repository's `user.name`
to `Liangliang Han` and `user.email` to your GitHub-verified or GitHub-provided private
email. The exact email is intentionally not guessed or stored here. Do not alter
global Git identity settings for this purpose.

## Suggested repository description

```text
DICEF: two-stage multimodal imaging learning and prototype-constrained clinical evidence fusion for CN/MCI/AD classification.
```

Suggested topics: `multimodal-learning`, `medical-imaging`, `alzheimers-disease`,
`pytorch`, `knowledge-distillation`.

MIT applies to this source release, not to datasets or model artifacts. Preserving
a sole-author initial release does not authorize removal of third-party notices
or concealment of contributions accepted in future releases.
