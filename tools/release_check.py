"""Conservative source-release scan; never reads files outside this repository."""
import ast
import json
from pathlib import Path
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SUFFIXES = {".py",".md",".json",".toml",".yml",".yaml",".txt",".cff",".typed"}
ALLOWED_NAMES = {"LICENSE",".gitignore",".gitattributes","CODEOWNERS"}
GENERATED = {".git","__pycache__",".pytest_cache",".venv","build","dist"}
PATTERNS = {
    "absolute Windows path":re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/][A-Za-z]"),
    "original-format participant ID":re.compile(r"\b\d{3}_S_\d{4,6}\b"),
    "private key":re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub credential":re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "cloud access key":re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "personal email":re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}"),
    "coauthor trailer":re.compile(r"^Co-authored-by:",re.I|re.M),
}


def check():
    errors,files = [],[]
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if path.is_symlink():
            errors.append(f"Symlink is not allowed: {rel}")
            continue
        if any(part in GENERATED or part.endswith(".egg-info") for part in rel.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix not in ALLOWED_SUFFIXES and path.name not in ALLOWED_NAMES:
            errors.append(f"Unexpected release file type: {rel}")
            continue
        if path.stat().st_size > 250000:
            errors.append(f"Unexpectedly large source file: {rel}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"Non-text release file: {rel}")
            continue
        files.append(str(rel))
        for label,pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append(f"Possible {label}: {rel}")
        if path.suffix == ".py":
            ast.parse(text,filename=str(rel))
        if path.suffix == ".json":
            payload = json.loads(text)
            if isinstance(payload,dict) and any(key in payload for key in (
                "model_state","prototypes","coef","median","cv_scores","outer_test_predictions")):
                errors.append(f"Possible fitted/data-derived artifact: {rel}")
    meta = tomllib.loads((ROOT/"pyproject.toml").read_text())
    for role in ("authors","maintainers"):
        if meta["project"][role] != [{"name":"Liangliang Han"}]:
            errors.append(f"Unexpected {role} metadata")
    if (ROOT/".github/CODEOWNERS").read_text().strip() != "* @HanGhost001":
        errors.append("Unexpected CODEOWNERS")
    if "Copyright (c) 2026 Liangliang Han" not in (ROOT/"LICENSE").read_text():
        errors.append("Unexpected license holder")
    citation = (ROOT/"CITATION.cff").read_text()
    if '  - family-names: "Han"\n    given-names: "Liangliang"' not in citation:
        errors.append("Unexpected software citation author")
    package = ast.parse((ROOT/"src/dicef/__init__.py").read_text())
    authors = [ast.literal_eval(node.value) for node in package.body
               if isinstance(node,ast.Assign) and any(
                   isinstance(target,ast.Name) and target.id == "__author__"
                   for target in node.targets)]
    if authors != ["Liangliang Han"]:
        errors.append("Unexpected package author")
    if errors:
        raise RuntimeError("\n".join(errors))
    print(f"Source release check passed: {len(files)} text files; author=Liangliang Han; license=MIT")
    print("No checked-in data arrays, participant tables, trained artifacts or detected credentials.")
    print("Manually review upload selection; pattern scanning is not proof of anonymity.")


if __name__ == "__main__":
    try:
        check()
    except (ValueError,RuntimeError,SyntaxError) as error:
        print(error,file=sys.stderr)
        sys.exit(1)
