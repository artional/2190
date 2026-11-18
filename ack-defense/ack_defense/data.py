import hashlib
import json
import re
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_rows(path):
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty JSON array")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("instruction"), str):
            raise ValueError("Each record needs a string instruction")
        if not row["instruction"].strip():
            raise ValueError("Empty instruction")
        for key in ("input", "output"):
            if key in row and not isinstance(row[key], str):
                raise ValueError(f"{key} must be a string")
    return rows


def identity(row):
    values = [" ".join(re.findall(r"\w+", row.get(k, "").casefold())) for k in ("instruction", "input")]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def identities(rows):
    return [identity(row) for row in rows]


def check_unique(rows):
    keys = identities(rows)
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate base requests")


def check_disjoint(*groups):
    seen = set()
    for group in groups:
        keys = set(group)
        if seen & keys:
            raise ValueError("Base-request overlap across data splits")
        seen.update(keys)
