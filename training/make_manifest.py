"""Create the manifest required by uomind_repair_finetune_v3.py."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a corpus shard manifest.")
    parser.add_argument("jsonl")
    args = parser.parse_args()
    path = Path(args.jsonl).resolve()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = {
        "name": path.name,
        "version": "custom-task",
        "sha256": sha256(path),
        "rows": len(rows),
        "preservation_rows": sum(bool(r.get("meta", {}).get("preservation", False)) for r in rows),
        "repair_rows": sum(not bool(r.get("meta", {}).get("preservation", False)) for r in rows),
        "category_counts": dict(sorted(Counter(r.get("meta", {}).get("category", "") for r in rows).items())),
    }
    destination = path.with_name(path.stem + "_manifest.json")
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(destination), **manifest}, indent=2))


if __name__ == "__main__":
    main()
