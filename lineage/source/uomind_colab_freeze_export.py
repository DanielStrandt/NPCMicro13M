#!/usr/bin/env python3
"""
UO-Mind Colab freeze/export utility.

Creates a self-describing ZIP containing the production checkpoint, tokenizer,
runtime/model metadata, source scripts, and evaluation artifacts needed to
freeze a UO-Mind release before the Colab runtime disappears.

Typical Colab usage:
    !python uomind_colab_freeze_export.py --download

Default production checkpoint:
    /content/uomind_tinystories/sft_phase3_relevance_v3/checkpoints/best_balanced.pt

Modes:
    runtime  - smallest useful production/reproducibility bundle (default)
    archive  - also include major lineage checkpoints (larger)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import shutil
import sys
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

FREEZE_FORMAT_VERSION = 1

EXPECTED_ARCHITECTURE = {
    "model_name": "UO-Mind",
    "parameter_count_expected": 13_640_064,
    "vocab_size": 4096,
    "max_seq_len": 128,
    "d_model": 256,
    "n_layers": 16,
    "n_heads": 8,
    "head_dim": 32,
    "d_ff": 1024,
    "activation": "GELU (approximate='tanh')",
    "norm": "RMSNorm, pre-norm",
    "attention": "scaled dot-product attention + fixed relational geometry",
    "positional_embeddings": False,
    "geometry_scales": [2, 4, 8, 16, 32, 64, 128, 256],
    "geometry_formula": "G_ij^(h) = -g_h * |i-j| / tau_h",
    "geometry_gain": {
        "parameterization": "g_h = 4 * sigmoid(a_h)",
        "range": "(0, 4)",
        "initial_value": 1.0,
    },
    "qkv": "fused, biasless",
    "mlp_projections": "biasless",
    "embedding_lm_head": "tied",
}

RUNTIME_CONTRACT = (
    "<BOS><STATE> NPC facts/persona <PLAYER> current player speech "
    "<SAY> NPC response <EOS>"
)

CONTROL_TOKENS = [
    "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<NPC>", "<PLAYER>", "<STATE>",
    "<MEMORY>", "<NOW>", "<ACTION>", "<TONE>", "<SAY>", "<REL>", "<MOOD>",
    "<WORLD>", "<FACT>", "<ITEM>", "<GOLD>", "<QUEST>", "<TARGET>",
    "<LOCATION>", "<FACTION>", "<CRIME>", "<FAME>", "<KARMA>", "<HEALTH>",
    "<END_STATE>",
] + [f"<R{i:02d}>" for i in range(32)]


@dataclass
class BundleEntry:
    archive_path: str
    source_path: str
    bytes: int
    sha256: str
    category: str
    required: bool


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n} B"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.is_file():
            return p.resolve()
    return None


def dedupe_existing(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        if not p.is_file():
            continue
        rp = p.resolve()
        key = str(rp)
        if key not in seen:
            seen.add(key)
            out.append(rp)
    return out


def safe_metadata_files(workspace: Path, max_bytes: int) -> list[Path]:
    """
    Collect small human-readable metadata only.
    Deliberately excludes checkpoints, tensors, caches, and raw datasets.
    """
    roots = [
        workspace,
        workspace / "final_sft",
        workspace / "sft_phase2_grounding",
        workspace / "sft_phase3_relevance_v3",
        workspace / "inference_test",
    ]
    allowed = {".json", ".txt", ".md", ".yaml", ".yml", ".toml"}
    excluded_parts = {
        "checkpoints", "cache", "caches", "__pycache__", "dataset_cache",
        "tinystories_cache", "data",
    }
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        iterator = root.glob("*") if root == workspace else root.rglob("*")
        for p in iterator:
            if not p.is_file() or p.suffix.lower() not in allowed:
                continue
            if any(part.lower() in excluded_parts for part in p.parts):
                continue
            try:
                if p.stat().st_size <= max_bytes:
                    found.append(p)
            except OSError:
                pass
    return dedupe_existing(found)


def source_scripts(content_root: Path) -> list[Path]:
    preferred = [
        content_root / "uomind_colab_inference_test_conversation_v4.py",
        content_root / "uomind_colab_sft_phase3_relevance_v3.py",
        content_root / "uomind_colab_sft_phase2_grounding.py",
        content_root / "uomind_colab_sft.py",
        content_root / "uomind_colab_uo_curriculum.py",
        content_root / "uomind_colab_tinystories_pretrain.py",
    ]
    discovered = sorted(content_root.glob("uomind*.py"))
    return dedupe_existing(preferred + discovered)


def tokenizer_files(workspace: Path) -> list[Path]:
    candidates = [
        workspace / "tokenizer.json",
        workspace / "vocab.json",
        workspace / "merges.txt",
        workspace / "tokenizer_config.json",
        workspace / "special_tokens_map.json",
        workspace / "added_tokens.json",
    ]
    return dedupe_existing(candidates)


def lineage_checkpoints(workspace: Path, production: Path) -> list[Path]:
    candidates = [
        production,
        workspace / "sft_phase3_relevance_v3/checkpoints/best_phase3.pt",
        workspace / "sft_phase3_relevance_v3/checkpoints/final.pt",
        workspace / "sft_phase2_grounding/checkpoints/best_balanced.pt",
        workspace / "final_sft/checkpoints/best_balanced.pt",
        workspace / "final_sft/checkpoints/best_sft.pt",
        workspace / "uo_curriculum/checkpoints/best_uo.pt",
        workspace / "best_uo.pt",
    ]
    return dedupe_existing(candidates)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_readme(
    freeze_name: str,
    checkpoint_name: str,
    tokenizer_name: str,
    mode: str,
) -> str:
    return f"""UO-Mind frozen release bundle
================================

Freeze name: {freeze_name}
Freeze format: {FREEZE_FORMAT_VERSION}
Mode: {mode}

PRIMARY RUNTIME ASSETS
----------------------
Checkpoint:
  model/{checkpoint_name}

Tokenizer:
  tokenizer/{tokenizer_name}

Runtime contract:
  {RUNTIME_CONTRACT}

Canonical architecture:
  13,640,064 parameters
  vocab 4096
  context 128
  d_model 256
  16 transformer layers
  8 attention heads (head_dim 32)
  d_ff 1024
  RMSNorm pre-norm
  GELU tanh approximation
  no positional embeddings
  fixed relational geometry scales:
    2, 4, 8, 16, 32, 64, 128, 256
  bounded trainable geometry gain:
    g = 4 * sigmoid(a)
  tied token embeddings / LM head

IMPORTANT
---------
The production model is a conversational NPC. It does not require an action
policy, trust/debt/danger variables, or hidden game-state numbers.

The game/runtime should provide relevant NPC and world facts through <STATE>.
The current player utterance follows <PLAYER>, then generation begins after
<SAY> and ends at <EOS>.

VERIFYING THE ARCHIVE
---------------------
SHA256SUMS.txt contains SHA-256 hashes for the frozen artifacts.
manifest.json contains source paths, archive paths, sizes, and hashes.

If any production asset changes, create a NEW freeze instead of overwriting
this archive.

SOURCE / EVALUATION
-------------------
When available, source/ contains the Colab modules used to train/test the
model, and evaluation/ contains the latest inference outputs. These are kept
with the release because model weights alone are not sufficient to document
how the model was serialized and invoked.

RESTORING
---------
1. Extract this ZIP.
2. Keep model/ and tokenizer/ together with the included source scripts.
3. Point the conversation inference module at model/{checkpoint_name}.
4. Use tokenizer/{tokenizer_name}.
5. Preserve the runtime contract written above.

For an exact integrity check:
  sha256sum -c SHA256SUMS.txt

This bundle intentionally does NOT include full training datasets or large
caches by default.
"""


def add_source(
    sources: list[tuple[Path, str, str, bool]],
    path: Path,
    archive_path: str,
    category: str,
    required: bool,
) -> None:
    sources.append((path.resolve(), archive_path, category, required))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze/export a UO-Mind Colab release.")
    p.add_argument(
        "--workspace",
        default="/content/uomind_tinystories",
        help="UO-Mind workspace root.",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Production checkpoint. Defaults to revised Phase-3 best_balanced.pt.",
    )
    p.add_argument(
        "--content-root",
        default="/content",
        help="Directory containing UO-Mind Colab source modules.",
    )
    p.add_argument(
        "--output-dir",
        default="/content",
        help="Where the ZIP and staging directory are created.",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Freeze/release name. Default includes UTC timestamp.",
    )
    p.add_argument(
        "--mode",
        choices=["runtime", "archive"],
        default="runtime",
        help="runtime=production essentials; archive=also major lineage checkpoints.",
    )
    p.add_argument(
        "--metadata-max-mb",
        type=float,
        default=5.0,
        help="Maximum size of an individual metadata text/JSON file.",
    )
    p.add_argument("--no-scripts", action="store_true", help="Do not include source .py files.")
    p.add_argument("--no-evaluation", action="store_true", help="Do not include inference results.")
    p.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep the uncompressed staging directory after ZIP creation.",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="In Google Colab, automatically trigger browser download of the ZIP.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    content_root = Path(args.content_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not workspace.exists():
        raise SystemExit(f"Workspace does not exist: {workspace}")

    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser()
        if not checkpoint.is_file():
            raise SystemExit(f"Checkpoint not found: {checkpoint}")
        checkpoint = checkpoint.resolve()
    else:
        checkpoint = first_existing([
            workspace / "sft_phase3_relevance_v3/checkpoints/best_balanced.pt",
            workspace / "sft_phase3_relevance_v3/checkpoints/best_phase3.pt",
            workspace / "sft_phase3_relevance_v3/checkpoints/final.pt",
            workspace / "sft_phase2_grounding/checkpoints/best_balanced.pt",
        ])
        if checkpoint is None:
            raise SystemExit(
                "Could not find a production checkpoint. "
                "Pass --checkpoint /path/to/model.pt"
            )

    toks = tokenizer_files(workspace)
    tokenizer = first_existing([workspace / "tokenizer.json"] + toks)
    if tokenizer is None:
        raise SystemExit(
            f"Could not find tokenizer.json or tokenizer assets under {workspace}"
        )

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    freeze_name = args.name or f"UO-Mind-phase3-v3-freeze-{timestamp}"
    freeze_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in freeze_name)

    stage = output_dir / freeze_name
    zip_path = output_dir / f"{freeze_name}.zip"

    if stage.exists():
        shutil.rmtree(stage)
    if zip_path.exists():
        zip_path.unlink()
    stage.mkdir(parents=True)

    sources: list[tuple[Path, str, str, bool]] = []

    add_source(
        sources,
        checkpoint,
        f"model/{checkpoint.name}",
        "production_checkpoint",
        True,
    )

    for p in toks:
        add_source(
            sources,
            p,
            f"tokenizer/{p.name}",
            "tokenizer",
            p == tokenizer,
        )

    if args.mode == "archive":
        for p in lineage_checkpoints(workspace, checkpoint):
            if p == checkpoint:
                continue
            try:
                rel = p.relative_to(workspace)
                archive_path = f"lineage/{rel.as_posix()}"
            except ValueError:
                archive_path = f"lineage/{p.name}"
            add_source(sources, p, archive_path, "lineage_checkpoint", False)

    if not args.no_scripts:
        for p in source_scripts(content_root):
            add_source(sources, p, f"source/{p.name}", "source_code", False)

    metadata = safe_metadata_files(
        workspace, max_bytes=int(args.metadata_max_mb * 1024 * 1024)
    )
    for p in metadata:
        if "inference_test" in p.parts and not args.no_evaluation:
            add_source(
                sources,
                p,
                f"evaluation/{p.name}",
                "evaluation",
                False,
            )
        elif "inference_test" not in p.parts:
            try:
                rel = p.relative_to(workspace)
                ap = f"metadata/{rel.as_posix()}"
            except ValueError:
                ap = f"metadata/{p.name}"
            add_source(sources, p, ap, "metadata", False)

    unique: list[tuple[Path, str, str, bool]] = []
    seen_src: set[str] = set()
    seen_arc: set[str] = set()
    for src, arc, cat, req in sources:
        skey = str(src)
        if skey in seen_src or arc in seen_arc:
            continue
        seen_src.add(skey)
        seen_arc.add(arc)
        unique.append((src, arc, cat, req))
    sources = unique

    print("=" * 78)
    print("UO-MIND FREEZE / EXPORT")
    print("=" * 78)
    print(f"workspace:   {workspace}")
    print(f"checkpoint:  {checkpoint}")
    print(f"tokenizer:   {tokenizer}")
    print(f"mode:        {args.mode}")
    print(f"stage:       {stage}")
    print(f"zip:         {zip_path}")
    print()

    entries: list[BundleEntry] = []

    for i, (src, arc, category, required) in enumerate(sources, 1):
        dest = stage / arc
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i:02d}/{len(sources):02d}] {category:22s} {src.name}")
        shutil.copy2(src, dest)
        digest = sha256_file(dest)
        size = dest.stat().st_size
        entries.append(BundleEntry(
            archive_path=arc,
            source_path=str(src),
            bytes=size,
            sha256=digest,
            category=category,
            required=required,
        ))

    runtime_info = {
        "freeze_format_version": FREEZE_FORMAT_VERSION,
        "freeze_name": freeze_name,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "mode": args.mode,
        "runtime_contract": RUNTIME_CONTRACT,
        "production_checkpoint_archive_path": f"model/{checkpoint.name}",
        "tokenizer_archive_path": f"tokenizer/{tokenizer.name}",
        "architecture": EXPECTED_ARCHITECTURE,
        "reserved_control_tokens": CONTROL_TOKENS,
        "production_notes": {
            "role": "single-turn conversational NPC",
            "action_policy_required": False,
            "hidden_numeric_state_required": False,
            "world_facts_delivery": "provide relevant facts in <STATE>",
            "generation_start": "<SAY>",
            "generation_stop": "<EOS>",
        },
        "host_at_freeze": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "files": [asdict(e) for e in entries],
    }

    write_json(stage / "manifest.json", runtime_info)
    write_json(stage / "model" / "model_config.json", EXPECTED_ARCHITECTURE)
    write_json(
        stage / "tokenizer" / "tokenizer_freeze_info.json",
        {
            "expected_vocab_size": 4096,
            "primary_tokenizer_file": tokenizer.name,
            "byte_fallback": True,
            "reserved_control_tokens": CONTROL_TOKENS,
        },
    )
    (stage / "runtime_contract.txt").write_text(
        RUNTIME_CONTRACT + "\n", encoding="utf-8"
    )
    (stage / "README_FREEZE.txt").write_text(
        make_readme(freeze_name, checkpoint.name, tokenizer.name, args.mode),
        encoding="utf-8",
    )

    generated_paths = [
        stage / "manifest.json",
        stage / "model" / "model_config.json",
        stage / "tokenizer" / "tokenizer_freeze_info.json",
        stage / "runtime_contract.txt",
        stage / "README_FREEZE.txt",
    ]

    sha_lines: list[str] = []
    for p in sorted(
        [stage / e.archive_path for e in entries] + generated_paths,
        key=lambda x: x.relative_to(stage).as_posix(),
    ):
        rel = p.relative_to(stage).as_posix()
        sha_lines.append(f"{sha256_file(p)}  {rel}")
    (stage / "SHA256SUMS.txt").write_text(
        "\n".join(sha_lines) + "\n", encoding="utf-8"
    )

    inventory_lines = [
        f"UO-Mind freeze: {freeze_name}",
        f"Created UTC: {_dt.datetime.now(_dt.timezone.utc).isoformat()}",
        f"Mode: {args.mode}",
        "",
        "Frozen source artifacts:",
    ]
    for e in entries:
        inventory_lines.append(
            f"  {human_bytes(e.bytes):>11s}  {e.sha256[:16]}...  {e.archive_path}"
        )
    (stage / "BUNDLE_INVENTORY.txt").write_text(
        "\n".join(inventory_lines) + "\n", encoding="utf-8"
    )

    all_files = sorted(p for p in stage.rglob("*") if p.is_file())
    print()
    print(f"Creating ZIP with {len(all_files)} files...")
    with zipfile.ZipFile(zip_path, "w", allowZip64=True) as zf:
        for p in all_files:
            rel = p.relative_to(stage).as_posix()
            compression = (
                zipfile.ZIP_STORED
                if p.suffix.lower() in {".pt", ".bin", ".safetensors"}
                else zipfile.ZIP_DEFLATED
            )
            zf.write(
                p,
                arcname=f"{freeze_name}/{rel}",
                compress_type=compression,
            )

    zip_sha = sha256_file(zip_path)
    zip_size = zip_path.stat().st_size

    print()
    print("=" * 78)
    print("FREEZE COMPLETE")
    print("=" * 78)
    print(f"ZIP:        {zip_path}")
    print(f"size:       {human_bytes(zip_size)}")
    print(f"sha256:     {zip_sha}")
    print(f"checkpoint: {checkpoint.name}")
    print(f"tokenizer:  {tokenizer.name}")
    print()
    print("Keep the ZIP SHA-256 somewhere outside Colab:")
    print(zip_sha)
    print()

    if not args.keep_staging:
        shutil.rmtree(stage)
        print("Staging directory removed; ZIP retained.")
    else:
        print(f"Staging directory retained: {stage}")

    if args.download:
        try:
            from google.colab import files as colab_files
            print("Triggering Colab browser download...")
            colab_files.download(str(zip_path))
        except Exception as exc:
            print(f"Could not trigger automatic Colab download: {exc}")
            print("Run this in a Colab cell instead:")
            print(f"from google.colab import files; files.download({str(zip_path)!r})")
    else:
        print()
        print("To download from Colab, either rerun with --download:")
        print(f"  !python {Path(__file__).name} --download")
        print("or run:")
        print(f"  from google.colab import files; files.download({str(zip_path)!r})")


if __name__ == "__main__":
    main()
