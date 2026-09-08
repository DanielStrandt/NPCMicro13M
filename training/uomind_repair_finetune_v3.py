#!/usr/bin/env python3
"""
UO-Mind Flexible Preservation-First Repair Fine-Tuner V3
=========================================================

Purpose
-------
Fine-tune the frozen Phase-3-v3 UO-Mind checkpoint on any number of static
preservation/repair corpus shards while minimizing regression.

Corpus discovery
----------------
By default the trainer scans:

  <bundle>/repaircorpus/uomind_preservation_repair_corpus_part*.jsonl

So part1, part2, part3, part4, future part5/part6/etc. are picked up
automatically. Companion *_manifest.json files are verified whenever present.
Use --require-manifests to make manifests mandatory, or --corpus-glob to use a
different naming convention.

Default strategy
----------------
* Start from the frozen production Phase-3-v3 checkpoint.
* Auto-discover every matching corpus shard; row totals are inferred.
* Reject duplicate full rows and duplicate STATE+PLAYER pairs by default.
* Grouped 80/10/10 split; pair_id/family siblings never cross splits.
* Default effective batch = 19 repair + 13 preservation, both adjustable.
* Default trainable scope = final 4 transformer blocks + final RMSNorm.
* Tied embedding/LM-head frozen by default; enable with --train-embedding.
* Geometry gains frozen by default; enable with --train-geometry.
* Default core LR = 2e-6 -> 2e-7, fully adjustable.
* Cosine/linear/constant LR schedules are selectable.
* One epoch by default; --epochs is adjustable.
* Query-aware STATE packing is shared by training and inference probes.
* Quick behavior/validation eval every 5 steps by default (monitoring only).
* Full 56-prompt repair behavior eval every 15 steps and at the end.
* Only full-suite evaluations may rank or save behavioral checkpoints.
* Final holdout is never used for checkpoint selection.
* Preservation is judged relative to the frozen source, not absolute perfection.
* final_unpromoted.pt is always saved for diagnosis.

Recommended Windows commands
----------------------------
Smoke test with defaults:
  python uomind_repair_finetune_v3.py --bundle . --smoke-test --device cpu

Default selective top-4 run:
  python uomind_repair_finetune_v3.py --bundle . --train --device cpu

Adjust LR without editing code:
  python uomind_repair_finetune_v3.py --bundle . --train --device cpu --max-lr 1.5e-6 --min-lr 1.5e-7

Train all transformer blocks instead of only the top four:
  python uomind_repair_finetune_v3.py --bundle . --train --device cpu --train-last-n-layers 0

Train all blocks plus tied embedding/LM head:
  python uomind_repair_finetune_v3.py --bundle . --train --device cpu --train-last-n-layers 0 --train-embedding

Use a custom corpus pattern:
  python uomind_repair_finetune_v3.py --bundle . --train --corpus-glob "repair_*.jsonl"

Output
------
By default output is written beside the extracted frozen bundle:
  ../uomind_repair_finetune_v3/

Important
---------
This script never modifies the frozen bundle or corpus files.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tokenizers import Tokenizer
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'tokenizers'. Install with:\n"
        "  python -m pip install tokenizers"
    ) from exc


EXPECTED_PARAMETER_COUNT = 13_640_064
EXPECTED_SOURCE_SHA256 = "eb0c2538a253e72158192f5004c941e19798febe987e35fbdbc71f19a0e7a170"

RUNTIME_CONTRACT = (
    "<BOS><STATE> NPC facts/persona <PLAYER> current player speech "
    "<SAY> NPC response <EOS>"
)

CONTROL_MARKERS = (
    "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<NPC>", "<PLAYER>", "<STATE>",
    "<MEMORY>", "<NOW>", "<ACTION>", "<TONE>", "<SAY>", "<REL>", "<MOOD>",
    "<WORLD>", "<FACT>", "<ITEM>", "<GOLD>", "<QUEST>", "<TARGET>",
    "<LOCATION>", "<FACTION>", "<CRIME>", "<FAME>", "<KARMA>", "<HEALTH>",
    "<END_STATE>"
)

FALSE_REFERRAL_PHRASES = (
    "ask a farmer", "seek a farmer",
    "ask a blacksmith", "seek a blacksmith",
    "ask a baker", "seek a baker",
    "ask a tanner", "seek a tanner",
    "ask a gardener", "seek a gardener",
    "ask a healer", "seek a healer",
    "ask a merchant", "seek a merchant",
    "ask a shepherd", "seek a shepherd",
    "ask a mason", "seek a mason",
    "not my trade", "not my craft",
    "outside my trade", "outside my craft",
    "would know that better than i",
    "would know better than i",
    "i know little of that craft",
    "i know little of that work",
)


# =============================================================================
# Exact UO-Mind architecture
# =============================================================================

@dataclass
class ModelConfig:
    vocab_size: int = 4096
    max_seq_len: int = 128
    d_model: int = 256
    n_layers: int = 16
    n_heads: int = 8
    d_ff: int = 1024
    norm_eps: float = 1e-6
    geometry_scales: Tuple[float, ...] = (2., 4., 8., 16., 32., 64., 128., 256.)
    geometry_gain_init: float = 1.0
    geometry_gain_max: float = 4.0
    init_std: float = 0.02
    attention_backend: str = "sdpa"

    def __post_init__(self):
        if isinstance(self.geometry_scales, list):
            self.geometry_scales = tuple(float(x) for x in self.geometry_scales)
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if len(self.geometry_scales) != self.n_heads:
            raise ValueError("one geometry scale per head is required")

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return xf.to(dtype) * self.weight


class GeometryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.max_seq_len = cfg.max_seq_len
        self.backend = cfg.attention_backend
        self.gain_max = float(cfg.geometry_gain_max)

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        scales = torch.tensor(cfg.geometry_scales, dtype=torch.float32)
        pos = torch.arange(cfg.max_seq_len, dtype=torch.float32)
        dist = torch.abs(pos[:, None] - pos[None, :])
        base = (-dist.unsqueeze(0) / scales[:, None, None]).to(torch.float16)
        self.register_buffer("base_geometry", base.unsqueeze(0), persistent=True)

        causal = torch.zeros(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.float16)
        causal = causal.masked_fill(
            torch.triu(torch.ones_like(causal, dtype=torch.bool), diagonal=1),
            float("-inf"),
        )
        self.register_buffer("causal_bias", causal[None, None], persistent=False)

        ratio = cfg.geometry_gain_init / cfg.geometry_gain_max
        raw_init = math.log(ratio / (1.0 - ratio))
        self.raw_gain = nn.Parameter(torch.full((cfg.n_heads,), raw_init))

    def gains(self):
        return self.gain_max * torch.sigmoid(self.raw_gain)

    def forward(self, x):
        b, t, _ = x.shape
        if t != self.max_seq_len:
            raise ValueError(f"Expected sequence length {self.max_seq_len}, got {t}")

        qkv = self.qkv_proj(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        gain = self.gains().to(q.dtype).view(1, self.n_heads, 1, 1)
        bias = gain * self.base_geometry.to(q.dtype) + self.causal_bias.to(q.dtype)

        if self.backend == "sdpa":
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=bias,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + bias
            y = torch.softmax(scores, dim=-1) @ v

        y = y.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GeometryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class UOMindLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    def num_parameters(self):
        total, seen = 0, set()
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    def geometry_summary(self):
        g = torch.stack([b.attn.gains() for b in self.blocks])
        return {
            "mean": float(g.mean().detach().cpu()),
            "min": float(g.min().detach().cpu()),
            "max": float(g.max().detach().cpu()),
        }


# =============================================================================
# General utilities
# =============================================================================

def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def normalize_text(s):
    return " ".join(str(s).replace("\r", " ").replace("\n", " ").split()).strip()


def looks_like_bundle(p: Path):
    return p.is_dir() and (p / "model").is_dir() and (p / "tokenizer" / "tokenizer.json").is_file()


def discover_bundle(start: Path):
    start = start.expanduser().resolve()
    if looks_like_bundle(start):
        return start
    if not start.is_dir():
        raise SystemExit(f"Bundle directory does not exist: {start}")
    direct = [p for p in start.iterdir() if p.is_dir() and looks_like_bundle(p)]
    if len(direct) == 1:
        return direct[0].resolve()
    raise SystemExit(f"Could not uniquely locate extracted UO-Mind bundle under {start}")


def discover_source_checkpoint(bundle: Path, override: Optional[str]):
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"Checkpoint not found: {p}")
        return p

    manifest = bundle / "manifest.json"
    if manifest.is_file():
        try:
            obj = json.loads(manifest.read_text(encoding="utf-8"))
            rel = obj.get("production_checkpoint_archive_path")
            if rel and (bundle / rel).is_file():
                return (bundle / rel).resolve()
        except Exception:
            pass

    p = bundle / "model" / "best_balanced.pt"
    if p.is_file():
        return p.resolve()
    raise SystemExit("Could not find frozen Phase-3 production checkpoint.")


def choose_device(requested):
    if requested != "auto":
        d = torch.device(requested)
        if d.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable.")
        return d
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(device, requested):
    if requested == "fp32":
        return torch.float32
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def autocast_context(device, dtype):
    if device.type in {"cuda", "mps"} and dtype in {torch.float16, torch.bfloat16}:
        try:
            return torch.autocast(device_type=device.type, dtype=dtype)
        except Exception:
            pass
    return contextlib.nullcontext()


def marker(tok, name):
    x = tok.token_to_id(name)
    if x is None:
        raise RuntimeError(f"Tokenizer missing marker {name}")
    return int(x)


def encode_content(tok, text):
    return tok.encode(" " + normalize_text(text)).ids


def validate_tokenizer(tok):
    if tok.get_vocab_size() != 4096:
        raise RuntimeError(f"Tokenizer vocab is {tok.get_vocab_size()}, expected 4096")
    for name in ("<PAD>", "<BOS>", "<EOS>", "<STATE>", "<PLAYER>", "<SAY>"):
        mid = marker(tok, name)
        if tok.encode(name).ids != [mid]:
            raise RuntimeError(f"Tokenizer marker {name} is not atomic.")


def load_model(checkpoint, tok, device, attention_backend=None):
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")

    if "model_config" not in payload or "model_state_dict" not in payload:
        raise RuntimeError("Checkpoint does not have expected UO-Mind keys.")

    cfg_dict = dict(payload["model_config"])
    if attention_backend:
        cfg_dict["attention_backend"] = attention_backend
    cfg = ModelConfig(**cfg_dict)

    if cfg.vocab_size != tok.get_vocab_size():
        raise RuntimeError("Checkpoint/tokenizer vocabulary mismatch.")

    model = UOMindLM(cfg)
    inc = model.load_state_dict(payload["model_state_dict"], strict=False)
    missing = [k for k in inc.missing_keys if "causal_bias" not in k]
    if missing or inc.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={missing}, unexpected={inc.unexpected_keys}"
        )
    if model.num_parameters() != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Parameter count {model.num_parameters():,}; expected {EXPECTED_PARAMETER_COUNT:,}"
        )

    model.to(device)
    return model, payload, cfg


def json_dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


# =============================================================================
# Corpus loading / audit
# =============================================================================

@dataclass(frozen=True)
class CorpusRow:
    uid: str
    shard: str
    category: str
    family: str
    pair_id: Optional[str]
    preservation: bool
    system: str
    user: str
    assistant: str

    @property
    def split_group(self):
        # pair_id has highest priority. Otherwise keep semantic family siblings
        # together inside the shard/category so paraphrase variants do not leak.
        if self.pair_id:
            return f"pair::{self.shard}::{self.category}::{self.pair_id}"
        return f"family::{self.shard}::{self.category}::{self.family}"


def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.name.casefold())
    return [int(x) if x.isdigit() else x for x in parts]


def find_corpus_files(corpus_dir: Path, corpus_glob: str, require_manifests: bool = False):
    """Discover any number of JSONL shards.

    Default pattern is ``uomind_preservation_repair_corpus_part*.jsonl``.
    Future part5/part6/etc. files are picked up automatically. A companion
    ``<stem>_manifest.json`` is verified whenever it exists.
    """
    corpus_dir = corpus_dir.expanduser().resolve()
    if not corpus_dir.is_dir():
        raise SystemExit(f"Corpus directory not found: {corpus_dir}")

    jsonls = sorted(corpus_dir.glob(corpus_glob), key=natural_sort_key)
    jsonls = [p for p in jsonls if p.is_file()]
    if not jsonls:
        raise SystemExit(
            f"No corpus JSONL files matched {corpus_glob!r} under {corpus_dir}"
        )

    parts = []
    for ordinal, j in enumerate(jsonls, 1):
        m = j.with_name(j.stem + "_manifest.json")
        if not m.is_file():
            if require_manifests:
                raise SystemExit(f"Missing required corpus manifest: {m}")
            m = None

        match = re.search(r"part[_-]?(\d+)", j.stem, flags=re.I)
        shard = match.group(1) if match else f"file{ordinal}"
        parts.append((shard, j.resolve(), m.resolve() if m else None))
    return parts


def load_corpus(
    corpus_dir: Path,
    corpus_glob: str,
    allow_mismatch: bool = False,
    require_manifests: bool = False,
):
    all_rows = []
    shard_audit = []

    for shard, jsonl_path, manifest_path in find_corpus_files(
        corpus_dir, corpus_glob, require_manifests
    ):
        actual_sha = sha256_file(jsonl_path)
        manifest = None

        if manifest_path is not None:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_sha = manifest.get("sha256")
            if expected_sha and expected_sha != actual_sha:
                msg = (
                    f"Shard {shard} hash mismatch:\n"
                    f"  manifest: {expected_sha}\n"
                    f"  actual:   {actual_sha}"
                )
                if allow_mismatch:
                    print("WARNING:", msg)
                else:
                    raise SystemExit(msg)

        raw_lines = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if manifest and manifest.get("rows") is not None and int(manifest["rows"]) != len(raw_lines):
            msg = f"Shard {shard} manifest says {manifest['rows']} rows, found {len(raw_lines)}"
            if allow_mismatch:
                print("WARNING:", msg)
            else:
                raise RuntimeError(msg)

        preserve = 0
        repair = 0

        for line_no, obj in enumerate(raw_lines, 1):
            msgs = obj.get("messages")
            md = obj.get("meta", {})
            if not isinstance(msgs, list) or len(msgs) != 3:
                raise RuntimeError(f"Shard {shard} line {line_no}: expected 3 messages")
            if [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
                raise RuntimeError(
                    f"Shard {shard} line {line_no}: role order must be system/user/assistant"
                )

            category = str(md.get("category", "")).strip()
            family = str(md.get("family", "")).strip()
            if not category or not family:
                raise RuntimeError(
                    f"Shard {shard} line {line_no}: missing category/family metadata"
                )

            is_preserve = bool(md.get("preservation", False))
            preserve += int(is_preserve)
            repair += int(not is_preserve)

            all_rows.append(CorpusRow(
                uid=f"s{shard}:{line_no}",
                shard=str(shard),
                category=category,
                family=family,
                pair_id=md.get("pair_id"),
                preservation=is_preserve,
                system=normalize_text(msgs[0]["content"]),
                user=normalize_text(msgs[1]["content"]),
                assistant=normalize_text(msgs[2]["content"]),
            ))

        shard_audit.append({
            "shard": str(shard),
            "jsonl": str(jsonl_path),
            "manifest": str(manifest_path) if manifest_path else None,
            "manifest_present": manifest_path is not None,
            "sha256": actual_sha,
            "rows": len(raw_lines),
            "preservation": preserve,
            "repair": repair,
        })

    full_keys = [(r.system, r.user, r.assistant) for r in all_rows]
    pair_keys = [(r.system, r.user) for r in all_rows]
    dup_full = len(full_keys) - len(set(full_keys))
    dup_pair = len(pair_keys) - len(set(pair_keys))

    preserve_total = sum(r.preservation for r in all_rows)
    repair_total = len(all_rows) - preserve_total

    if not allow_mismatch:
        if dup_full:
            raise SystemExit(f"Found {dup_full} duplicate full rows across discovered shards")
        if dup_pair:
            raise SystemExit(f"Found {dup_pair} duplicate STATE+PLAYER pairs across discovered shards")
        if preserve_total == 0 or repair_total == 0:
            raise SystemExit("Corpus must contain both preservation and repair rows")

    audit = {
        "corpus_glob": corpus_glob,
        "discovered_shards": len(shard_audit),
        "total_rows": len(all_rows),
        "preservation_rows": preserve_total,
        "repair_rows": repair_total,
        "preservation_fraction": preserve_total / max(1, len(all_rows)),
        "repair_fraction": repair_total / max(1, len(all_rows)),
        "duplicate_full_rows": dup_full,
        "duplicate_state_player_pairs": dup_pair,
        "categories": dict(Counter(r.category for r in all_rows)),
        "shards": shard_audit,
    }
    return all_rows, audit


# =============================================================================
# Grouped split
# =============================================================================

def grouped_split(rows: Sequence[CorpusRow], seed: int):
    """
    Grouped 80/10/10 split, stratified by preservation flag and category.

    Groups are semantic families / pair groups, not individual rows.
    The resulting counts can differ slightly from exact 80/10/10 because
    entire groups stay together.
    """
    strata = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = ("preserve" if r.preservation else "repair", r.category)
        strata[key][r.split_group].append(r)

    train, val, hold = [], [], []
    details = {}

    for stratum_key in sorted(strata):
        groups = list(strata[stratum_key].items())
        local_seed = seed + sum(ord(c) for c in "::".join(stratum_key)) * 1009
        random.Random(local_seed).shuffle(groups)

        total_rows = sum(len(rr) for _, rr in groups)
        target = {
            "train": 0.80 * total_rows,
            "val": 0.10 * total_rows,
            "hold": 0.10 * total_rows,
        }
        assigned = {"train": 0, "val": 0, "hold": 0}
        buckets = {"train": [], "val": [], "hold": []}

        # Largest semantic groups first after randomized tie ordering.
        groups.sort(key=lambda x: len(x[1]), reverse=True)

        for gid, rr in groups:
            # Choose split with lowest relative fill against target.
            def fill(split):
                return assigned[split] / max(target[split], 1e-9)
            split = min(("train", "val", "hold"), key=fill)
            buckets[split].extend(rr)
            assigned[split] += len(rr)

        train.extend(buckets["train"])
        val.extend(buckets["val"])
        hold.extend(buckets["hold"])

        details["::".join(stratum_key)] = {
            "total_rows": total_rows,
            "groups": len(groups),
            "train_rows": len(buckets["train"]),
            "val_rows": len(buckets["val"]),
            "holdout_rows": len(buckets["hold"]),
        }

    # Isolation assertion.
    tr = {r.split_group for r in train}
    va = {r.split_group for r in val}
    ho = {r.split_group for r in hold}
    if tr & va or tr & ho or va & ho:
        raise RuntimeError("Semantic group leakage detected across splits")

    random.Random(seed + 1).shuffle(train)
    random.Random(seed + 2).shuffle(val)
    random.Random(seed + 3).shuffle(hold)
    return train, val, hold, details


# =============================================================================
# Serialization
# =============================================================================

QUERY_STOPWORDS = {
    "a","an","the","i","me","my","mine","you","your","yours","we","our","ours",
    "is","are","was","were","be","been","being","do","does","did","can","could",
    "should","would","will","what","which","who","whom","whose","where","when","why",
    "how","this","that","these","those","it","its","to","of","for","from","in","on",
    "at","by","with","and","or","but","if","then","today","now","please","tell","know"
}

NUMBER_WORDS = {
    "one","two","three","four","five","six","seven","eight","nine","ten","eleven",
    "twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty"
}


def lexical_terms(text):
    return {
        w for w in re.findall(r"[a-z0-9']+", normalize_text(text).casefold())
        if len(w) > 1 and w not in QUERY_STOPWORDS
    }


def sentence_relevance(sentence, query):
    """Runtime-safe relevance score using only STATE text + current PLAYER query."""
    s = normalize_text(sentence).casefold()
    q = normalize_text(query).casefold()
    st = lexical_terms(s)
    qt = lexical_terms(q)
    score = 5.0 * len(st & qt)

    # Preserve values/relations when the question shape tells us what kind of
    # STATE sentence is likely to matter, even when synonyms differ (loaf/bread).
    if any(x in q for x in ("how much", "price", "cost", "copper", "afford", "enough coin")):
        if "copper" in s or re.search(r"\\b\\d+\\b", s) or (st & NUMBER_WORDS):
            score += 7.0
        if any(x in s for x in ("cost", "price", "sells", "charges", "fee")):
            score += 4.0
    if "when" in q or "what time" in q or "what hour" in q:
        if any(x in s for x in ("dawn", "noon", "sundown", "bell", "morning", "evening", "night", "tomorrow")):
            score += 7.0
    if "where" in q or "which way" in q or "road" in q or "route" in q:
        if any(x in s for x in (" at ", " by ", " near ", " beside ", " behind ", " under ", " above ",
                                  "road", "gate", "bridge", "ferry", "market", "bank", "shrine", "well", "inn")):
            score += 5.0
    if "who" in q or "whose" in q:
        # Proper-name-bearing relation sentences are often the answer source.
        if re.search(r"\\b[A-Z][a-z]+\\b", sentence):
            score += 4.0
        if any(x in s for x in ("belongs", "owns", "named", "name is", "led by", "guarded by")):
            score += 4.0
    if any(x in q for x in ("open", "closed", "blocked", "safe", "danger", "watch for")):
        if any(x in s for x in ("open", "closed", "blocked", "flooded", "danger", "undead", "spider", "trap")):
            score += 5.0

    return score


def pack_state(tok, text, max_system_tokens, query=""):
    sid = marker(tok, "<STATE>")
    full = [sid] + encode_content(tok, text)
    if len(full) <= max_system_tokens:
        return full, False

    sentences = [
        normalize_text(x)
        for x in re.split(r"(?<=[.!?])\\s+", normalize_text(text))
        if normalize_text(x)
    ]
    if not sentences:
        return full[:max_system_tokens], True

    # First sentence normally carries the NPC name and is kept first. Remaining
    # sentences are ranked by relevance to the current PLAYER utterance. The
    # profession/persona sentence gets a small bonus, but a query-relevant fact
    # outranks an unrelated trailing distractor. This exact logic is used by
    # both training serialization and runtime probe generation.
    priorities = []
    for idx, sentence in enumerate(sentences):
        score = sentence_relevance(sentence, query)
        if idx == 0:
            score += 100.0
        elif idx == 1:
            score += 2.0
        priorities.append((score, -idx, idx))
    priorities.sort(reverse=True)

    chosen = []
    for _, _, idx in priorities:
        trial_idx = sorted(chosen + [idx])
        trial = " ".join(sentences[j] for j in trial_idx)
        ids = [sid] + encode_content(tok, trial)
        if len(ids) <= max_system_tokens:
            chosen.append(idx)

    if chosen:
        packed = " ".join(sentences[j] for j in sorted(chosen))
        return [sid] + encode_content(tok, packed), True

    return full[:max_system_tokens], True

def serialize_row(tok, row, seq_len, max_system_tokens, max_target_tokens):
    bos = marker(tok, "<BOS>")
    eos = marker(tok, "<EOS>")
    player_id = marker(tok, "<PLAYER>")
    say = marker(tok, "<SAY>")
    pad = marker(tok, "<PAD>")

    state_tokens, repacked = pack_state(tok, row.system, max_system_tokens, row.user)
    player_tokens = [player_id] + encode_content(tok, row.user)
    target = encode_content(tok, row.assistant) + [eos]

    if len(target) > max_target_tokens:
        raise RuntimeError(
            f"{row.uid}: target has {len(target)} tokens, exceeds max {max_target_tokens}"
        )

    prompt = [bos] + state_tokens + player_tokens + [say]
    max_full = seq_len + 1

    if len(prompt) + len(target) > max_full:
        overflow = len(prompt) + len(target) - max_full
        removable = max(0, len(player_tokens) - 1)
        take = min(overflow, removable)
        if take:
            player_tokens = [player_tokens[0]] + player_tokens[1 + take:]
            prompt = [bos] + state_tokens + player_tokens + [say]

    if len(prompt) + len(target) > max_full:
        raise RuntimeError(
            f"{row.uid}: cannot fit sequence safely in {seq_len} tokens"
        )

    full = prompt + target
    inp = full[:-1]

    x = torch.full((seq_len,), pad, dtype=torch.long)
    y = torch.full((seq_len,), -100, dtype=torch.long)
    x[:len(inp)] = torch.tensor(inp, dtype=torch.long)

    # <SAY> input predicts first assistant token.
    supervised_start = len(prompt) - 1
    y[supervised_start:supervised_start + len(target)] = torch.tensor(target, dtype=torch.long)

    return x, y, {
        "repacked": repacked,
        "prompt_tokens": len(prompt),
        "target_tokens": len(target),
    }


class TensorRows(Dataset):
    def __init__(self, x, y, rows):
        self.x = x
        self.y = y
        self.rows = list(rows)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def tensorize(rows, tok, seq_len, max_system_tokens, max_target_tokens):
    x = torch.empty((len(rows), seq_len), dtype=torch.long)
    y = torch.empty((len(rows), seq_len), dtype=torch.long)
    stats = {
        "rows": len(rows),
        "repacked": 0,
        "max_prompt": 0,
        "max_target": 0,
        "preservation": 0,
        "repair": 0,
        "categories": {},
    }

    for i, r in enumerate(rows):
        xi, yi, m = serialize_row(
            tok, r, seq_len, max_system_tokens, max_target_tokens
        )
        x[i], y[i] = xi, yi
        stats["repacked"] += int(m["repacked"])
        stats["max_prompt"] = max(stats["max_prompt"], m["prompt_tokens"])
        stats["max_target"] = max(stats["max_target"], m["target_tokens"])
        stats["preservation"] += int(r.preservation)
        stats["repair"] += int(not r.preservation)
        stats["categories"][r.category] = stats["categories"].get(r.category, 0) + 1

    return TensorRows(x, y, rows), stats


# =============================================================================
# Balanced effective-batch schedule
# =============================================================================

def build_effective_batches(
    dataset: TensorRows,
    repair_per_batch: int,
    preserve_per_batch: int,
    seed: int,
    smoke_steps: Optional[int] = None,
):
    repair = [i for i, r in enumerate(dataset.rows) if not r.preservation]
    preserve = [i for i, r in enumerate(dataset.rows) if r.preservation]

    if not repair or not preserve:
        raise RuntimeError("Training split must contain both repair and preservation rows")

    rr = random.Random(seed)
    rr.shuffle(repair)
    rr.shuffle(preserve)

    # One maximum pass through the repair pool. Preservation can wrap because
    # the intended effective-batch mix is ~60/40 rather than corpus's 62.5/37.5.
    steps = math.ceil(len(repair) / repair_per_batch)
    if smoke_steps is not None:
        steps = min(steps, smoke_steps)

    def take_with_wrap(pool, cursor, n):
        out = []
        while len(out) < n:
            if cursor >= len(pool):
                rr.shuffle(pool)
                cursor = 0
            k = min(n - len(out), len(pool) - cursor)
            out.extend(pool[cursor:cursor+k])
            cursor += k
        return out, cursor

    rep_cursor = 0
    pre_cursor = 0
    batches = []

    for _ in range(steps):
        rep_idx, rep_cursor = take_with_wrap(repair, rep_cursor, repair_per_batch)
        pre_idx, pre_cursor = take_with_wrap(preserve, pre_cursor, preserve_per_batch)
        batch = rep_idx + pre_idx
        rr.shuffle(batch)
        batches.append(batch)

    return batches


# =============================================================================
# Loss and optimizer
# =============================================================================

def sequence_balanced_loss(logits, labels):
    b, t, v = logits.shape
    ce = F.cross_entropy(
        logits.reshape(-1, v),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(b, t)
    mask = labels.ne(-100)
    row_loss = (ce * mask).sum(1) / mask.sum(1).clamp_min(1)
    return row_loss.mean()


def preservation_anchor_loss(
    student_logits, teacher_logits, labels, preserve_rows, temperature=1.0
):
    """Distill frozen-source token probabilities on preservation rows only.

    The ordinary preservation SFT target can still allow a full-model update to
    collapse a previously good greedy response into a terse answer.  This
    optional anchor constrains the student's distribution toward the frozen
    production model wherever the batch contains preservation examples, while
    leaving repair rows unconstrained by the teacher.
    """
    token_mask = labels.ne(-100) & preserve_rows[:, None]
    if not bool(token_mask.any()):
        return student_logits.new_zeros(())

    temperature = float(temperature)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    token_kl = F.kl_div(
        student_log_probs, teacher_probs, reduction="none"
    ).sum(dim=-1)
    return (
        (token_kl * token_mask).sum()
        / token_mask.sum().clamp_min(1)
        * (temperature * temperature)
    )


def configure_trainable_scope(
    model,
    train_last_n_layers: int,
    train_embedding: bool,
    train_final_norm: bool,
    train_geometry: bool,
):
    """Configure exactly which parameters may move.

    train_last_n_layers:
      0 = all transformer blocks
      N = only final N transformer blocks

    The tied token embedding / LM head is controlled by train_embedding.
    Geometry gains remain frozen unless train_geometry is explicitly enabled.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = len(model.blocks)
    if train_last_n_layers < 0 or train_last_n_layers > n_layers:
        raise ValueError(
            f"--train-last-n-layers must be between 0 and {n_layers}; got {train_last_n_layers}"
        )

    start = 0 if train_last_n_layers == 0 else n_layers - train_last_n_layers
    for idx in range(start, n_layers):
        for p in model.blocks[idx].parameters():
            p.requires_grad_(True)

    if train_embedding:
        model.token_embedding.weight.requires_grad_(True)
        # lm_head is tied to token_embedding; same Parameter object.

    if train_final_norm:
        for p in model.final_norm.parameters():
            p.requires_grad_(True)

    # Geometry is a special exception because block-level enabling above also
    # turns raw_gain on. Force it back off unless explicitly requested.
    for block in model.blocks:
        block.attn.raw_gain.requires_grad_(bool(train_geometry))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = model.num_parameters()
    if trainable == 0:
        raise RuntimeError("Training scope selected zero trainable parameters")

    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "train_last_n_layers": train_last_n_layers,
        "first_trainable_block": start,
        "train_embedding": bool(train_embedding),
        "train_final_norm": bool(train_final_norm),
        "train_geometry": bool(train_geometry),
    }


def build_optimizer(model, core_lr, embedding_lr, geometry_lr, weight_decay):
    embed_id = id(model.token_embedding.weight)
    geometry_ids = {id(b.attn.raw_gain) for b in model.blocks}
    decay, no_decay, embed, geometry = [], [], [], []
    seen = set()

    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))

        if id(p) == embed_id:
            embed.append(p)
        elif id(p) in geometry_ids:
            geometry.append(p)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    groups = []
    if decay:
        groups.append({"params":decay, "lr":core_lr, "weight_decay":weight_decay, "lr_scale":1.0})
    if no_decay:
        groups.append({"params":no_decay, "lr":core_lr, "weight_decay":0.0, "lr_scale":1.0})
    if embed:
        groups.append({
            "params":embed, "lr":embedding_lr, "weight_decay":weight_decay,
            "lr_scale":embedding_lr / core_lr,
        })
    if geometry:
        groups.append({
            "params":geometry, "lr":geometry_lr, "weight_decay":0.0,
            "lr_scale":geometry_lr / core_lr,
        })

    kwargs = dict(betas=(0.9, 0.95), eps=1e-8)
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(groups, fused=True, **kwargs)
        except Exception:
            pass
    return torch.optim.AdamW(groups, **kwargs)


def set_learning_rate(
    optimizer,
    step,
    total_steps,
    warmup_steps,
    max_lr,
    min_lr,
    schedule,
):
    if step < warmup_steps:
        base = max_lr * (step + 1) / max(1, warmup_steps)
    else:
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        p = min(max(p, 0.0), 1.0)
        if schedule == "cosine":
            base = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * p))
        elif schedule == "linear":
            base = max_lr + (min_lr - max_lr) * p
        elif schedule == "constant":
            base = max_lr
        else:
            raise ValueError(f"Unknown LR schedule: {schedule}")

    for group in optimizer.param_groups:
        group["lr"] = base * group["lr_scale"]
    return base


# =============================================================================
# Teacher-forced evaluation
# =============================================================================

@torch.inference_mode()
def evaluate_loss(model, loader, device, dtype):
    model.eval()
    losses = []
    good = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast_context(device, dtype):
            logits = model(x)
            loss = sequence_balanced_loss(logits, y)

        losses.append(float(loss.detach().cpu()))
        mask = y.ne(-100)
        pred = logits.argmax(-1)
        good += int(((pred == y) & mask).sum().cpu())
        total += int(mask.sum().cpu())

    loss = sum(losses) / max(1, len(losses))
    return {
        "loss": loss,
        "ppl": math.exp(min(loss, 20.0)),
        "token_acc": good / max(1, total),
        "tokens": total,
    }


def split_val_subsets(val_ds):
    preserve_idx = [i for i, r in enumerate(val_ds.rows) if r.preservation]
    repair_idx = [i for i, r in enumerate(val_ds.rows) if not r.preservation]

    def subset(indices):
        return TensorRows(val_ds.x[indices], val_ds.y[indices], [val_ds.rows[i] for i in indices])

    return subset(preserve_idx), subset(repair_idx)


# =============================================================================
# Runtime generation / behavioral probes
# =============================================================================

def runtime_prompt(tok, state, player, max_seq_len, max_system_tokens):
    bos = marker(tok, "<BOS>")
    player_id = marker(tok, "<PLAYER>")
    say = marker(tok, "<SAY>")

    state_tokens, repacked = pack_state(tok, state, max_system_tokens, player)
    player_tokens = [player_id] + encode_content(tok, player)

    # BOS + ... + SAY must fit max_seq_len.
    remaining = max_seq_len - 2
    keep_state = state_tokens[:remaining]
    remaining -= len(keep_state)
    player_trimmed = len(player_tokens) > remaining
    keep_player = player_tokens[-remaining:] if remaining > 0 else []

    return [bos] + keep_state + keep_player + [say], repacked, player_trimmed


@torch.inference_mode()
def greedy_generate(model, tok, state, player, device, dtype, max_new_tokens, max_system_tokens):
    context, state_repacked, player_trimmed = runtime_prompt(
        tok, state, player, model.config.max_seq_len, max_system_tokens
    )
    pad = marker(tok, "<PAD>")
    eos = marker(tok, "<EOS>")

    out = []
    stop = "max_new_tokens"

    for _ in range(max_new_tokens):
        x = torch.full(
            (1, model.config.max_seq_len),
            pad,
            dtype=torch.long,
            device=device,
        )
        x[0, :len(context)] = torch.tensor(context, dtype=torch.long, device=device)

        with autocast_context(device, dtype):
            logits = model(x)

        nxt = int(torch.argmax(logits[0, len(context) - 1]).item())

        if nxt == eos:
            stop = "EOS"
            break

        out.append(nxt)

        if len(context) >= model.config.max_seq_len:
            stop = "context_limit"
            break
        context.append(nxt)

    return {
        "text": tok.decode(out, skip_special_tokens=True).strip(),
        "tokens": len(out),
        "stop": stop,
        "state_repacked": state_repacked,
        "player_trimmed": player_trimmed,
    }


def norm_check(s):
    return " ".join(re.sub(r"[^a-z0-9']+", " ", s.casefold()).split())


def repeated_3gram(text):
    words = norm_check(text).split()
    seen = set()
    for i in range(len(words) - 2):
        g = tuple(words[i:i+3])
        if g in seen:
            return True
        seen.add(g)
    return False


def control_leak(text):
    return any(m.casefold() in text.casefold() for m in CONTROL_MARKERS)


def false_referral(text):
    s = norm_check(text)
    return any(p in s for p in FALSE_REFERRAL_PHRASES)


def check_probe(probe, text):
    s = norm_check(text)
    reasons = []

    must_all = probe.get("must_all", [])
    missing = [x for x in must_all if norm_check(x) not in s]
    if missing:
        reasons.append("missing: " + ", ".join(missing))

    must_any = probe.get("must_any", [])
    if must_any and not any(norm_check(x) in s for x in must_any):
        reasons.append("missing any: " + ", ".join(must_any))

    forbidden = [x for x in probe.get("forbid", []) if norm_check(x) in s]
    if forbidden:
        reasons.append("forbidden: " + ", ".join(forbidden))

    if probe.get("no_referral") and false_referral(text):
        reasons.append("false referral")

    if repeated_3gram(text):
        reasons.append("repeated 3-gram")

    if control_leak(text):
        reasons.append("control-token leak")

    return len(reasons) == 0, reasons


# Preservation probes: deliberately centered on known-good production behavior.
PRESERVATION_PROBES = [
    {
        "name":"v4_identity",
        "state":"Your name is Marta. You are a baker from Britain. You are warm and proud of your craft.",
        "player":"Who are you?",
        "must_all":["marta","baker","britain"],
    },
    {
        "name":"v4_profession",
        "state":"Your name is Thorin. You are a blacksmith from Minoc. You are gruff but take pride in honest work.",
        "player":"What work do you do?",
        "must_any":["blacksmith","smith"],
    },
    {
        "name":"v4_hometown",
        "state":"Your name is Clara. You are a baker from Yew. You are known for your pies.",
        "player":"Where are you from?",
        "must_any":["yew"],
    },
    {
        "name":"v4_false_identity",
        "state":"Your name is Marta. You are a baker from Britain. You are warm but proud of your craft.",
        "player":"Are you Jorah the fisherman from Vesper?",
        "must_all":["marta","baker","britain"],
        "forbid":["i am jorah","fisherman from vesper"],
    },
    {
        "name":"v4_innkeeper_food",
        "state":"Your name is Mira. You keep a small inn in Trinsic. You are practical and good-hearted.",
        "player":"I have only three coppers. What can fill my belly?",
        "must_any":["porridge","bread","stew","soup","meal","food","bowl"],
        "no_referral":True,
    },
    {
        "name":"v4_dawn",
        "state":"Your name is Tobias. You are a baker from Moonglow. You watch the world quietly.",
        "player":"What do you hear in town at dawn?",
        "must_any":["bell","bird","cart","rooster","shutter","wheel"],
        "no_referral":True,
    },
    {
        "name":"v4_farmer_hay",
        "state":"Your name is Edric. You are an experienced farmer near Minoc.",
        "player":"Black clouds are rolling in and the wind has turned cold. Should I leave the cut hay outside?",
        "must_any":["cover","roof","barn","bring","inside","rain","hay"],
        "no_referral":True,
    },
    {
        "name":"v4_uncertainty",
        "state":"Your name is Polly. You are a shopkeeper from Minoc. You have never travelled beyond the next valley and do not pretend to know distant roads.",
        "player":"Canst thou tell me the road to Serpent Isle?",
        "must_any":["know not","do not know","don't know","cannot","never travelled","beyond my knowledge"],
        "forbid":["east road","ferry","take the road","follow the road"],
    },
    # Additional preservation behaviors.
    {
        "name":"state_price",
        "state":"Your name is Pip. You are a cooper in Moonglow. Bread costs thirteen coppers today.",
        "player":"How much is a loaf today?",
        "must_any":["thirteen","13"],
    },
    {
        "name":"state_location",
        "state":"Your name is Nessa. You are a weaver in Yew. Mara the healer is at the eastern well.",
        "player":"Where is Mara?",
        "must_any":["eastern well","well"],
    },
    {
        "name":"known_route",
        "state":"Your name is Polly. You are a shopkeeper in Minoc. To reach Serpent Isle, follow the east road to the ferry at dawn.",
        "player":"How do I reach Serpent Isle?",
        "must_all":["east","ferry"],
        "must_any":["dawn"],
        "forbid":["know not","do not know"],
    },
    {
        "name":"false_profession",
        "state":"Your name is Bryn. You are a tanner in Britain.",
        "player":"You're the local blacksmith, yes?",
        "must_any":["tanner","leather","hides"],
        "forbid":["i am a blacksmith"],
    },
]

# Repair probes: 56 fresh prompts across eight observable failure classes.
# A 16-prompt quick subset runs every short evaluation; the full suite runs
# every --full-eval-every steps and at the end.
def _rp(name, category, state, player, must_all=None, must_any=None, forbid=None, no_referral=False):
    return {
        "name": name,
        "category": category,
        "state": state,
        "player": player,
        "must_all": must_all or [],
        "must_any": must_any or [],
        "forbid": forbid or [],
        "no_referral": no_referral,
    }


REPAIR_PROBES = [
    # question shape / direct answering (7)
    _rp("shape_price_rope", "question_shape", "Your name is Alys. You keep a shop in Britain. Rope costs five coppers today.", "oi, rope price?", must_any=["five","5"], no_referral=True),
    _rp("shape_when_kitchen", "question_shape", "Your name is Mira. You keep an inn in Trinsic. The kitchen closes at the evening bell.", "food stops when?", must_all=["bell"], must_any=["evening","bell"], no_referral=True),
    _rp("shape_where_healer", "question_shape", "Your name is Corin. You are a mason in Trinsic. Mara the healer is beside the eastern well.", "where'd mara go?", must_any=["well","eastern"], no_referral=True),
    _rp("shape_who_guard", "question_shape", "Your name is Brina. You are a brewer in Cove. Rowan guards the north gate tonight.", "who's on the north gate?", must_any=["rowan"], no_referral=True),
    _rp("shape_which_bridge", "question_shape", "Your name is Pip. You are a cooper in Moonglow. The north bridge is closed. The south bridge is open.", "which bridge can i use?", must_any=["south"], no_referral=True),
    _rp("shape_why_gate", "question_shape", "Your name is Sage. You are a scribe in Britain. The east gate is closed because the bridge beyond it is damaged.", "why's east gate shut?", must_any=["bridge","damaged"], no_referral=True),
    _rp("shape_yes_room", "question_shape", "Your name is Talia. You keep an inn in Yew. One room is free tonight.", "got a bed tonight or not?", must_any=["aye","yes","room","bed"], forbid=["nay","no room"], no_referral=True),

    # entity / relation binding (7)
    _rp("entity_brass_archive", "entity_relation", "Your name is Cedric. You are a scribe in Moonglow. The brass key opens the archive. The iron key opens the cellar.", "which key gets me into the archive?", must_any=["brass"]),
    _rp("entity_iron_cellar", "entity_relation", "Your name is Cedric. You are a scribe in Moonglow. The brass key opens the archive. The iron key opens the cellar.", "what does the iron key open?", must_any=["cellar"]),
    _rp("entity_blue_horse", "entity_relation", "Your name is Nora. You are a merchant in Vesper. Alys owns the red horse. Finn owns the blue horse.", "whose blue horse is that?", must_any=["finn"]),
    _rp("entity_ship", "entity_relation", "Your name is Finn. You fish near Vesper. The Sea Wren sails to Jhelom. The Dawn Gull sails to Cove.", "where's the Sea Wren bound?", must_any=["jhelom"]),
    _rp("entity_chest", "entity_relation", "Your name is Hugh. You are a carpenter in Britain. Rowan owns the red chest. Flora owns the green chest.", "which chest belongs to flora?", must_any=["green"]),
    _rp("entity_person_place", "entity_relation", "Your name is Marta. You are a baker in Britain. Mara is at the shrine. Rowan is at the bank.", "who's at the bank?", must_any=["rowan"]),
    _rp("entity_gate_time", "entity_relation", "Your name is Moss. You are a gardener in Yew. The north gate opens at dawn. The south gate opens at noon.", "which gate opens first?", must_any=["north"]),

    # numeric/value (7)
    _rp("numeric_afford_yes", "numeric", "Your name is Marta. You are a baker in Britain. A loaf costs nine coppers. You have twelve coppers.", "I've twelve. Enough for one loaf?", must_any=["aye","yes","enough"], forbid=["nay","not enough"]),
    _rp("numeric_afford_no", "numeric", "Your name is Lark. You keep a shop in Jhelom. A potion costs eleven coppers. You have eight coppers.", "eight coppers. can i buy it?", must_any=["nay","no","need","not enough"], forbid=["aye","yes"]),
    _rp("numeric_divide", "numeric", "Your name is Clara. You are a baker in Yew. Two pies together cost eight coppers.", "two for eight, so one is?", must_any=["four","4"]),
    _rp("numeric_difference", "numeric", "Your name is Alys. You are a merchant in Britain. Bread costs five coppers. Cheese costs twelve.", "how much more is cheese than bread?", must_any=["seven","7"]),
    _rp("numeric_compare", "numeric", "Your name is Nora. You are a merchant in Vesper. Rope costs six coppers. Cloth costs nine.", "which is cheaper?", must_any=["rope"]),
    _rp("numeric_direct", "numeric", "Your name is Petra. You are a healer in Cove. A clean bandage costs four coppers.", "bandage price?", must_any=["four","4"]),
    _rp("numeric_three", "numeric", "Your name is Polly. You keep a shop in Minoc. Three torches cost fifteen coppers together.", "what's one torch cost?", must_any=["five","5"]),

    # uncertainty / anti-hallucination (7)
    _rp("unknown_spellbook", "uncertainty", "Your name is Nora. You are a merchant in Vesper. You have no information about the player's missing spellbook.", "where did i lose my spellbook?", must_any=["know not","do not know","don't know","cannot","not know"], no_referral=True),
    _rp("unknown_beth", "uncertainty", "Your name is Edric. You are a farmer near Minoc. You have never met anyone named Beth the baker.", "do you know beth the baker?", must_any=["know not","do not know","don't know","never met","not anyone"], no_referral=True),
    _rp("unknown_hythloth", "uncertainty", "Your name is Mira. You keep an inn in Trinsic. You have never visited Hythloth.", "safest route through Hythloth?", must_any=["know not","do not know","don't know","never","cannot"], no_referral=True),
    _rp("unknown_thief", "uncertainty", "Your name is Cedric. You are a scribe in Moonglow. You do not know who stole the blue gem.", "who stole the blue gem?", must_any=["know not","do not know","don't know","cannot"], no_referral=True),
    _rp("known_spellbook", "uncertainty", "Your name is Nora. You are a merchant in Vesper. The player's spellbook is beside the bank steps.", "where's my spellbook?", must_any=["bank","steps"], forbid=["know not","do not know"]),
    _rp("known_route", "uncertainty", "Your name is Mira. You keep an inn in Trinsic. The safe road to Cove is the south road past the old mill.", "which way to Cove?", must_any=["south","mill"], forbid=["know not","do not know"]),
    _rp("unknown_price", "uncertainty", "Your name is Alden. You are a farmer near Minoc. You do not know today's nightshade price in Britain.", "what's nightshade cost in Britain today?", must_any=["know not","do not know","don't know","cannot"], no_referral=True),

    # practical common-sense/profession conversation (7)
    _rp("practical_field", "practical", "Your name is Alden. You have farmed near Vesper for many years.", "soil's waterlogged. sow now?", must_any=["wait","drain","wet","soil","nay"], no_referral=True),
    _rp("practical_blade", "practical", "Your name is Garrick. You are a blacksmith in Minoc.", "blade's got a little nick. ruined?", must_any=["nay","no","repair","smooth","grind","fix"], no_referral=True),
    _rp("practical_leather", "practical", "Your name is Bryn. You are a tanner in Jhelom.", "wet leather. right by the fire?", must_any=["nay","no","away","slow","heat","fire"], no_referral=True),
    _rp("practical_cut", "practical", "Your name is Mara. You are a healer in Cove.", "road grit in a shallow cut. first thing?", must_any=["wash","clean","water"], no_referral=True),
    _rp("practical_door", "practical", "Your name is Hugh. You are a carpenter in Britain.", "door drags at one corner. check what first?", must_any=["hinge","hinges"], no_referral=True),
    _rp("practical_stew", "practical", "Your name is Greta. You are a cook in Trinsic.", "stew's too thin. save it?", must_any=["simmer","thicken","meal","reduce"], no_referral=True),
    _rp("practical_mine", "practical", "Your name is Thorne. You are a miner near Minoc.", "roof's cracking overhead. keep digging?", must_any=["nay","no","leave","safe","roof","stop"], no_referral=True),

    # social pragmatics (7)
    _rp("social_thanks", "social", "Your name is Elara. You are a weaver in Yew.", "thanks for the help.", must_any=["welcome","glad","friend","think nothing"], no_referral=True),
    _rp("social_goodbye", "social", "Your name is Rolf. You are a cooper in Yew.", "fare thee well, friend.", must_any=["farewell","safe","roads","well","friend"], no_referral=True),
    _rp("social_miss_home", "social", "Your name is Elara. You are a weaver in Yew.", "i miss home.", must_any=["home","hope","road","sorry","soon"], no_referral=True),
    _rp("social_hard_day", "social", "Your name is Brina. You are a brewer in Cove.", "been a rotten day.", must_any=["hope","better","hard","sorry","heart","tomorrow"], no_referral=True),
    _rp("social_morning", "social", "Your name is Tobias. You are a baker in Moonglow.", "fine morning, eh?", must_any=["aye","fair","morning","fine"], no_referral=True),
    _rp("social_rude_apology", "social", "Your name is Sage. You are a scribe in Britain.", "i was rude before. sorry.", must_any=["forgive","nothing","well","accepted","friend","trouble"], no_referral=True),
    _rp("social_confused", "social", "Your name is Marta. You are a baker in Britain.", "i didn't understand that.", must_any=["plain","simple","again","say","mean"], no_referral=True),

    # UO-specific compositional conversation (7)
    _rp("uo_bank", "uo_world", "Your name is Rowan. You are a merchant in Britain.", "bank's packed with adventurers again.", must_any=["bank","adventurer","traveler","crowd","busy","aye"], no_referral=True),
    _rp("uo_moongate", "uo_world", "Your name is Nessa. You are a weaver in Yew.", "moongate's been flashing all morning.", must_any=["traveler","busy","folk","road","aye","traffic"], no_referral=True),
    _rp("uo_ettin", "uo_world", "Your name is Edric. You are a farmer near Minoc.", "adventurers keep chasing ettins through my crops.", must_any=["field","crop","adventurer","battle","keep","away"], no_referral=True),
    _rp("uo_dungeon_food", "uo_world", "Your name is Greta. You are a cook in Trinsic.", "what food would you pack for a dungeon run?", must_any=["bread","meat","water","food"], no_referral=True),
    _rp("uo_guild_cloaks", "uo_world", "Your name is Elara. You are a weaver in Yew.", "guild wants matching cloaks. good work?", must_any=["aye","guild","cloth","cloak","work","weaver"], no_referral=True),
    _rp("uo_mine_monsters", "uo_world", "Your name is Thorne. You are a miner near Minoc.", "monsters are blocking the ore tunnel.", must_any=["tunnel","safe","clear","work","monster","danger"], no_referral=True),
    _rp("uo_tamer", "uo_world", "Your name is Maeve. You are a shepherd near Yew.", "tamer walked by with a huge beast and spooked the flock.", must_any=["flock","sheep","beast","clear","away","calm"], no_referral=True),

    # style robustness / terse / archaic / indirect (7)
    _rp("style_rope", "style", "Your name is Polly. You keep a shop in Minoc. Rope costs six coppers.", "u got rope? how much", must_any=["six","6"], no_referral=True),
    _rp("style_ferry", "style", "Your name is Finn. You fish near Vesper. The ferry leaves at dawn.", "ferry off when mate", must_any=["dawn"], no_referral=True),
    _rp("style_archaic_name", "style", "Your name is Orin. You are a mason in Trinsic.", "Pray, by what name art thou known?", must_any=["orin"]),
    _rp("style_terse_room", "style", "Your name is Mira. You keep an inn in Trinsic. One cheap room remains tonight.", "room. cheap. tonight?", must_any=["aye","room","cheap","one"], no_referral=True),
    _rp("style_formal_contract", "style", "Your name is Cedric. You are a scribe in Moonglow. A sound bargain should be written plainly, dated, and witnessed.", "Good sir, how should a bargain be made harder to deny?", must_any=["write","written","date","witness"], no_referral=True),
    _rp("style_sloppy_hay", "style", "Your name is Edric. You are a farmer near Minoc. Rain is coming and cut hay is still outside.", "hay + rain = bad yeah?", must_any=["aye","cover","bring","rain","hay"], no_referral=True),
    _rp("style_indirect_bread", "style", "Your name is Marta. You are a baker in Britain. Bread costs seven coppers.", "Suppose a traveler had seven coppers and wanted thy bread. Enough?", must_any=["aye","yes","enough","seven"], forbid=["nay","not enough"]),
]

QUICK_REPAIR_NAMES = {
    "shape_price_rope", "shape_when_kitchen",
    "entity_brass_archive", "entity_ship",
    "numeric_afford_yes", "numeric_divide",
    "unknown_spellbook", "unknown_beth",
    "practical_field", "practical_blade",
    "social_thanks", "social_goodbye",
    "uo_bank", "uo_dungeon_food",
    "style_rope", "style_archaic_name",
}
QUICK_REPAIR_PROBES = [p for p in REPAIR_PROBES if p["name"] in QUICK_REPAIR_NAMES]
assert len(REPAIR_PROBES) == 56
assert len(QUICK_REPAIR_PROBES) == 16


@torch.inference_mode()
def evaluate_behavior(
    model, tok, device, dtype, max_new_tokens, max_system_tokens,
    repair_probes=None,
    baseline_preserve_mask=None,
    baseline_mechanics=None,
    baseline_repair_rate=None,
    promotion_min_repair_rate=0.50,
    promotion_min_improvement=0.15,
    verbose_failures=True,
):
    model.eval()
    repair_probes = REPAIR_PROBES if repair_probes is None else repair_probes

    records = []
    preserve_pass = 0
    repair_pass = 0
    eos_ok = 0
    leaks = 0
    repeats = 0
    repair_false_referrals = 0
    category_total = Counter()
    category_pass = Counter()

    for group, probes in (("preserve", PRESERVATION_PROBES), ("repair", repair_probes)):
        for probe in probes:
            out = greedy_generate(
                model, tok, probe["state"], probe["player"],
                device, dtype, max_new_tokens, max_system_tokens,
            )
            ok, reasons = check_probe(probe, out["text"])

            if group == "preserve":
                preserve_pass += int(ok)
            else:
                repair_pass += int(ok)
                repair_false_referrals += int(false_referral(out["text"]))
                cat = probe.get("category", "repair")
                category_total[cat] += 1
                category_pass[cat] += int(ok)

            eos_ok += int(out["stop"] == "EOS")
            leaks += int(control_leak(out["text"]))
            repeats += int(repeated_3gram(out["text"]))

            rec = {
                "group": group,
                "category": probe.get("category", "preserve"),
                "name": probe["name"],
                "pass": ok,
                "state": probe["state"],
                "player": probe["player"],
                "model": out["text"],
                "stop": out["stop"],
                "state_repacked": out["state_repacked"],
                "reasons": reasons,
            }
            records.append(rec)

            if verbose_failures and not ok:
                print(f"  MISS {group}:{probe['name']}: {out['text']}")
                if reasons:
                    print("       " + "; ".join(reasons))

    preserve_records = [r for r in records if r["group"] == "preserve"]
    repair_records = [r for r in records if r["group"] == "repair"]

    if baseline_preserve_mask is None:
        preservation_relative_pass = True
        lost_source_passes = []
    else:
        lost_source_passes = []
        for i, r in enumerate(preserve_records):
            if baseline_preserve_mask[i] and not r["pass"]:
                lost_source_passes.append(r["name"])
        preservation_relative_pass = not lost_source_passes

    mechanics = {
        "eos_ok": eos_ok,
        "eos_total": len(records),
        "control_leaks": leaks,
        "repetitive_outputs": repeats,
    }

    # Mechanics are preservation-relative too. The frozen model is the baseline,
    # so an existing single repetition no longer invalidates every checkpoint.
    if baseline_mechanics is None:
        mechanics_relative_pass = True
    else:
        mechanics_relative_pass = (
            eos_ok >= baseline_mechanics["eos_ok"]
            and leaks <= baseline_mechanics["control_leaks"]
            and repeats <= baseline_mechanics["repetitive_outputs"]
        )

    eligible = preservation_relative_pass and mechanics_relative_pass

    repair_rate = repair_pass / max(1, len(repair_records))
    repair_improvement = (
        0.0 if baseline_repair_rate is None
        else repair_rate - baseline_repair_rate
    )

    promotable = (
        eligible
        and len(repair_records) >= 40  # promotion only from the full suite
        and repair_rate >= promotion_min_repair_rate
        and repair_improvement >= promotion_min_improvement
        and repair_false_referrals <= max(1, int(0.03 * len(repair_records)))
    )

    category_rates = {
        cat: {
            "pass": category_pass[cat],
            "total": category_total[cat],
            "rate": category_pass[cat] / max(1, category_total[cat]),
        }
        for cat in sorted(category_total)
    }

    return {
        "preserve_pass": preserve_pass,
        "preserve_total": len(preserve_records),
        "preserve_rate": preserve_pass / max(1, len(preserve_records)),
        "repair_pass": repair_pass,
        "repair_total": len(repair_records),
        "repair_rate": repair_rate,
        "repair_improvement": repair_improvement,
        "repair_false_referrals": repair_false_referrals,
        "category_rates": category_rates,
        "eos_ok": eos_ok,
        "eos_total": len(records),
        "control_leaks": leaks,
        "repetitive_outputs": repeats,
        "preservation_relative_pass": preservation_relative_pass,
        "lost_source_passes": lost_source_passes,
        "mechanics_relative_pass": mechanics_relative_pass,
        "eligible": eligible,
        "promotable": promotable,
        "records": records,
    }


def behavior_mechanics(result):
    return {
        "eos_ok": result["eos_ok"],
        "control_leaks": result["control_leaks"],
        "repetitive_outputs": result["repetitive_outputs"],
    }


def write_probe_transcript(path, behavior):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in behavior["records"]:
        if r["group"] != "repair":
            continue
        lines.append(f"[{r['category']}] PLAYER: {r['player']}")
        lines.append(f"NPC: {r['model']}")
        lines.append(f"PASS: {'YES' if r['pass'] else 'NO'}")
        if r["reasons"]:
            lines.append("WHY: " + "; ".join(r["reasons"]))
        lines.append("")
    path.write_text("\\n".join(lines), encoding="utf-8")


# =============================================================================
# Checkpoint save
# =============================================================================

def save_checkpoint(path, model, cfg, optimizer, step, metrics, provenance):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": dataclasses.asdict(cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": 1,
        "global_step": step,
        "metrics": metrics,
        "training_stage": "preservation_first_repair_v3_flexible",
        "runtime_contract": RUNTIME_CONTRACT,
        "provenance": provenance,
        "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    torch.save(payload, path)


# =============================================================================
# Training
# =============================================================================

def run(args):
    if args.cpu_threads and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, args.cpu_threads // 2)))
        except RuntimeError:
            pass

    bundle = discover_bundle(Path(args.bundle))
    source = discover_source_checkpoint(bundle, args.checkpoint)
    source_sha = sha256_file(source)

    if source_sha != EXPECTED_SOURCE_SHA256 and not args.allow_source_mismatch:
        raise SystemExit(
            "Frozen source SHA-256 does not match the known Phase-3-v3 production checkpoint.\n"
            f"Expected: {EXPECTED_SOURCE_SHA256}\n"
            f"Actual:   {source_sha}\n"
            "Use --allow-source-mismatch only if this is intentional."
        )

    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else bundle / "repaircorpus"
    rows, corpus_audit = load_corpus(
        corpus_dir, args.corpus_glob, args.allow_corpus_mismatch, args.require_manifests
    )

    train_rows, val_rows, hold_rows, split_details = grouped_split(rows, args.split_seed)

    tokenizer_path = bundle / "tokenizer" / "tokenizer.json"
    tok = Tokenizer.from_file(str(tokenizer_path))
    validate_tokenizer(tok)

    device = choose_device(args.device)
    dtype = choose_dtype(device, args.precision)

    model, source_payload, cfg = load_model(
        source, tok, device, args.attention_backend
    )
    scope_info = configure_trainable_scope(
        model,
        train_last_n_layers=args.train_last_n_layers,
        train_embedding=args.train_embedding,
        train_final_norm=args.train_final_norm,
        train_geometry=args.train_geometry,
    )

    geom_before = model.geometry_summary()

    # Keep an immutable copy only when preservation distillation is requested.
    # The default remains the original single-model low-cost path.
    teacher_model = None
    if args.preservation_anchor_weight > 0.0:
        teacher_model = copy.deepcopy(model).to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (bundle.parent / "uomind_repair_finetune_v3").resolve()
    )
    ckpt_dir = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Optional smoke reduction after grouped split.
    if args.smoke_test:
        def balanced_smoke(source_rows, max_preserve=48, max_repair=80):
            pre = [r for r in source_rows if r.preservation][:max_preserve]
            rep = [r for r in source_rows if not r.preservation][:max_repair]
            return pre + rep
        train_use = balanced_smoke(train_rows)
        val_use = balanced_smoke(val_rows, 24, 40)
    else:
        train_use = train_rows
        val_use = val_rows

    train_ds, train_stats = tensorize(
        train_use, tok, cfg.max_seq_len,
        args.max_system_tokens, args.max_target_tokens
    )
    val_ds, val_stats = tensorize(
        val_use, tok, cfg.max_seq_len,
        args.max_system_tokens, args.max_target_tokens
    )
    hold_ds, hold_stats = tensorize(
        hold_rows, tok, cfg.max_seq_len,
        args.max_system_tokens, args.max_target_tokens
    )

    preserve_val_ds, repair_val_ds = split_val_subsets(val_ds)

    pin = device.type == "cuda"
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size,
        shuffle=False, num_workers=0, pin_memory=pin
    )
    preserve_val_loader = DataLoader(
        preserve_val_ds, batch_size=args.eval_batch_size,
        shuffle=False, num_workers=0, pin_memory=pin
    )
    repair_val_loader = DataLoader(
        repair_val_ds, batch_size=args.eval_batch_size,
        shuffle=False, num_workers=0, pin_memory=pin
    )

    print("=" * 82)
    print("UO-MIND PRESERVATION-FIRST REPAIR FINE-TUNE")
    print("=" * 82)
    print(f"bundle:                  {bundle}")
    print(f"source checkpoint:       {source}")
    print(f"source sha256:           {source_sha}")
    print(f"repair corpus:           {Path(corpus_dir).resolve()}")
    print(f"rows:                    {corpus_audit['total_rows']} "
          f"(preserve={corpus_audit['preservation_rows']} repair={corpus_audit['repair_rows']})")
    print(f"grouped split:           train={len(train_rows)} val={len(val_rows)} holdout={len(hold_rows)}")
    print(f"active train:            {len(train_use)}")
    print(f"active validation:       {len(val_use)}")
    print(f"device:                  {device}")
    print(f"precision:               {str(dtype).replace('torch.', '')}")
    print(f"parameters:              {model.num_parameters():,}")
    print(f"trainable parameters:    {scope_info['trainable_parameters']:,} "
          f"({scope_info['trainable_fraction']*100:.1f}%)")
    layer_desc = "all" if args.train_last_n_layers == 0 else f"last {args.train_last_n_layers}"
    print(f"trainable blocks:        {layer_desc}")
    print(f"embedding/LM head:       {'TRAIN' if args.train_embedding else 'FROZEN'}")
    print(f"final norm:              {'TRAIN' if args.train_final_norm else 'FROZEN'}")
    print(f"geometry:                {'TRAIN' if args.train_geometry else 'FROZEN'} "
          f"mean={geom_before['mean']:.4f} range={geom_before['min']:.4f}..{geom_before['max']:.4f}")
    print(f"STATE repacks:           train={train_stats['repacked']} "
          f"val={val_stats['repacked']} holdout={hold_stats['repacked']}")
    print(f"core LR:                 {args.max_lr:.2e} -> {args.min_lr:.2e}")
    print(f"embedding/LM-head LR:    {args.embedding_lr:.2e} (used only if trainable)")
    print(f"geometry LR:             {args.geometry_lr:.2e} (used only if trainable)")
    print(f"LR schedule:             {args.lr_schedule}")
    print(f"epochs:                  {args.epochs}")
    print(f"effective mix:           repair={args.repair_per_batch} "
          f"preserve={args.preserve_per_batch}")
    print(f"micro batch:             {args.micro_batch_size}")
    print()

    provenance = {
        "source_checkpoint": str(source),
        "source_sha256": source_sha,
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "corpus_dir": str(Path(corpus_dir).resolve()),
        "corpus_glob": args.corpus_glob,
        "corpus_audit": corpus_audit,
        "trainable_scope": scope_info,
        "split_seed": args.split_seed,
        "train_seed": args.train_seed,
    }

    json_dump(output / "corpus_audit.json", corpus_audit)
    json_dump(output / "split_details.json", split_details)
    json_dump(output / "run_config.json", {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime_contract": RUNTIME_CONTRACT,
        "source_sha256": source_sha,
        "corpus_audit": corpus_audit,
        "split_sizes": {
            "train": len(train_rows),
            "validation": len(val_rows),
            "final_holdout": len(hold_rows),
        },
        "active_sizes": {
            "train": len(train_use),
            "validation": len(val_use),
        },
        "training": {
            "max_lr": args.max_lr,
            "min_lr": args.min_lr,
            "embedding_lr": args.embedding_lr,
            "geometry_lr": args.geometry_lr,
            "lr_schedule": args.lr_schedule,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "trainable_scope": scope_info,
            "repair_per_batch": args.repair_per_batch,
            "preserve_per_batch": args.preserve_per_batch,
            "micro_batch_size": args.micro_batch_size,
            "eval_every": args.eval_every,
            "full_eval_every": args.full_eval_every,
            "promotion_min_repair_rate": args.promotion_min_repair_rate,
            "promotion_min_improvement": args.promotion_min_improvement,
            "preservation_anchor_weight": args.preservation_anchor_weight,
            "preservation_anchor_temperature": args.preservation_anchor_temperature,
            "max_steps": args.max_steps,
            "geometry_frozen": not args.train_geometry,
        },
    })

    # Copy manifests only for reproducibility; leave user corpus untouched.
    manifests_out = output / "corpus_manifests"
    manifests_out.mkdir(exist_ok=True)
    for _, _, mp in find_corpus_files(Path(corpus_dir), args.corpus_glob, args.require_manifests):
        if mp is not None:
            shutil.copy2(mp, manifests_out / mp.name)

    # Baseline teacher-forced metrics.
    baseline_val = evaluate_loss(model, val_loader, device, dtype)
    baseline_preserve_val = evaluate_loss(model, preserve_val_loader, device, dtype)
    baseline_repair_val = evaluate_loss(model, repair_val_loader, device, dtype)

    print("Baseline behavior (full 56-prompt repair suite):")
    baseline_behavior = evaluate_behavior(
        model, tok, device, dtype,
        args.max_new_tokens, args.max_system_tokens,
        repair_probes=REPAIR_PROBES,
        baseline_preserve_mask=None,
        baseline_mechanics=None,
        baseline_repair_rate=None,
        promotion_min_repair_rate=args.promotion_min_repair_rate,
        promotion_min_improvement=args.promotion_min_improvement,
        verbose_failures=args.verbose_probe_failures,
    )
    baseline_mask = [
        r["pass"] for r in baseline_behavior["records"] if r["group"] == "preserve"
    ]
    baseline_full_mechanics = behavior_mechanics(baseline_behavior)
    baseline_full_repair_rate = baseline_behavior["repair_rate"]

    baseline_quick_behavior = evaluate_behavior(
        model, tok, device, dtype,
        args.max_new_tokens, args.max_system_tokens,
        repair_probes=QUICK_REPAIR_PROBES,
        baseline_preserve_mask=None,
        baseline_mechanics=None,
        baseline_repair_rate=None,
        promotion_min_repair_rate=args.promotion_min_repair_rate,
        promotion_min_improvement=args.promotion_min_improvement,
        verbose_failures=False,
    )
    baseline_quick_mechanics = behavior_mechanics(baseline_quick_behavior)
    baseline_quick_repair_rate = baseline_quick_behavior["repair_rate"]

    print(f"  preserve probes: {baseline_behavior['preserve_pass']}/{baseline_behavior['preserve_total']}")
    print(f"  repair probes:   {baseline_behavior['repair_pass']}/{baseline_behavior['repair_total']} "
          f"({baseline_behavior['repair_rate']*100:.1f}%)")
    print("  repair categories:")
    for cat, cr in baseline_behavior["category_rates"].items():
        print(f"    {cat:18s} {cr['pass']}/{cr['total']} ({cr['rate']*100:.1f}%)")
    print(
        f"  val loss:        all={baseline_val['loss']:.4f} "
        f"preserve={baseline_preserve_val['loss']:.4f} repair={baseline_repair_val['loss']:.4f}"
    )
    print()

    optimizer = build_optimizer(
        model, args.max_lr, args.embedding_lr, args.geometry_lr, args.weight_decay
    )

    scaler = None
    if device.type == "cuda" and dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except Exception:
            scaler = torch.cuda.amp.GradScaler()

    effective_batches = []
    for epoch_idx in range(args.epochs):
        epoch_batches = build_effective_batches(
            train_ds,
            repair_per_batch=args.repair_per_batch,
            preserve_per_batch=args.preserve_per_batch,
            seed=args.train_seed + 1009 * epoch_idx,
            smoke_steps=(args.smoke_steps if args.smoke_test and epoch_idx == 0 else None),
        )
        effective_batches.extend((epoch_idx + 1, b) for b in epoch_batches)
        if args.smoke_test:
            break

    if args.max_steps is not None:
        effective_batches = effective_batches[:args.max_steps]

    total_steps = len(effective_batches)
    warmup_steps = max(1, int(total_steps * args.warmup_fraction))

    print("=" * 82)
    print("FLEXIBLE REPAIR TRAINING")
    print("=" * 82)
    print(f"optimizer steps:         {total_steps}")
    print(f"quick behavior/val every:{args.eval_every} steps")
    print(f"full 56-probe eval every:{args.full_eval_every} steps + final")
    print("hard gate:               no loss of frozen-source passes; mechanics no worse than frozen")
    print(f"promotable target:       full repair >= {args.promotion_min_repair_rate*100:.0f}% and "
          f">= +{args.promotion_min_improvement*100:.0f} points vs frozen")
    print()

    history = []
    probe_history = {"baseline_full": baseline_behavior["records"], "baseline_quick": baseline_quick_behavior["records"]}

    best_eligible_score = -1e9
    best_eligible_step = None
    best_promotable_score = -1e9
    best_promotable_step = None

    global_step = 0
    start = time.perf_counter()

    def score_candidate(val_metrics, behavior):
        # Behavior dominates. Validation loss remains only a tie-breaker.
        return (
            5.0 * behavior["repair_rate"]
            + 2.0 * behavior["preserve_rate"]
            + 2.0 * behavior["repair_improvement"]
            - 0.04 * val_metrics["loss"]
        )

    model.train()

    for epoch_num, batch_indices in effective_batches:
        global_step += 1
        lr = set_learning_rate(
            optimizer,
            global_step - 1,
            total_steps,
            warmup_steps,
            args.max_lr,
            args.min_lr,
            args.lr_schedule,
        )

        optimizer.zero_grad(set_to_none=True)
        step_sft = 0.0
        step_anchor = 0.0
        seen = 0

        for start_i in range(0, len(batch_indices), args.micro_batch_size):
            idx = batch_indices[start_i:start_i + args.micro_batch_size]
            xb = train_ds.x[idx].to(device, non_blocking=True)
            yb = train_ds.y[idx].to(device, non_blocking=True)
            frac = len(idx) / len(batch_indices)

            with autocast_context(device, dtype):
                logits = model(xb)
                sft = sequence_balanced_loss(logits, yb)
                if teacher_model is not None:
                    preserve_rows = torch.tensor(
                        [train_ds.rows[i].preservation for i in idx],
                        dtype=torch.bool,
                        device=device,
                    )
                    with torch.no_grad():
                        teacher_logits = teacher_model(xb)
                    anchor = preservation_anchor_loss(
                        logits,
                        teacher_logits,
                        yb,
                        preserve_rows,
                        args.preservation_anchor_temperature,
                    )
                else:
                    anchor = logits.new_zeros(())
                objective = sft + args.preservation_anchor_weight * anchor

            loss = objective.float() * frac

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_sft += float(sft.detach().cpu()) * frac
            step_anchor += float(anchor.detach().cpu()) * frac
            seen += len(idx)

        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        elapsed = time.perf_counter() - start

        if global_step == 1 or global_step % args.log_every == 0 or global_step == total_steps:
            train_line = (
                f"epoch {epoch_num}/{args.epochs} step {global_step:3d}/{total_steps} "
                f"sft={step_sft:.4f} "
            )
            if teacher_model is not None:
                train_line += f"anchor={step_anchor:.4f} "
            print(train_line + f"lr={lr:.2e} elapsed={elapsed:.1f}s")

        should_eval = (
            global_step % args.eval_every == 0
            or global_step == total_steps
        )
        if not should_eval:
            continue

        val_metrics = evaluate_loss(model, val_loader, device, dtype)
        preserve_val_metrics = evaluate_loss(model, preserve_val_loader, device, dtype)
        repair_val_metrics = evaluate_loss(model, repair_val_loader, device, dtype)

        full_eval = (global_step % args.full_eval_every == 0) or (global_step == total_steps)
        active_repair_probes = REPAIR_PROBES if full_eval else QUICK_REPAIR_PROBES
        baseline_mech = baseline_full_mechanics if full_eval else baseline_quick_mechanics
        baseline_rr = baseline_full_repair_rate if full_eval else baseline_quick_repair_rate

        print(f"\nEvaluation at step {global_step} ({'FULL' if full_eval else 'QUICK'} behavior suite):")
        behavior = evaluate_behavior(
            model, tok, device, dtype,
            args.max_new_tokens, args.max_system_tokens,
            repair_probes=active_repair_probes,
            baseline_preserve_mask=baseline_mask,
            baseline_mechanics=baseline_mech,
            baseline_repair_rate=baseline_rr,
            promotion_min_repair_rate=args.promotion_min_repair_rate,
            promotion_min_improvement=args.promotion_min_improvement,
            verbose_failures=args.verbose_probe_failures,
        )

        if full_eval:
            write_probe_transcript(
                output / "probe_transcripts" / f"step_{global_step:03d}.txt",
                behavior,
            )

        geom_now = model.geometry_summary()
        score = score_candidate(val_metrics, behavior)

        summary = {
            "step": global_step,
            "eval_mode": "full" if full_eval else "quick",
            "train_sft_loss": step_sft,
            "train_anchor_loss": step_anchor if teacher_model is not None else None,
            "lr": lr,
            "val": val_metrics,
            "preserve_val": preserve_val_metrics,
            "repair_val": repair_val_metrics,
            "behavior": {k: v for k, v in behavior.items() if k != "records"},
            "score": score,
            "geometry": geom_now,
        }
        history.append(summary)
        probe_history[f"step_{global_step}"] = behavior["records"]

        print(
            f"  preserve: {behavior['preserve_pass']}/{behavior['preserve_total']} "
            f"repair: {behavior['repair_pass']}/{behavior['repair_total']} "
            f"delta={behavior['repair_improvement']*100:+.1f}pp "
            f"lost-source={len(behavior['lost_source_passes'])} "
            f"EOS={behavior['eos_ok']}/{behavior['eos_total']} "
            f"repeat={behavior['repetitive_outputs']} "
            f"repair-referrals={behavior['repair_false_referrals']}"
        )
        print(
            f"  val: all={val_metrics['loss']:.4f} "
            f"preserve={preserve_val_metrics['loss']:.4f} "
            f"repair={repair_val_metrics['loss']:.4f}"
        )
        print(
            f"  eligible={'YES' if behavior['eligible'] else 'NO'} "
            f"promotable={'YES' if behavior['promotable'] else 'NO'} "
            f"score={score:.4f}"
        )

        # Quick evaluations are intentionally monitoring-only.  Their repair
        # subset is not comparable to the full 56-prompt suite, so neither
        # preservation-safe nor promotable checkpoint selection may use them.
        if full_eval and behavior["eligible"] and score > best_eligible_score:
            best_eligible_score = score
            best_eligible_step = global_step
            save_checkpoint(
                ckpt_dir / "best_preserved.pt",
                model, cfg, optimizer, global_step, summary, provenance
            )
            print("  -> saved best_preserved.pt")

        if full_eval and behavior["promotable"] and score > best_promotable_score:
            best_promotable_score = score
            best_promotable_step = global_step
            save_checkpoint(
                ckpt_dir / "best_promotable.pt",
                model, cfg, optimizer, global_step, summary, provenance
            )
            print("  -> saved best_promotable.pt")

        json_dump(output / "training_history.json", history)
        json_dump(output / "behavior_probes.json", probe_history)
        model.train()
        print()

    # Always save final state for diagnosis. Never auto-promote it.
    final_val = evaluate_loss(model, val_loader, device, dtype)
    final_behavior = evaluate_behavior(
        model, tok, device, dtype,
        args.max_new_tokens, args.max_system_tokens,
        repair_probes=REPAIR_PROBES,
        baseline_preserve_mask=baseline_mask,
        baseline_mechanics=baseline_full_mechanics,
        baseline_repair_rate=baseline_full_repair_rate,
        promotion_min_repair_rate=args.promotion_min_repair_rate,
        promotion_min_improvement=args.promotion_min_improvement,
        verbose_failures=False,
    )
    write_probe_transcript(output / "probe_transcripts" / "final_terminal.txt", final_behavior)
    final_summary = {
        "step": global_step,
        "val": final_val,
        "behavior": {k: v for k, v in final_behavior.items() if k != "records"},
        "geometry": model.geometry_summary(),
    }
    save_checkpoint(
        ckpt_dir / "final_unpromoted.pt",
        model, cfg, optimizer, global_step, final_summary, provenance
    )

    # The teacher is no longer needed after the final student checkpoint is
    # written. Release it before loading/evaluating the selected checkpoint.
    if teacher_model is not None:
        del teacher_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Final holdout: only evaluate after training and only on the strongest
    # saved candidate. Never use this to choose a checkpoint.
    hold_loader = DataLoader(
        hold_ds, batch_size=args.eval_batch_size,
        shuffle=False, num_workers=0, pin_memory=pin
    )

    chosen_path = None
    chosen_label = None
    if (ckpt_dir / "best_promotable.pt").is_file():
        chosen_path = ckpt_dir / "best_promotable.pt"
        chosen_label = "best_promotable"
    elif (ckpt_dir / "best_preserved.pt").is_file():
        chosen_path = ckpt_dir / "best_preserved.pt"
        chosen_label = "best_preserved"

    holdout_metrics = None
    chosen_behavior = None

    if chosen_path:
        chosen_model, _, _ = load_model(
            chosen_path, tok, device, args.attention_backend
        )
        holdout_metrics = evaluate_loss(chosen_model, hold_loader, device, dtype)
        chosen_behavior = evaluate_behavior(
            chosen_model, tok, device, dtype,
            args.max_new_tokens, args.max_system_tokens,
            repair_probes=REPAIR_PROBES,
            baseline_preserve_mask=baseline_mask,
            baseline_mechanics=baseline_full_mechanics,
            baseline_repair_rate=baseline_full_repair_rate,
            promotion_min_repair_rate=args.promotion_min_repair_rate,
            promotion_min_improvement=args.promotion_min_improvement,
            verbose_failures=False,
        )
        del chosen_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_checkpoint": str(source),
        "source_sha256": source_sha,
        "best_preserved_step": best_eligible_step,
        "best_preserved_score": None if best_eligible_step is None else best_eligible_score,
        "best_preserved_path": str(ckpt_dir / "best_preserved.pt") if best_eligible_step is not None else None,
        "best_promotable_step": best_promotable_step,
        "best_promotable_score": None if best_promotable_step is None else best_promotable_score,
        "best_promotable_path": str(ckpt_dir / "best_promotable.pt") if best_promotable_step is not None else None,
        "final_unpromoted_path": str(ckpt_dir / "final_unpromoted.pt"),
        "holdout_evaluated_checkpoint": chosen_label,
        "holdout_teacher_forced": holdout_metrics,
        "chosen_behavior": (
            {k: v for k, v in chosen_behavior.items() if k != "records"}
            if chosen_behavior else None
        ),
        "history": history,
    }
    json_dump(output / "training_summary.json", result)

    print("=" * 82)
    print("TRAINING COMPLETE")
    print("=" * 82)

    if best_promotable_step is not None:
        print(f"Best promotable candidate: step {best_promotable_step}")
        print(f"  {ckpt_dir / 'best_promotable.pt'}")
        print("This is the first checkpoint to run through the large independent UO battery.")
    elif best_eligible_step is not None:
        print(f"No checkpoint met the stronger repair promotion target.")
        print(f"Best preservation-safe checkpoint: step {best_eligible_step}")
        print(f"  {ckpt_dir / 'best_preserved.pt'}")
        print("Treat it as experimental until manual inference shows a clear improvement.")
    else:
        print("NO CHECKPOINT PRESERVED ALL FROZEN-SOURCE BEHAVIORAL PASSES.")
        print("Keep the frozen Phase-3-v3 production checkpoint.")

    print(f"Diagnostic terminal state: {ckpt_dir / 'final_unpromoted.pt'}")
    print("Frozen production bundle was not modified.")

    if holdout_metrics:
        print(
            f"Final holdout ({chosen_label}): "
            f"loss={holdout_metrics['loss']:.4f} "
            f"ppl={holdout_metrics['ppl']:.2f} "
            f"tok_acc={holdout_metrics['token_acc']*100:.2f}%"
        )

    return 0


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="UO-Mind preservation-first repair fine-tuner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--bundle", default=".")
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--corpus-dir",
        default=None,
        help="Defaults to <bundle>/repaircorpus",
    )
    p.add_argument(
        "--corpus-glob",
        default="uomind_preservation_repair_corpus_part*.jsonl",
        help="Glob used to auto-discover corpus shards; future part5/part6 files are automatic.",
    )
    p.add_argument(
        "--require-manifests",
        action="store_true",
        help="Fail if any discovered JSONL does not have a sibling *_manifest.json.",
    )
    p.add_argument("--output-dir", default=None)

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")

    # Default to CPU so another training job may occupy CUDA.
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--precision",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="fp32",
    )
    p.add_argument(
        "--attention-backend",
        choices=["sdpa", "manual"],
        default=None,
    )
    p.add_argument("--cpu-threads", type=int, default=0)

    # Trainable scope. Default is the selective top-4 experiment.
    p.add_argument(
        "--train-last-n-layers", type=int, default=4,
        help="0 trains all transformer blocks; N trains only the final N blocks.",
    )
    p.add_argument(
        "--train-embedding", action=argparse.BooleanOptionalAction, default=False,
        help="Train the tied token embedding/LM-head matrix.",
    )
    p.add_argument(
        "--train-final-norm", action=argparse.BooleanOptionalAction, default=True,
        help="Train the final RMSNorm.",
    )
    p.add_argument(
        "--train-geometry", action=argparse.BooleanOptionalAction, default=False,
        help="Train relational geometry gains. Default is frozen.",
    )

    # Effective batch 32: 19 repair + 13 preservation.
    p.add_argument("--repair-per-batch", type=int, default=19)
    p.add_argument("--preserve-per-batch", type=int, default=13)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)

    p.add_argument("--max-lr", type=float, default=2e-6, help="Peak LR for trainable transformer/final-norm parameters.")
    p.add_argument("--min-lr", type=float, default=2e-7, help="Final/base minimum LR.")
    p.add_argument("--embedding-lr", type=float, default=4e-7, help="LR for tied embedding/LM head when enabled.")
    p.add_argument("--geometry-lr", type=float, default=2e-8, help="LR for geometry gains when enabled.")
    p.add_argument(
        "--preservation-anchor-weight", type=float, default=0.0,
        help="Optional frozen-source KL weight on preservation rows; 0 disables the extra teacher pass.",
    )
    p.add_argument(
        "--preservation-anchor-temperature", type=float, default=1.0,
        help="Temperature for the optional frozen-source preservation anchor.",
    )
    p.add_argument("--lr-schedule", choices=["cosine","linear","constant"], default="cosine")
    p.add_argument("--epochs", type=int, default=1, help="Number of repair-pool passes. Start with 1.")
    p.add_argument(
        "--max-steps", type=int, default=None,
        help="Optional early-stop limit after batch construction; final full evaluation and holdout still run.",
    )
    p.add_argument("--weight-decay", type=float, default=0.00075)
    p.add_argument("--warmup-fraction", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--max-system-tokens", type=int, default=40)
    p.add_argument("--max-target-tokens", type=int, default=56)
    p.add_argument("--max-new-tokens", type=int, default=24)

    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--full-eval-every", type=int, default=15)
    p.add_argument("--promotion-min-repair-rate", type=float, default=0.50)
    p.add_argument("--promotion-min-improvement", type=float, default=0.15)
    p.add_argument("--verbose-probe-failures", action="store_true")
    p.add_argument("--log-every", type=int, default=2)
    p.add_argument("--smoke-steps", type=int, default=4)
    p.add_argument("--split-seed", type=int, default=5042)
    p.add_argument("--train-seed", type=int, default=9261)

    p.add_argument("--allow-source-mismatch", action="store_true")
    p.add_argument("--allow-corpus-mismatch", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.repair_per_batch < 1 or args.preserve_per_batch < 1:
        raise SystemExit("repair/preservation rows per batch must be >= 1")
    effective = args.repair_per_batch + args.preserve_per_batch
    if args.micro_batch_size < 1 or args.micro_batch_size > effective:
        raise SystemExit("--micro-batch-size must be between 1 and the effective batch size")
    if args.eval_every < 1:
        raise SystemExit("--eval-every must be >= 1")
    if args.full_eval_every < args.eval_every:
        raise SystemExit("--full-eval-every must be >= --eval-every")
    if not (0.0 <= args.promotion_min_repair_rate <= 1.0):
        raise SystemExit("--promotion-min-repair-rate must be between 0 and 1")
    if not (0.0 <= args.promotion_min_improvement <= 1.0):
        raise SystemExit("--promotion-min-improvement must be between 0 and 1")
    if args.min_lr <= 0 or args.max_lr <= 0 or args.embedding_lr <= 0 or args.geometry_lr <= 0:
        raise SystemExit("learning rates must be positive")
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")
    if args.train_last_n_layers < 0 or args.train_last_n_layers > 16:
        raise SystemExit("--train-last-n-layers must be between 0 and 16")
    if args.min_lr > args.max_lr:
        raise SystemExit("--min-lr cannot exceed --max-lr")
    if args.preservation_anchor_weight < 0:
        raise SystemExit("--preservation-anchor-weight must be >= 0")
    if args.preservation_anchor_temperature <= 0:
        raise SystemExit("--preservation-anchor-temperature must be > 0")

    random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
