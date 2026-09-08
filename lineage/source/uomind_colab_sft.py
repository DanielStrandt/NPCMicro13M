#!/usr/bin/env python3
"""
uomind_colab_sft.py

Final supervised fine-tuning trainer for the 128-token UO-Mind model.

Expected Colab layout:
  /content/uomind_tinystories/tokenizer.json
  /content/uomind_tinystories/npc_behavior/checkpoints/best_balanced.pt
  /content/SFT/*.jsonl

Every JSONL file is treated as one category. Records use OpenAI-style
{"messages": [...]} chat format. Multi-turn conversations are supported.
Each assistant turn becomes one response-only SFT example.

Default final-SFT policy:
  * 4 SFT epochs
  * 3e-5 core max LR, cosine to 3e-6
  * tied embedding/output matrix at 0.5x LR
  * fixed-geometry trust gains at 0.25x LR
  * weight decay 0.02, warmup 5%, grad clip 1.0
  * 10% replay from the previous structured NPC-behavior dataset if present
  * sequence-balanced response-only loss
  * category-interleaved epochs
  * deterministic per-category conversation-level validation split
  * BF16 on A100-class hardware
  * fused AdamW when available
  * optional torch.compile only for long enough runs

Run:
  !python uomind_colab_sft.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def ensure_packages() -> None:
    if importlib.util.find_spec("tokenizers") is not None:
        return
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "tokenizers>=0.21"
    ])


ensure_packages()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

DEFAULT_WORKSPACE = Path("/content/uomind_tinystories")
DEFAULT_SFT_DIR = Path("/content/SFT")
REQUIRED_MARKERS = ("<PAD>", "<BOS>", "<EOS>", "<STATE>", "<PLAYER>", "<SAY>")


# -----------------------------------------------------------------------------
# Exact model architecture used by the previous UO-Mind trainers
# -----------------------------------------------------------------------------

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
            self.geometry_scales = tuple(self.geometry_scales)
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if len(self.geometry_scales) != self.n_heads:
            raise ValueError("one geometry scale is required per head")
        if self.max_seq_len != 128:
            raise ValueError("this final SFT build is intentionally fixed at context 128")
        if self.attention_backend not in {"sdpa", "manual"}:
            raise ValueError("attention_backend must be sdpa or manual")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def gains(self) -> torch.Tensor:
        return self.gain_max * torch.sigmoid(self.raw_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        if t != self.max_seq_len:
            raise ValueError(f"static graph expects T={self.max_seq_len}, got {t}")

        qkv = self.qkv_proj(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        gain = self.gains().to(q.dtype).view(1, self.n_heads, 1, 1)
        bias = gain * self.base_geometry.to(q.dtype) + self.causal_bias.to(q.dtype)

        if self.backend == "sdpa":
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + bias
            y = torch.softmax(scores, dim=-1) @ v

        y = y.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GeometryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class UOMindLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
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

    def num_parameters(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p)); total += p.numel()
        return total

    def geometry_gain_summary(self):
        g = torch.stack([b.attn.gains() for b in self.blocks])
        return {
            "mean": float(g.mean().detach().cpu()),
            "min": float(g.min().detach().cpu()),
            "max": float(g.max().detach().cpu()),
            "matrix": g.detach().cpu().tolist(),
        }


class SequenceBalancedLoss(nn.Module):
    """Average token CE inside each row, then average active rows."""
    def __init__(self, model: UOMindLM):
        super().__init__()
        self.model = model

    def forward(self, x, labels):
        logits = self.model(x)
        tok = F.cross_entropy(
            logits.reshape(-1, self.model.config.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(labels)
        mask = labels.ne(-100)
        row = tok.sum(1) / mask.sum(1).clamp_min(1)
        active = mask.any(1).to(row.dtype)
        return (row * active).sum() / active.sum().clamp_min(1)


# -----------------------------------------------------------------------------
# Runtime and checkpoint helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def amp_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast(dtype):
    return torch.autocast("cuda", dtype=dtype)


def grad_scaler(dtype):
    enabled = dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b: break
            h.update(b)
    return h.hexdigest()


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def norm_text(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split()).strip()


def find_checkpoint(requested: str, workspace: Path) -> Path:
    if requested != "auto":
        p = Path(requested)
        if not p.exists(): raise FileNotFoundError(p)
        return p
    candidates = [
        workspace / "npc_behavior/checkpoints/best_balanced.pt",
        workspace / "npc_behavior/checkpoints/best_npc.pt",
        workspace / "npc_behavior/checkpoints/final.pt",
        workspace / "npc_behavior/checkpoints/latest.pt",
        workspace / "uo_curriculum/checkpoints/best_uo.pt",
    ]
    for p in candidates:
        if p.exists(): return p
    raise FileNotFoundError("No trained checkpoint found; pass --checkpoint PATH")


def find_tokenizer(requested: str, workspace: Path, checkpoint: Path) -> Path:
    if requested != "auto":
        p = Path(requested)
        if p.exists(): return p
    candidates = [
        workspace / "tokenizer.json",
        checkpoint.parent / "tokenizer.json",
        checkpoint.parent.parent / "tokenizer.json",
        checkpoint.parent.parent.parent / "tokenizer.json",
    ]
    for p in candidates:
        if p.exists(): return p
    raise FileNotFoundError("Exact pretrained tokenizer.json not found")


def validate_tokenizer(tok: Tokenizer):
    missing = [m for m in REQUIRED_MARKERS if tok.token_to_id(m) is None]
    if missing: raise RuntimeError(f"Tokenizer missing required markers: {missing}")
    bad = [m for m in REQUIRED_MARKERS if tok.encode(m).ids != [tok.token_to_id(m)]]
    if bad: raise RuntimeError(f"Markers are not atomic tokens: {bad}")


def load_model(checkpoint: Path, tok: Tokenizer, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**dict(payload["model_config"]))
    if cfg.vocab_size != tok.get_vocab_size():
        raise RuntimeError("checkpoint/tokenizer vocab mismatch")
    model = UOMindLM(cfg)
    inc = model.load_state_dict(payload["model_state_dict"], strict=False)
    missing = [k for k in inc.missing_keys if "causal_bias" not in k]
    if missing or inc.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={inc.unexpected_keys}")
    model.to(device)
    print(f"Loaded {model.num_parameters():,} parameters; geometry gain mean={model.geometry_gain_summary()['mean']:.4f}", flush=True)
    return model, payload, cfg

# -----------------------------------------------------------------------------
# Curated JSONL parsing and response-only encoding
# -----------------------------------------------------------------------------

@dataclass
class Conversation:
    category: str
    source: str
    line: int
    messages: List[Dict[str, str]]
    fingerprint: str


@dataclass
class DatasetMeta:
    train_inputs: str
    train_labels: str
    train_categories: str
    val_inputs: str
    val_labels: str
    val_categories: str
    category_map: Dict[str, int]
    train_examples: int
    val_examples: int
    conversations: int
    duplicates: int
    invalid: int
    context_truncated: int
    response_truncated: int
    fingerprint: str


def discover_jsonl(root: Path) -> List[Path]:
    if not root.exists(): raise FileNotFoundError(root)
    files = sorted(p for p in root.rglob("*.jsonl") if p.is_file())
    if not files: raise RuntimeError(f"No JSONL files found under {root}")
    print(f"Discovered {len(files)} SFT category files:", flush=True)
    for p in files: print(f"  {p}", flush=True)
    return files


def category_for(path: Path, root: Path) -> str:
    return str(path.relative_to(root).with_suffix("")).replace(os.sep, "/")


def conv_fingerprint(messages) -> str:
    payload = json.dumps(
        [{"role": m["role"], "content": norm_text(m["content"])} for m in messages],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_jsonl(files: Sequence[Path], root: Path):
    out, seen = [], set()
    duplicates = invalid = 0
    stats = {}
    allowed = {"system", "user", "assistant"}

    for path in files:
        cat = category_for(path, root)
        st = dict(records=0, valid=0, duplicates=0, invalid=0, assistant_turns=0)
        with path.open("r", encoding="utf-8-sig") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                st["records"] += 1
                try: obj = json.loads(line)
                except Exception:
                    invalid += 1; st["invalid"] += 1; continue
                msgs = obj.get("messages")
                if not isinstance(msgs, list):
                    invalid += 1; st["invalid"] += 1; continue
                clean, assistants, ok = [], 0, True
                for m in msgs:
                    if not isinstance(m, dict): ok = False; break
                    role, content = m.get("role"), m.get("content")
                    if role not in allowed or not isinstance(content, str): ok = False; break
                    content = norm_text(content)
                    if not content: ok = False; break
                    clean.append({"role": role, "content": content})
                    assistants += int(role == "assistant")
                if not ok or assistants == 0:
                    invalid += 1; st["invalid"] += 1; continue
                fp = conv_fingerprint(clean)
                if fp in seen:
                    duplicates += 1; st["duplicates"] += 1; continue
                seen.add(fp)
                st["valid"] += 1; st["assistant_turns"] += assistants
                out.append(Conversation(cat, str(path), ln, clean, fp))
        stats[cat] = st
    return out, duplicates, invalid, stats


def split_by_category(conversations, val_fraction: float, seed: int):
    groups = defaultdict(list)
    for c in conversations: groups[c.category].append(c)
    train, val = [], []
    for cat in sorted(groups):
        items = sorted(groups[cat], key=lambda c: stable_hash(f"{seed}|{c.fingerprint}"))
        if val_fraction <= 0 or len(items) < 5:
            nval = 0
        else:
            nval = min(len(items) - 1, max(1, round(len(items) * val_fraction)))
        val.extend(items[:nval]); train.extend(items[nval:])
    return train, val


def encode_content(tok: Tokenizer, text: str) -> List[int]:
    return tok.encode(" " + text).ids


def context_piece(tok: Tokenizer, m: Dict[str, str]) -> List[int]:
    marker = {"system": "<STATE>", "user": "<PLAYER>", "assistant": "<SAY>"}[m["role"]]
    ids = [tok.token_to_id(marker)] + encode_content(tok, m["content"])
    if m["role"] == "assistant": ids.append(tok.token_to_id("<EOS>"))
    return ids


def build_prompt(tok: Tokenizer, preceding, budget: int, max_system_tokens: int):
    bos, say = tok.token_to_id("<BOS>"), tok.token_to_id("<SAY>")
    if budget < 2: return [bos, say][:budget], True
    system, dynamic = [], []
    for m in preceding:
        piece = context_piece(tok, m)
        if m["role"] == "system" and not system:
            system = piece[:max_system_tokens]
        else:
            dynamic.extend(piece)
    original = 2 + len(system) + len(dynamic)
    remaining = budget - 2
    keep_system = system[:remaining]
    remaining -= len(keep_system)
    keep_dynamic = dynamic[-remaining:] if remaining > 0 else []
    prompt = [bos] + keep_system + keep_dynamic + [say]
    return prompt, len(prompt) < original


def encode_assistant(tok, preceding, response: str, seq_len: int, max_target: int, max_system: int):
    eos, pad = tok.token_to_id("<EOS>"), tok.token_to_id("<PAD>")
    target = encode_content(tok, response)
    response_truncated = len(target) > max_target - 1
    if response_truncated: target = target[:max_target - 1]
    target = target + [eos]
    prefix_budget = seq_len + 1 - len(target)
    if prefix_budget < 2: return None, None, True, True
    prompt, context_truncated = build_prompt(tok, preceding, prefix_budget, max_system)
    full = prompt + target
    if len(full) > seq_len + 1 or len(full) < 2: return None, None, response_truncated, context_truncated

    x = np.full((seq_len,), pad, dtype=np.uint16)
    y = np.full((seq_len,), -100, dtype=np.int16)
    inp, nxt = full[:-1], full[1:]
    x[:len(inp)] = np.asarray(inp, dtype=np.uint16)
    supervised_start = len(prompt) - 1  # <SAY> predicts response token 0
    y[supervised_start:len(nxt)] = np.asarray(nxt[supervised_start:], dtype=np.int16)
    return x, y, response_truncated, context_truncated


def source_fingerprint(files, tokenizer_path, val_fraction, max_target, max_system):
    h = hashlib.sha256(b"uomind-final-sft-v3")
    h.update(sha256(tokenizer_path).encode())
    h.update(repr((val_fraction, max_target, max_system)).encode())
    for p in files:
        h.update(str(p).encode()); h.update(sha256(p).encode())
    return h.hexdigest()


def write_split(name, conversations, tok, cat_map, seq_len, max_target, max_system, outdir):
    rows = []
    response_truncated = context_truncated = rejected = 0
    for c in conversations:
        for i, m in enumerate(c.messages):
            if m["role"] != "assistant": continue
            x, y, rt, ct = encode_assistant(tok, c.messages[:i], m["content"], seq_len, max_target, max_system)
            if x is None:
                rejected += 1; continue
            rows.append((x, y, cat_map[c.category]))
            response_truncated += int(rt); context_truncated += int(ct)
    if not rows: raise RuntimeError(f"No encodable examples in {name} split")
    inputs = np.stack([r[0] for r in rows])
    labels = np.stack([r[1] for r in rows])
    cats = np.asarray([r[2] for r in rows], dtype=np.uint16)
    ip, lp, cp = outdir/f"sft_{name}_inputs.npy", outdir/f"sft_{name}_labels.npy", outdir/f"sft_{name}_categories.npy"
    np.save(ip, inputs); np.save(lp, labels); np.save(cp, cats)
    return ip, lp, cp, dict(examples=len(rows), response_truncated=response_truncated, context_truncated=context_truncated, rejected=rejected)


def prepare_sft(root, files, tok, tokenizer_path, seq_len, val_fraction, max_target, max_system, outdir, seed):
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = outdir / "sft_dataset_meta.json"
    fp = source_fingerprint(files, tokenizer_path, val_fraction, max_target, max_system)
    if meta_path.exists():
        try:
            c = json.loads(meta_path.read_text())
            required = [c[k] for k in ("train_inputs","train_labels","train_categories","val_inputs","val_labels","val_categories")]
            if c.get("fingerprint") == fp and all(Path(p).exists() for p in required):
                print("SFT cache valid; skipping JSONL parsing/tokenization.", flush=True)
                return DatasetMeta(**{k:c[k] for k in DatasetMeta.__dataclass_fields__})
        except Exception: pass

    conversations, dup, invalid, stats = parse_jsonl(files, root)
    if not conversations: raise RuntimeError("No valid SFT conversations")
    train_conv, val_conv = split_by_category(conversations, val_fraction, seed)
    if not val_conv: raise RuntimeError("No SFT validation conversations; need >=5 records/category or lower split constraints")
    categories = sorted({c.category for c in conversations})
    cat_map = {c:i for i,c in enumerate(categories)}
    print(f"SFT conversations={len(conversations):,} train={len(train_conv):,} val={len(val_conv):,} duplicates={dup:,} invalid={invalid:,}", flush=True)
    for cat in categories:
        s = stats[cat]
        print(f"  {cat}: valid={s['valid']:,} assistant_turns={s['assistant_turns']:,} dup={s['duplicates']:,} invalid={s['invalid']:,}", flush=True)

    ti, tl, tc, ts = write_split("train", train_conv, tok, cat_map, seq_len, max_target, max_system, outdir)
    vi, vl, vc, vs = write_split("validation", val_conv, tok, cat_map, seq_len, max_target, max_system, outdir)
    meta = dict(
        train_inputs=str(ti), train_labels=str(tl), train_categories=str(tc),
        val_inputs=str(vi), val_labels=str(vl), val_categories=str(vc),
        category_map=cat_map, train_examples=ts["examples"], val_examples=vs["examples"],
        conversations=len(conversations), duplicates=dup, invalid=invalid,
        context_truncated=ts["context_truncated"]+vs["context_truncated"],
        response_truncated=ts["response_truncated"]+vs["response_truncated"],
        fingerprint=fp,
    )
    meta_path.write_text(json.dumps({**meta, "file_stats":stats}, indent=2))
    print(f"Encoded SFT: {meta['train_examples']:,} train + {meta['val_examples']:,} val targets; context_trimmed={meta['context_truncated']:,}; response_clipped={meta['response_truncated']:,}", flush=True)
    return DatasetMeta(**meta)


# -----------------------------------------------------------------------------
# Array datasets, behavior replay, and schedules
# -----------------------------------------------------------------------------

class ArrayDataset:
    def __init__(self, input_path, label_path, category_path=None, ram=False):
        x = np.load(input_path, mmap_mode="r"); y = np.load(label_path, mmap_mode="r")
        if x.ndim != 2 or x.shape[1] != 128 or y.shape != x.shape: raise RuntimeError("bad dataset array shapes")
        c = np.load(category_path, mmap_mode="r") if category_path else None
        if c is not None and c.shape != (x.shape[0],): raise RuntimeError("bad category shape")
        self.inputs = np.asarray(x).copy() if ram else x
        self.labels = np.asarray(y).copy() if ram else y
        self.categories = np.asarray(c).copy() if ram and c is not None else c
        self.count = x.shape[0]

    def gather(self, indices, xdst, ydst, rows):
        if len(indices):
            xdst[rows] = self.inputs[indices]; ydst[rows] = self.labels[indices]


def load_behavior_replay(workspace: Path, fraction: float):
    if fraction <= 0: return None, None
    meta_path = workspace / "npc_behavior/npc_dataset_meta.json"
    if not meta_path.exists():
        print("Behavior replay metadata not found; SFT will run without replay.", flush=True); return None, None
    m = json.loads(meta_path.read_text())
    try:
        train = ArrayDataset(m["train_inputs"], m["train_labels"], ram=False)
        val = ArrayDataset(m["validation_inputs"], m["validation_labels"], ram=False)
    except Exception as e:
        print(f"Behavior replay unavailable: {e}", flush=True); return None, None
    print(f"Behavior replay: {train.count:,} train / {val.count:,} validation", flush=True)
    return train, val


@dataclass
class Schedule:
    domains: np.ndarray
    indices: np.ndarray
    sft_rows: int
    replay_rows: int
    total_rows: int
    batches: int
    batch_size: int


def category_interleaved(categories, seed):
    groups = defaultdict(list)
    for i,c in enumerate(np.asarray(categories).tolist()): groups[int(c)].append(i)
    rng = random.Random(seed); qs = {}
    for c, inds in groups.items(): rng.shuffle(inds); qs[c] = deque(inds)
    active, out = list(qs), []
    while active:
        rng.shuffle(active); nxt = []
        for c in active:
            if qs[c]: out.append(qs[c].popleft())
            if qs[c]: nxt.append(c)
        active = nxt
    return np.asarray(out, dtype=np.int64)


def replay_indices(count, size, seed):
    if count <= 0: return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(size, size=count, replace=(count > size)).astype(np.int64)


def build_schedule(epoch, sft: ArrayDataset, replay: Optional[ArrayDataset], fraction, batch_size, seed):
    sft_order = category_interleaved(sft.categories, seed + 10007*(epoch+1))
    nr = 0 if replay is None or fraction <= 0 else round(len(sft_order)*fraction/(1-fraction))
    ri = replay_indices(nr, replay.count, seed + 20011*(epoch+1)) if nr else np.empty(0, dtype=np.int64)
    domains = np.concatenate([np.zeros(len(sft_order),np.uint8), np.ones(nr,np.uint8)])
    rng = np.random.default_rng(seed + 30013*(epoch+1)); rng.shuffle(domains)
    inds = np.empty(len(domains), dtype=np.int64)
    sp, rp = np.flatnonzero(domains==0), np.flatnonzero(domains==1)
    inds[sp] = sft_order
    if nr: inds[rp] = ri
    return Schedule(domains, inds, len(sft_order), nr, len(domains), math.ceil(len(domains)/batch_size), batch_size)


def choose_batch(requested, sft_rows, replay_fraction, target_updates):
    if requested != "auto": return int(requested)
    effective = round(sft_rows / max(1e-6,1-replay_fraction))
    ideal = max(1, effective // max(1,target_updates))
    candidates = [8,12,16,24,32,40,48,64,80,96,128,160,192,256]
    safe = [b for b in candidates if b <= ideal]
    batch = max(safe) if safe else 8
    print(f"Auto batch: ~{effective:,} rows/epoch, target >= {target_updates} updates -> batch={batch}", flush=True)
    return batch


@dataclass
class Prepared:
    buffer_id: int
    x: torch.Tensor
    y: torch.Tensor
    real: int
    sft: int
    replay: int
    batch_index: int


class Prefetcher:
    def __init__(self, schedule, sft, replay, batch_size, pad_id, start_batch, buffers=3):
        self.schedule,self.sft,self.replay,self.bs,self.pad,self.start = schedule,sft,replay,batch_size,pad_id,start_batch
        self.buffers=[]
        for _ in range(max(2,buffers)):
            x=torch.empty((batch_size,128),dtype=torch.long,pin_memory=True); y=torch.empty_like(x)
            self.buffers.append((x,y,x.numpy(),y.numpy()))
        self.free=queue.Queue(); self.ready=queue.Queue(maxsize=len(self.buffers)); self.error=None
        for i in range(len(self.buffers)): self.free.put(i)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            for bi in range(self.start, self.schedule.batches):
                bid=self.free.get(); x,y,xn,yn=self.buffers[bid]; xn.fill(self.pad); yn.fill(-100)
                a=bi*self.bs; b=min(a+self.bs,self.schedule.total_rows)
                dom=self.schedule.domains[a:b]; inds=self.schedule.indices[a:b]; rows=np.arange(len(dom),dtype=np.int64)
                sr,rr=rows[dom==0],rows[dom==1]
                if len(sr): self.sft.gather(inds[dom==0],xn,yn,sr)
                if len(rr):
                    if self.replay is None: raise RuntimeError("replay schedule without dataset")
                    self.replay.gather(inds[dom==1],xn,yn,rr)
                self.ready.put(Prepared(bid,x,y,len(dom),len(sr),len(rr),bi))
            self.ready.put(None)
        except BaseException as e:
            self.error=e; self.ready.put(None)

    def __iter__(self):
        while True:
            item=self.ready.get()
            if item is None:
                if self.error: raise self.error
                break
            yield item

    def release(self,item): self.free.put(item.buffer_id)

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

@torch.inference_mode()
def eval_masked(model, ds: ArrayDataset, batch_size, max_examples, dtype, device, cat_names=None):
    was=model.training; model.eval(); n=ds.count if max_examples<=0 else min(ds.count,max_examples)
    loss_sum=0.0; rows=tok_ok=tok_n=exact=0
    cat_loss=defaultdict(float); cat_exact=defaultdict(int); cat_n=defaultdict(int)
    for a in range(0,n,batch_size):
        b=min(a+batch_size,n)
        x=torch.as_tensor(np.asarray(ds.inputs[a:b],dtype=np.int64),dtype=torch.long,device=device)
        y=torch.as_tensor(np.asarray(ds.labels[a:b],dtype=np.int64),dtype=torch.long,device=device)
        with autocast(dtype):
            logits=model(x)
            tl=F.cross_entropy(logits.reshape(-1,model.config.vocab_size),y.reshape(-1),ignore_index=-100,reduction="none").view_as(y)
        mask=y.ne(-100); counts=mask.sum(1).clamp_min(1); rl=tl.sum(1)/counts
        pred=logits.argmax(-1); ok=(pred.eq(y)&mask); rex=((~mask)|pred.eq(y)).all(1)
        rlc=rl.float().cpu().numpy(); rec=rex.cpu().numpy()
        loss_sum += float(rlc.sum()); rows += b-a; tok_ok += int(ok.sum()); tok_n += int(mask.sum()); exact += int(rex.sum())
        if ds.categories is not None:
            cats=np.asarray(ds.categories[a:b],dtype=np.int64)
            for i,c in enumerate(cats.tolist()):
                cat_loss[c]+=float(rlc[i]); cat_exact[c]+=int(rec[i]); cat_n[c]+=1
    result=dict(loss=loss_sum/max(1,rows), token_accuracy=tok_ok/max(1,tok_n), exact_accuracy=exact/max(1,rows), examples=rows, supervised_tokens=tok_n)
    result["ppl"]=math.exp(min(result["loss"],20.0))
    if ds.categories is not None:
        result["by_category"]={
            (cat_names.get(c,str(c)) if cat_names else str(c)):{"loss":cat_loss[c]/cat_n[c],"exact_accuracy":cat_exact[c]/cat_n[c],"examples":cat_n[c]}
            for c in sorted(cat_n)
        }
    if was:model.train()
    return result


class LMValidation:
    def __init__(self,path,token_count):
        self.path=Path(path); self.n=int(token_count); self.data=np.memmap(self.path,mode="r",dtype=np.uint16,shape=(self.n,)); self.blocks=(self.n-1)//128


@torch.inference_mode()
def eval_lm(model, ds: LMValidation, batch_size, max_batches, dtype, device):
    was=model.training; model.eval(); losses=[]; tokens=0
    nb=min(max_batches,math.ceil(ds.blocks/batch_size))
    for bi in range(nb):
        first=bi*batch_size; last=min(first+batch_size,ds.blocks); real=last-first
        if real<=0:break
        x=np.empty((real,128),np.int64); y=np.empty_like(x)
        for r,block in enumerate(range(first,last)):
            s=block*128; stream=ds.data[s:s+129]; x[r]=stream[:-1]; y[r]=stream[1:]
        xt=torch.as_tensor(x,dtype=torch.long,device=device); yt=torch.as_tensor(y,dtype=torch.long,device=device)
        with autocast(dtype):
            logits=model(xt); loss=F.cross_entropy(logits.reshape(-1,model.config.vocab_size),yt.reshape(-1))
        losses.append(loss.detach()); tokens += real*128
    mean=float(torch.stack(losses).mean())
    if was:model.train()
    return {"loss":mean,"ppl":math.exp(min(mean,20.0)),"tokens":tokens}


def optional_lm_validations(workspace: Path):
    uo=tiny=None
    candidates=[("uo",workspace/"uo_curriculum/uo_corpus_meta.json"),("tiny",workspace/"corpus_meta.json")]
    for name,mp in candidates:
        if not mp.exists():continue
        try:
            m=json.loads(mp.read_text()); p=m.get("validation_bin"); n=int(m.get("validation_tokens",0))
            if p and Path(p).exists() and n>128:
                if name=="uo":uo=LMValidation(p,n)
                else:tiny=LMValidation(p,n)
        except Exception:pass
    return uo,tiny


def validate_all(model,sft_val,behavior_val,uo_val,tiny_val,cat_names,args,dtype,device):
    out={}
    out["sft"]=eval_masked(model,sft_val,args.val_batch_size,args.sft_val_examples,dtype,device,cat_names)
    out["behavior"]=eval_masked(model,behavior_val,args.val_batch_size,args.behavior_val_examples,dtype,device) if behavior_val else None
    out["uo_story"]=eval_lm(model,uo_val,args.lm_val_batch_size,args.uo_val_batches,dtype,device) if uo_val else None
    out["tinystories"]=eval_lm(model,tiny_val,args.lm_val_batch_size,args.tiny_val_batches,dtype,device) if tiny_val else None
    return out


def print_metric(name,m):
    if m is None: print(f"  {name}: unavailable",flush=True); return
    if "token_accuracy" in m:
        print(f"  {name}: loss={m['loss']:.4f} ppl={m['ppl']:.3f} token_acc={m['token_accuracy']*100:.2f}% exact={m['exact_accuracy']*100:.2f}% n={m['examples']:,}",flush=True)
    else:
        print(f"  {name}: loss={m['loss']:.4f} ppl={m['ppl']:.3f} tokens={m['tokens']:,}",flush=True)


def balanced_score(metrics,baseline,args):
    score=float(metrics["sft"]["loss"])
    for key,weight in (("behavior",args.behavior_loss_penalty),("uo_story",args.uo_loss_penalty),("tinystories",args.tiny_loss_penalty)):
        if metrics.get(key) is not None and baseline.get(key) is not None:
            score += weight*max(0.0,float(metrics[key]["loss"])-float(baseline[key]["loss"]))
    return score


# -----------------------------------------------------------------------------
# Optimizer / LR / checkpoints
# -----------------------------------------------------------------------------

def build_optimizer(model,max_lr,wd,embed_scale,geometry_scale):
    specs={
        "core":dict(params=[],weight_decay=wd,lr_scale=1.0),
        "nodecay":dict(params=[],weight_decay=0.0,lr_scale=1.0),
        "embedding":dict(params=[],weight_decay=0.0,lr_scale=embed_scale),
        "geometry":dict(params=[],weight_decay=0.0,lr_scale=geometry_scale),
    }
    seen=set()
    for name,p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:continue
        seen.add(id(p)); low=name.lower()
        if name in {"token_embedding.weight","lm_head.weight"}:key="embedding"
        elif "raw_gain" in low:key="geometry"
        elif p.ndim<2 or "norm" in low:key="nodecay"
        else:key="core"
        specs[key]["params"].append(p)
    groups=[]
    for name,s in specs.items():
        if s["params"]:
            groups.append({"params":s["params"],"weight_decay":s["weight_decay"],"lr":max_lr*s["lr_scale"],"lr_scale":s["lr_scale"],"group_name":name})
    try:
        opt=torch.optim.AdamW(groups,lr=max_lr,betas=(0.9,0.95),eps=1e-8,fused=True); print("Optimizer: fused AdamW",flush=True)
    except Exception as e:
        print(f"Fused AdamW unavailable ({e}); using foreach.",flush=True); opt=torch.optim.AdamW(groups,lr=max_lr,betas=(0.9,0.95),eps=1e-8,foreach=True)
    print(f"LR scales: core=1.00x embedding={embed_scale:.2f}x geometry={geometry_scale:.2f}x",flush=True)
    return opt


def lr_at(step,total,max_lr,min_lr,warm_frac):
    warm=min(max(5,round(total*warm_frac)),max(1,total//4))
    if step<warm:return max_lr*(step+1)/warm
    progress=min(1.0,max(0.0,(step-warm)/max(1,total-warm)))
    return min_lr+(max_lr-min_lr)*0.5*(1+math.cos(math.pi*progress))


def set_lr(opt,base):
    for g in opt.param_groups:g["lr"]=base*float(g.get("lr_scale",1.0))


def recursive_cpu(v):
    if torch.is_tensor(v):return v.detach().cpu()
    if isinstance(v,dict):return {k:recursive_cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [recursive_cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(recursive_cpu(x) for x in v)
    return v


def optimizer_to(opt,device):
    for state in opt.state.values():
        for k,v in list(state.items()):
            if torch.is_tensor(v):state[k]=v.to(device)


def save_ckpt(path,model,opt,scaler,state,cfg,train_cfg,history):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    torch.save({"model_state_dict":recursive_cpu(model.state_dict()),"optimizer_state_dict":recursive_cpu(opt.state_dict()),"scaler_state_dict":scaler.state_dict(),"state":state,"model_config":asdict(cfg),"training_config":train_cfg,"history":history,"rng":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}},tmp); tmp.replace(path)


def restore_rng(payload):
    try:
        r=payload["rng"];random.setstate(r["python"]);np.random.set_state(r["numpy"]);torch.set_rng_state(r["torch"]);torch.cuda.set_rng_state_all(r["cuda"])
    except Exception:pass


def backup(path,folder):
    if folder:
        d=Path(folder);d.mkdir(parents=True,exist_ok=True);shutil.copy2(path,d/path.name)


def maybe_compile(loss_model,total_steps,args):
    if args.no_compile or total_steps<args.compile_min_steps:
        reason="disabled" if args.no_compile else f"only {total_steps} steps"
        print(f"torch.compile skipped ({reason}).",flush=True);return loss_model,False
    try:
        print(f"Compiling with mode={args.compile_mode!r}...",flush=True)
        return torch.compile(loss_model,mode=args.compile_mode,dynamic=False),True
    except Exception as e:
        print(f"Compile failed; eager fallback: {type(e).__name__}: {e}",flush=True);return loss_model,False

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train(args, model, cfg, source_ckpt, tokenizer_path, tok, meta, behavior_train, behavior_val, uo_val, tiny_val, outdir, dtype):
    device=torch.device("cuda")
    sft_train=ArrayDataset(meta.train_inputs,meta.train_labels,meta.train_categories,ram=True)
    sft_val=ArrayDataset(meta.val_inputs,meta.val_labels,meta.val_categories,ram=True)
    replay_fraction=args.behavior_replay_fraction if behavior_train is not None else 0.0
    batch_size=choose_batch(args.batch_size,sft_train.count,replay_fraction,args.target_updates_per_epoch)
    schedules=[build_schedule(e,sft_train,behavior_train,replay_fraction,batch_size,args.seed) for e in range(args.epochs)]
    total_steps=sum(s.batches for s in schedules)

    print("\nFinal SFT plan\n--------------",flush=True)
    print(f"curated train targets:   {sft_train.count:,}",flush=True)
    print(f"curated val targets:     {sft_val.count:,}",flush=True)
    print(f"categories:              {len(meta.category_map):,}",flush=True)
    print(f"behavior replay:         {replay_fraction*100:.1f}%",flush=True)
    print(f"batch size:              {batch_size:,}",flush=True)
    print(f"precision:               {'bfloat16' if dtype==torch.bfloat16 else 'float16'}",flush=True)
    print(f"epochs:                  {args.epochs}",flush=True)
    print(f"optimizer steps:         {total_steps:,}",flush=True)
    print(f"core max LR:             {args.lr:.2e}",flush=True)
    for e,s in enumerate(schedules,1):print(f"  epoch {e}: SFT={s.sft_rows:,} replay={s.replay_rows:,} batches={s.batches:,}",flush=True)

    opt=build_optimizer(model,args.lr,args.weight_decay,args.embedding_lr_scale,args.geometry_lr_scale)
    scaler=grad_scaler(dtype)
    ckdir=outdir/"checkpoints";ckdir.mkdir(parents=True,exist_ok=True)
    latest=ckdir/"latest.pt"; best_sft=ckdir/"best_sft.pt"; best_bal=ckdir/"best_balanced.pt"; final=ckdir/"final.pt"
    hist_path=outdir/"training_history.json"; history=[]
    state=dict(epoch_index=0,batch_in_epoch=0,global_step=0,sft_rows_seen=0,replay_rows_seen=0,best_sft_loss=float("inf"),best_balanced_score=float("inf"))
    train_cfg=dict(source_checkpoint=str(source_ckpt),tokenizer=str(tokenizer_path),sft_fingerprint=meta.fingerprint,epochs=args.epochs,batch_size=batch_size,behavior_replay_fraction=replay_fraction,lr=args.lr,min_lr=args.min_lr,warmup_fraction=args.warmup_fraction,weight_decay=args.weight_decay,embedding_lr_scale=args.embedding_lr_scale,geometry_lr_scale=args.geometry_lr_scale,grad_clip=args.grad_clip,total_steps=total_steps)

    resume=None
    if args.resume=="auto" and latest.exists():resume=torch.load(latest,map_location="cpu",weights_only=False)
    elif args.resume not in {"auto","none"}:
        p=Path(args.resume)
        if not p.exists():raise FileNotFoundError(p)
        resume=torch.load(p,map_location="cpu",weights_only=False)
    if resume is not None:
        old=resume.get("training_config",{})
        if old.get("sft_fingerprint")!=meta.fingerprint:raise RuntimeError("Cannot resume: SFT source/tokenizer changed")
        if int(old.get("batch_size",-1))!=batch_size:raise RuntimeError("Cannot resume with a different batch size")
        model.load_state_dict(resume["model_state_dict"],strict=True);opt.load_state_dict(resume["optimizer_state_dict"]);optimizer_to(opt,device)
        scaler.load_state_dict(resume.get("scaler_state_dict",{}));state.update(resume["state"]);history=list(resume.get("history",[]));restore_rng(resume)
        print(f"Resuming epoch {int(state['epoch_index'])+1}, batch {state['batch_in_epoch']}, global {state['global_step']:,}",flush=True)

    cat_names={v:k for k,v in meta.category_map.items()}
    if int(state["global_step"])==0:
        print("\nPre-SFT validation baseline:",flush=True)
        baseline=validate_all(model,sft_val,behavior_val,uo_val,tiny_val,cat_names,args,dtype,device)
        print_metric("Curated SFT val",baseline["sft"]);print_metric("NPC behavior val",baseline["behavior"]);print_metric("UO story val",baseline["uo_story"]);print_metric("TinyStories val",baseline["tinystories"])
        history.append({"event":"baseline","global_step":0,"metrics":baseline,"geometry":model.geometry_gain_summary()});hist_path.write_text(json.dumps(history,indent=2))
    else:
        base=[h for h in history if h.get("event")=="baseline"]
        if not base:raise RuntimeError("Resume history has no validation baseline")
        baseline=base[0]["metrics"]

    wrapper=SequenceBalancedLoss(model); train_model,compiled=maybe_compile(wrapper,total_steps,args);train_cfg["compile_used"]=compiled
    xgpu=torch.empty((batch_size,128),dtype=torch.long,device=device);ygpu=torch.empty_like(xgpu)
    if compiled:
        print("Warming compiled graph...",flush=True);xgpu.copy_(torch.as_tensor(np.asarray(sft_train.inputs[0:batch_size if sft_train.count>=batch_size else 1],dtype=np.int64),dtype=torch.long,device=device) if sft_train.count>=batch_size else xgpu.zero_())
        # Safer static warmup: fill and put one valid example.
        xgpu.fill_(tok.token_to_id("<PAD>"));ygpu.fill_(-100)
        xgpu[0].copy_(torch.as_tensor(np.asarray(sft_train.inputs[0],dtype=np.int64),dtype=torch.long,device=device));ygpu[0].copy_(torch.as_tensor(np.asarray(sft_train.labels[0],dtype=np.int64),dtype=torch.long,device=device))
        opt.zero_grad(set_to_none=True)
        with autocast(dtype):wl=train_model(xgpu,ygpu)
        scaler.scale(wl).backward();opt.zero_grad(set_to_none=True);del wl;torch.cuda.synchronize();print("Compile warmup complete.",flush=True)

    pad=tok.token_to_id("<PAD>");start_epoch=int(state["epoch_index"]);start_batch=int(state["batch_in_epoch"]);global_step=int(state["global_step"]);sft_seen=int(state["sft_rows_seen"]);replay_seen=int(state["replay_rows_seen"]);best_sft_loss=float(state["best_sft_loss"]);best_balanced=float(state["best_balanced_score"])
    loss_acc=torch.zeros((),dtype=torch.float32,device=device);loss_count=0;rows_log=0;log_start=time.perf_counter();model.train()

    for epoch in range(start_epoch,args.epochs):
        sched=schedules[epoch];sb=start_batch if epoch==start_epoch else 0
        print("\n"+"="*78,flush=True);print(f"FINAL SFT EPOCH {epoch+1}/{args.epochs} | curated={sched.sft_rows:,} | behavior replay={sched.replay_rows:,}",flush=True);print(f"Starting batch {sb:,}/{sched.batches:,}",flush=True);print("="*78,flush=True)
        pref=Prefetcher(sched,sft_train,behavior_train,batch_size,pad,sb,args.prefetch_buffers);epoch_sft=epoch_rep=0;epoch_start=time.perf_counter()
        for batch in pref:
            lr=lr_at(global_step,total_steps,args.lr,args.min_lr,args.warmup_fraction);set_lr(opt,lr)
            xgpu.copy_(batch.x,non_blocking=False);ygpu.copy_(batch.y,non_blocking=False);pref.release(batch);opt.zero_grad(set_to_none=True)
            if compiled and hasattr(torch,"compiler") and hasattr(torch.compiler,"cudagraph_mark_step_begin"):torch.compiler.cudagraph_mark_step_begin()
            with autocast(dtype):loss=train_model(xgpu,ygpu)
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip,foreach=True);scaler.step(opt);scaler.update()
            loss_acc.add_(loss.detach().float());del loss;loss_count+=1;global_step+=1;sft_seen+=batch.sft;replay_seen+=batch.replay;epoch_sft+=batch.sft;epoch_rep+=batch.replay;rows_log+=batch.real;next_batch=batch.batch_index+1

            if global_step%args.log_every==0:
                torch.cuda.synchronize();now=time.perf_counter();mean=float((loss_acc/max(1,loss_count)).item());elapsed=max(1e-9,now-log_start)
                print(f"epoch={epoch+1}/{args.epochs} batch={next_batch:,}/{sched.batches:,} ({100*next_batch/sched.batches:5.1f}%) global={global_step:,}/{total_steps:,} loss={mean:.4f} ppl={math.exp(min(mean,20)):.2f} lr={lr:.2e} rows/s={rows_log/elapsed:,.0f}",flush=True)
                loss_acc.zero_();loss_count=0;rows_log=0;log_start=now

            if args.checkpoint_every>0 and global_step%args.checkpoint_every==0:
                torch.cuda.synchronize();st=dict(epoch_index=epoch,batch_in_epoch=next_batch,global_step=global_step,sft_rows_seen=sft_seen,replay_rows_seen=replay_seen,best_sft_loss=best_sft_loss,best_balanced_score=best_balanced)
                save_ckpt(latest,model,opt,scaler,st,cfg,train_cfg,history);backup(latest,args.backup_dir);print(f"Saved resume checkpoint at step {global_step:,}",flush=True)

        torch.cuda.synchronize();elapsed=max(1e-9,time.perf_counter()-epoch_start);tail=None
        if loss_count:tail=float((loss_acc/loss_count).item());loss_acc.zero_();loss_count=0
        print(f"\nEpoch {epoch+1} complete: SFT rows={epoch_sft:,} replay={epoch_rep:,} rows/s={(epoch_sft+epoch_rep)/elapsed:,.0f}"+(f" tail_loss={tail:.4f}" if tail is not None else ""),flush=True)
        print("Validation:",flush=True);metrics=validate_all(model,sft_val,behavior_val,uo_val,tiny_val,cat_names,args,dtype,device)
        print_metric("Curated SFT val",metrics["sft"]);print_metric("NPC behavior val",metrics["behavior"]);print_metric("UO story val",metrics["uo_story"]);print_metric("TinyStories val",metrics["tinystories"])
        score=balanced_score(metrics,baseline,args);geom=model.geometry_gain_summary();print(f"  balanced score={score:.4f} geometry_gain_mean={geom['mean']:.4f}",flush=True)
        bycat=metrics["sft"].get("by_category",{})
        if bycat:
            print("  category validation:",flush=True)
            for cat in sorted(bycat):m=bycat[cat];print(f"    {cat}: loss={m['loss']:.3f} exact={m['exact_accuracy']*100:.1f}% n={m['examples']}",flush=True)
        record={"event":"epoch_complete","epoch":epoch+1,"global_step":global_step,"metrics":metrics,"balanced_score":score,"geometry":geom};history.append(record);hist_path.write_text(json.dumps(history,indent=2))
        sft_loss=float(metrics["sft"]["loss"]);next_state=dict(epoch_index=epoch+1,batch_in_epoch=0,global_step=global_step,sft_rows_seen=sft_seen,replay_rows_seen=replay_seen,best_sft_loss=min(best_sft_loss,sft_loss),best_balanced_score=min(best_balanced,score))
        ep_path=ckdir/f"epoch_{epoch+1}.pt"
        if sft_loss<best_sft_loss:
            best_sft_loss=sft_loss;next_state["best_sft_loss"]=best_sft_loss;save_ckpt(best_sft,model,opt,scaler,next_state,cfg,train_cfg,history);backup(best_sft,args.backup_dir);print(f"New best SFT loss {best_sft_loss:.4f}",flush=True)
        if score<best_balanced:
            best_balanced=score;next_state["best_balanced_score"]=best_balanced;save_ckpt(best_bal,model,opt,scaler,next_state,cfg,train_cfg,history);backup(best_bal,args.backup_dir);print(f"New best balanced score {best_balanced:.4f}",flush=True)
        save_ckpt(ep_path,model,opt,scaler,next_state,cfg,train_cfg,history);save_ckpt(latest,model,opt,scaler,next_state,cfg,train_cfg,history);backup(ep_path,args.backup_dir);backup(latest,args.backup_dir);start_batch=0

    final_state=dict(epoch_index=args.epochs,batch_in_epoch=0,global_step=global_step,sft_rows_seen=sft_seen,replay_rows_seen=replay_seen,best_sft_loss=best_sft_loss,best_balanced_score=best_balanced)
    save_ckpt(final,model,opt,scaler,final_state,cfg,train_cfg,history);backup(final,args.backup_dir)
    summary={"source_checkpoint":str(source_ckpt),"model_config":asdict(cfg),"training_config":train_cfg,"final_state":final_state,"geometry":model.geometry_gain_summary(),"best_sft_checkpoint":str(best_sft) if best_sft.exists() else None,"best_balanced_checkpoint":str(best_bal) if best_bal.exists() else None,"final_checkpoint":str(final),"history":str(hist_path)}
    (outdir/"training_summary.json").write_text(json.dumps(summary,indent=2))
    print("\n"+"="*78+"\nFINAL SFT COMPLETE\n"+"="*78,flush=True);print(f"best SFT:      {best_sft}\nbest balanced: {best_bal}\nfinal:         {final}",flush=True)
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p=argparse.ArgumentParser(description="Final response-only SFT for UO-Mind")
    p.add_argument("--workspace",default=str(DEFAULT_WORKSPACE));p.add_argument("--sft-dir",default=str(DEFAULT_SFT_DIR));p.add_argument("--checkpoint",default="auto");p.add_argument("--tokenizer",default="auto");p.add_argument("--output-dir",default="")
    p.add_argument("--epochs",type=int,default=4);p.add_argument("--val-fraction",type=float,default=0.08);p.add_argument("--behavior-replay-fraction",type=float,default=0.10)
    p.add_argument("--lr",type=float,default=3e-5);p.add_argument("--min-lr",type=float,default=3e-6);p.add_argument("--warmup-fraction",type=float,default=0.05);p.add_argument("--weight-decay",type=float,default=0.02);p.add_argument("--embedding-lr-scale",type=float,default=0.50);p.add_argument("--geometry-lr-scale",type=float,default=0.25);p.add_argument("--grad-clip",type=float,default=1.0)
    p.add_argument("--batch-size",default="auto");p.add_argument("--target-updates-per-epoch",type=int,default=80);p.add_argument("--max-target-tokens",type=int,default=56);p.add_argument("--max-system-tokens",type=int,default=40);p.add_argument("--prefetch-buffers",type=int,default=3)
    p.add_argument("--log-every",type=int,default=50);p.add_argument("--checkpoint-every",type=int,default=250);p.add_argument("--compile-min-steps",type=int,default=600);p.add_argument("--compile-mode",default="max-autotune-no-cudagraphs",choices=("default","reduce-overhead","max-autotune","max-autotune-no-cudagraphs"));p.add_argument("--no-compile",action="store_true")
    p.add_argument("--val-batch-size",type=int,default=256);p.add_argument("--sft-val-examples",type=int,default=0);p.add_argument("--behavior-val-examples",type=int,default=4000);p.add_argument("--lm-val-batch-size",type=int,default=256);p.add_argument("--uo-val-batches",type=int,default=64);p.add_argument("--tiny-val-batches",type=int,default=32)
    p.add_argument("--behavior-loss-penalty",type=float,default=0.25);p.add_argument("--uo-loss-penalty",type=float,default=0.10);p.add_argument("--tiny-loss-penalty",type=float,default=0.05);p.add_argument("--resume",default="auto");p.add_argument("--backup-dir",default="");p.add_argument("--seed",type=int,default=42)
    return p.parse_args()


def main():
    args=parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA is required")
    if args.epochs<=0:raise ValueError("epochs must be positive")
    if not 0<args.val_fraction<0.5:raise ValueError("val_fraction must be in (0,0.5)")
    if not 0<=args.behavior_replay_fraction<0.5:raise ValueError("behavior replay must be in [0,0.5)")
    if not 0<args.min_lr<=args.lr:raise ValueError("require 0 < min_lr <= lr")
    torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.set_float32_matmul_precision("high");set_seed(args.seed)
    workspace=Path(args.workspace);root=Path(args.sft_dir);outdir=Path(args.output_dir) if args.output_dir else workspace/"final_sft";outdir.mkdir(parents=True,exist_ok=True)
    print(f"Runtime\n-------\ntorch:  {torch.__version__}\ncuda:   {torch.version.cuda}\ngpu:    {torch.cuda.get_device_name(0)}",flush=True)
    ckpt=find_checkpoint(args.checkpoint,workspace);tok_path=find_tokenizer(args.tokenizer,workspace,ckpt);tok=Tokenizer.from_file(str(tok_path));validate_tokenizer(tok);model,_,cfg=load_model(ckpt,tok,torch.device("cuda"))
    files=discover_jsonl(root);meta=prepare_sft(root,files,tok,tok_path,cfg.max_seq_len,args.val_fraction,args.max_target_tokens,args.max_system_tokens,outdir,args.seed+5000)
    if meta.response_truncated:print(f"WARNING: {meta.response_truncated:,} responses clipped to {args.max_target_tokens} target tokens",flush=True)
    if meta.context_truncated:print(f"Context trimming in {meta.context_truncated:,} examples; system persona + recent dialogue retained",flush=True)
    behavior_train,behavior_val=load_behavior_replay(workspace,args.behavior_replay_fraction);uo_val,tiny_val=optional_lm_validations(workspace);dtype=amp_dtype()
    print(f"\nSFT source summary\n------------------\nsource checkpoint: {ckpt}\ntokenizer:         {tok_path}\ncategories:        {len(meta.category_map):,}\nconversations:     {meta.conversations:,}\ntrain targets:     {meta.train_examples:,}\nval targets:       {meta.val_examples:,}\nduplicates:        {meta.duplicates:,}\ninvalid:           {meta.invalid:,}",flush=True)
    train(args,model,cfg,ckpt,tok_path,tok,meta,behavior_train,behavior_val,uo_val,tiny_val,outdir,dtype)


if __name__ == "__main__":
    main()
