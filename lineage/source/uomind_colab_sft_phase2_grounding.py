#!/usr/bin/env python3
"""
uomind_colab_sft_phase2_grounding.py

Phase-2 grounding/refinement trainer for the 128-token UO-Mind conversational NPC.

Runtime contract being optimized:
  <BOS><STATE> NPC facts/persona <PLAYER> current player speech <SAY> response <EOS>

This phase deliberately does NOT replay the earlier synthetic action-policy task.
It derives short grounding anchors only from facts literally present in each SFT
system/STATE message (name, profession, hometown), while retaining every original
curated SFT training target once per epoch for conversational style/domain breadth.

Defaults:
  * start from final_sft/checkpoints/best_balanced.pt
  * 4 epochs
  * batch 24
  * 2 grounding anchors per persona per epoch
  * 1.2e-5 core max LR -> 1.2e-6 cosine floor
  * embeddings at 0.5x LR, geometry gains at 0.1x LR
  * no behavior/action replay
  * greedy grounding probes after baseline and every epoch

Run:
  !python uomind_colab_sft_phase2_grounding.py
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
# Phase-2 grounded conversation curriculum
# -----------------------------------------------------------------------------

import re

GROUNDING_VERSION = "phase2-grounding-v1"
DEFAULT_PHASE2_OUT = DEFAULT_WORKSPACE / "sft_phase2_grounding"


# Phase 2 must start from the conversational SFT checkpoint, not the old action trainer.
def find_checkpoint(requested: str, workspace: Path) -> Path:
    if requested != "auto":
        p = Path(requested)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    candidates = [
        workspace / "final_sft/checkpoints/best_balanced.pt",
        workspace / "final_sft/checkpoints/best_sft.pt",
        workspace / "final_sft/checkpoints/final.pt",
        workspace / "final_sft/checkpoints/latest.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No final conversational SFT checkpoint found. Expected "
        "/content/uomind_tinystories/final_sft/checkpoints/best_balanced.pt "
        "or pass --checkpoint PATH."
    )


@dataclass
class Conversation:
    category: str
    source: str
    line: int
    messages: List[Dict[str, str]]
    fingerprint: str


@dataclass
class Persona:
    conversation_index: int
    system_text: str
    name: str
    article: str
    profession: str
    town: str


@dataclass
class GroundAnchor:
    group_id: int
    anchor_type: str
    category: str
    system_text: str
    user_text: str
    response: str
    name: str
    profession: str
    town: str
    wrong_name: str = ""
    wrong_profession: str = ""
    wrong_town: str = ""


@dataclass
class Phase2Meta:
    original_train_inputs: str
    original_train_labels: str
    original_train_categories: str
    original_val_inputs: str
    original_val_labels: str
    original_val_categories: str
    ground_train_inputs: str
    ground_train_labels: str
    ground_train_groups: str
    ground_val_inputs: str
    ground_val_labels: str
    ground_val_groups: str
    original_train_manifest: str
    original_val_manifest: str
    ground_train_manifest: str
    ground_val_manifest: str
    category_map: Dict[str, int]
    original_train_examples: int
    original_val_examples: int
    ground_train_examples: int
    ground_val_examples: int
    train_personas: int
    val_personas: int
    conversations: int
    duplicates: int
    invalid: int
    persona_parse_failures: int
    context_truncated: int
    response_truncated: int
    fingerprint: str


def discover_jsonl(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted(p for p in root.rglob("*.jsonl") if p.is_file())
    if not files:
        raise RuntimeError(f"No JSONL files found under {root}")
    print(f"Discovered {len(files)} SFT category files:", flush=True)
    for p in files:
        print(f"  {p}", flush=True)
    return files


def category_for(path: Path, root: Path) -> str:
    return str(path.relative_to(root).with_suffix("")).replace(os.sep, "/")


def conv_fingerprint(messages) -> str:
    payload = json.dumps(
        [{"role": m["role"], "content": norm_text(m["content"])} for m in messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
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
                if not line:
                    continue
                st["records"] += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    invalid += 1
                    st["invalid"] += 1
                    continue
                msgs = obj.get("messages")
                if not isinstance(msgs, list):
                    invalid += 1
                    st["invalid"] += 1
                    continue
                clean, assistants, ok = [], 0, True
                for m in msgs:
                    if not isinstance(m, dict):
                        ok = False
                        break
                    role, content = m.get("role"), m.get("content")
                    if role not in allowed or not isinstance(content, str):
                        ok = False
                        break
                    content = norm_text(content)
                    if not content:
                        ok = False
                        break
                    clean.append({"role": role, "content": content})
                    assistants += int(role == "assistant")
                if not ok or assistants == 0:
                    invalid += 1
                    st["invalid"] += 1
                    continue
                fp = conv_fingerprint(clean)
                if fp in seen:
                    duplicates += 1
                    st["duplicates"] += 1
                    continue
                seen.add(fp)
                st["valid"] += 1
                st["assistant_turns"] += assistants
                out.append(Conversation(cat, str(path), ln, clean, fp))
        stats[cat] = st
    return out, duplicates, invalid, stats


def split_by_category(conversations, val_fraction: float, seed: int):
    groups = defaultdict(list)
    for c in conversations:
        groups[c.category].append(c)
    train, val = [], []
    for cat in sorted(groups):
        items = sorted(groups[cat], key=lambda c: stable_hash(f"{seed}|{c.fingerprint}"))
        if val_fraction <= 0 or len(items) < 5:
            nval = 0
        else:
            nval = min(len(items) - 1, max(1, round(len(items) * val_fraction)))
        val.extend(items[:nval])
        train.extend(items[nval:])
    return train, val


def first_system(messages: Sequence[Dict[str, str]]) -> Optional[str]:
    for m in messages:
        if m["role"] == "system":
            return m["content"]
    return None


def parse_persona(system_text: str, conversation_index: int) -> Optional[Persona]:
    # Deliberately conservative: Phase 2 only derives targets from facts literally
    # present in STATE. Nothing is inferred from category names or world lore.
    nm = re.search(r"(?:^|\s)Your name is\s+(.+?)\.", system_text, flags=re.I)
    rm = re.search(r"(?:^|\s)You are\s+(a|an)\s+(.+?)\s+from\s+([^.]+)\.", system_text, flags=re.I)
    if not nm or not rm:
        return None
    name = norm_text(nm.group(1)).strip(" ,;:")
    article = rm.group(1).lower()
    profession = norm_text(rm.group(2)).strip(" ,;:")
    town = norm_text(rm.group(3)).strip(" ,;:")
    if not name or not profession or not town:
        return None
    if len(name) > 40 or len(profession) > 64 or len(town) > 48:
        return None
    return Persona(conversation_index, system_text, name, article, profession, town)


def encode_content(tok: Tokenizer, text: str) -> List[int]:
    # Exact Phase-1 SFT convention: every natural-language segment is encoded
    # with one leading ASCII space.
    return tok.encode(" " + text).ids


def context_piece(tok: Tokenizer, m: Dict[str, str]) -> List[int]:
    marker = {"system": "<STATE>", "user": "<PLAYER>", "assistant": "<SAY>"}[m["role"]]
    ids = [tok.token_to_id(marker)] + encode_content(tok, m["content"])
    if m["role"] == "assistant":
        ids.append(tok.token_to_id("<EOS>"))
    return ids


def build_prompt(tok: Tokenizer, preceding, budget: int, max_system_tokens: int):
    bos, say = tok.token_to_id("<BOS>"), tok.token_to_id("<SAY>")
    if budget < 2:
        return [bos, say][:budget], True
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
    eos = tok.token_to_id("<EOS>")
    pad = tok.token_to_id("<PAD>")
    target = encode_content(tok, response) + [eos]
    response_clipped = len(target) > max_target
    target = target[:max_target]
    if target[-1] != eos:
        target[-1] = eos
    prompt_budget = seq_len + 1 - len(target)
    prompt, context_trimmed = build_prompt(tok, preceding, prompt_budget, max_system)
    stream = prompt + target
    if len(stream) > seq_len + 1:
        raise RuntimeError("internal SFT stream overflow")
    x = np.full(seq_len, pad, dtype=np.uint16)
    y = np.full(seq_len, -100, dtype=np.int16)
    usable = min(seq_len, len(stream) - 1)
    x[:usable] = np.asarray(stream[:-1][:usable], dtype=np.uint16)
    supervised_start = len(prompt) - 1
    supervised_end = min(seq_len, supervised_start + len(target))
    if supervised_end > supervised_start:
        y[supervised_start:supervised_end] = np.asarray(
            target[: supervised_end - supervised_start], dtype=np.int16
        )
    if int(x[supervised_start]) != tok.token_to_id("<SAY>"):
        raise RuntimeError("SFT alignment error: first supervised input is not <SAY>")
    return x, y, context_trimmed, response_clipped


def expand_conversation_targets(conversations: Sequence[Conversation]):
    rows = []
    for ci, conv in enumerate(conversations):
        for mi, m in enumerate(conv.messages):
            if m["role"] != "assistant":
                continue
            rows.append((ci, conv, conv.messages[:mi], m["content"]))
    return rows


def choose_wrong_persona(personas: Sequence[Persona], i: int, seed: int) -> Optional[Persona]:
    if len(personas) < 2:
        return None
    start = stable_hash(f"wrong|{seed}|{personas[i].name}|{personas[i].profession}|{personas[i].town}") % len(personas)
    for k in range(1, len(personas) + 1):
        q = personas[(start + k) % len(personas)]
        p = personas[i]
        if (
            q.name.casefold() != p.name.casefold()
            and q.profession.casefold() != p.profession.casefold()
            and q.town.casefold() != p.town.casefold()
        ):
            return q
    return personas[(i + 1) % len(personas)]


def make_grounding_anchors(conversations: Sequence[Conversation], seed: int):
    personas = []
    parse_fail = 0
    for ci, conv in enumerate(conversations):
        sys_text = first_system(conv.messages)
        p = parse_persona(sys_text, ci) if sys_text else None
        if p is None:
            parse_fail += 1
            continue
        personas.append(p)

    anchors: List[GroundAnchor] = []
    for pi, p in enumerate(personas):
        wrong = choose_wrong_persona(personas, pi, seed)
        # Every response below is entailed by the literal STATE string.
        base = [
            ("name", "What is your name?", f"My name is {p.name}."),
            ("identity", "Who are you?", f"I am {p.name}, {p.article} {p.profession} from {p.town}."),
            ("profession", "What work do you do?", f"I am {p.article} {p.profession}."),
            ("hometown", "Where are you from?", f"I am from {p.town}."),
        ]
        for atype, user, response in base:
            anchors.append(
                GroundAnchor(pi, atype, conversations[p.conversation_index].category, p.system_text,
                             user, response, p.name, p.profession, p.town)
            )
        if wrong is not None:
            anchors.extend([
                GroundAnchor(
                    pi, "false_identity", conversations[p.conversation_index].category, p.system_text,
                    f"Are you {wrong.name}, {wrong.article} {wrong.profession} from {wrong.town}?",
                    f"Nay. I am {p.name}, {p.article} {p.profession} from {p.town}.",
                    p.name, p.profession, p.town,
                    wrong.name, wrong.profession, wrong.town,
                ),
                GroundAnchor(
                    pi, "false_profession", conversations[p.conversation_index].category, p.system_text,
                    f"Are you {wrong.article} {wrong.profession}?",
                    f"Nay. I am {p.article} {p.profession}.",
                    p.name, p.profession, p.town,
                    wrong.name, wrong.profession, wrong.town,
                ),
                GroundAnchor(
                    pi, "false_hometown", conversations[p.conversation_index].category, p.system_text,
                    f"Are you from {wrong.town}?",
                    f"Nay. I am from {p.town}.",
                    p.name, p.profession, p.town,
                    wrong.name, wrong.profession, wrong.town,
                ),
            ])
    return anchors, personas, parse_fail


def source_fingerprint(files, tokenizer_path, val_fraction, max_target, max_system, seed):
    h = hashlib.sha256()
    h.update(GROUNDING_VERSION.encode())
    h.update(sha256(tokenizer_path).encode())
    h.update(f"{val_fraction}|{max_target}|{max_system}|{seed}".encode())
    for p in files:
        h.update(str(p).encode())
        h.update(sha256(p).encode())
    return h.hexdigest()


def _save_rows(prefix: str, rows, tok, seq_len, max_target, max_system, outdir, category_map=None):
    n = len(rows)
    x = np.empty((n, seq_len), dtype=np.uint16)
    y = np.empty((n, seq_len), dtype=np.int16)
    c = np.empty((n,), dtype=np.int16) if category_map is not None else None
    groups = np.empty((n,), dtype=np.int32) if prefix.startswith("ground_") else None
    manifest_path = outdir / f"{prefix}_manifest.jsonl"
    context_trimmed = response_truncated = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, row in enumerate(rows):
            if isinstance(row, GroundAnchor):
                preceding = [
                    {"role": "system", "content": row.system_text},
                    {"role": "user", "content": row.user_text},
                ]
                response = row.response
                xx, yy, ct, rt = encode_assistant(tok, preceding, response, seq_len, max_target, max_system)
                x[i], y[i] = xx, yy
                groups[i] = row.group_id
                rec = asdict(row)
                rec["row"] = i
            else:
                ci, conv, preceding, response = row
                xx, yy, ct, rt = encode_assistant(tok, preceding, response, seq_len, max_target, max_system)
                x[i], y[i] = xx, yy
                if c is not None:
                    c[i] = category_map[conv.category]
                sys_text = first_system(preceding) or ""
                last_user = ""
                for m in reversed(preceding):
                    if m["role"] == "user":
                        last_user = m["content"]
                        break
                rec = {
                    "row": i,
                    "conversation_index": ci,
                    "category": conv.category,
                    "source": conv.source,
                    "line": conv.line,
                    "system": sys_text,
                    "user": last_user,
                    "response": response,
                }
            context_trimmed += int(ct)
            response_truncated += int(rt)
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    xp = outdir / f"{prefix}_inputs.npy"
    yp = outdir / f"{prefix}_labels.npy"
    np.save(xp, x)
    np.save(yp, y)
    cp = gp = None
    if c is not None:
        cp = outdir / f"{prefix}_categories.npy"
        np.save(cp, c)
    if groups is not None:
        gp = outdir / f"{prefix}_groups.npy"
        np.save(gp, groups)
    return xp, yp, cp, gp, manifest_path, {
        "examples": n,
        "context_truncated": context_trimmed,
        "response_truncated": response_truncated,
    }


def prepare_phase2(root, files, tok, tokenizer_path, seq_len, val_fraction, max_target, max_system, outdir, seed):
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = outdir / "phase2_dataset_meta.json"
    fp = source_fingerprint(files, tokenizer_path, val_fraction, max_target, max_system, seed)
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text())
            required = [Path(old[k]) for k in (
                "original_train_inputs", "original_train_labels", "original_train_categories",
                "original_val_inputs", "original_val_labels", "original_val_categories",
                "ground_train_inputs", "ground_train_labels", "ground_train_groups",
                "ground_val_inputs", "ground_val_labels", "ground_val_groups",
                "original_train_manifest", "original_val_manifest",
                "ground_train_manifest", "ground_val_manifest",
            )]
            if old.get("fingerprint") == fp and all(p.exists() for p in required):
                print("Phase-2 dataset cache valid; skipping rebuild.", flush=True)
                return Phase2Meta(**{k: old[k] for k in Phase2Meta.__dataclass_fields__})
        except Exception:
            pass

    conversations, dup, invalid, stats = parse_jsonl(files, root)
    if not conversations:
        raise RuntimeError("No valid SFT conversations")
    train_conv, val_conv = split_by_category(conversations, val_fraction, seed)
    if not val_conv:
        raise RuntimeError("No validation conversations")
    categories = sorted({c.category for c in conversations})
    category_map = {c: i for i, c in enumerate(categories)}

    train_orig = expand_conversation_targets(train_conv)
    val_orig = expand_conversation_targets(val_conv)
    train_ground, train_personas, train_fail = make_grounding_anchors(train_conv, seed + 11)
    val_ground, val_personas, val_fail = make_grounding_anchors(val_conv, seed + 29)
    parse_fail = train_fail + val_fail
    if not train_ground or not val_ground:
        raise RuntimeError("Could not derive grounding anchors from the SFT STATE messages")

    print(
        f"SFT conversations={len(conversations):,} train={len(train_conv):,} val={len(val_conv):,} "
        f"duplicates={dup:,} invalid={invalid:,}", flush=True
    )
    print(
        f"Groundable personas: train={len(train_personas):,}/{len(train_conv):,} "
        f"val={len(val_personas):,}/{len(val_conv):,} parse_failures={parse_fail:,}", flush=True
    )
    print(
        f"Grounding anchor bank: train={len(train_ground):,} val={len(val_ground):,} "
        f"({len(train_ground)/max(1,len(train_personas)):.1f} anchors/persona)", flush=True
    )

    ot = _save_rows("original_train", train_orig, tok, seq_len, max_target, max_system, outdir, category_map)
    ov = _save_rows("original_val", val_orig, tok, seq_len, max_target, max_system, outdir, category_map)
    gt = _save_rows("ground_train", train_ground, tok, seq_len, max_target, max_system, outdir)
    gv = _save_rows("ground_val", val_ground, tok, seq_len, max_target, max_system, outdir)

    context_truncated = ot[-1]["context_truncated"] + ov[-1]["context_truncated"] + gt[-1]["context_truncated"] + gv[-1]["context_truncated"]
    response_truncated = ot[-1]["response_truncated"] + ov[-1]["response_truncated"] + gt[-1]["response_truncated"] + gv[-1]["response_truncated"]

    meta = dict(
        original_train_inputs=str(ot[0]), original_train_labels=str(ot[1]), original_train_categories=str(ot[2]),
        original_val_inputs=str(ov[0]), original_val_labels=str(ov[1]), original_val_categories=str(ov[2]),
        ground_train_inputs=str(gt[0]), ground_train_labels=str(gt[1]), ground_train_groups=str(gt[3]),
        ground_val_inputs=str(gv[0]), ground_val_labels=str(gv[1]), ground_val_groups=str(gv[3]),
        original_train_manifest=str(ot[4]), original_val_manifest=str(ov[4]),
        ground_train_manifest=str(gt[4]), ground_val_manifest=str(gv[4]),
        category_map=category_map,
        original_train_examples=ot[-1]["examples"], original_val_examples=ov[-1]["examples"],
        ground_train_examples=gt[-1]["examples"], ground_val_examples=gv[-1]["examples"],
        train_personas=len(train_personas), val_personas=len(val_personas),
        conversations=len(conversations), duplicates=dup, invalid=invalid,
        persona_parse_failures=parse_fail,
        context_truncated=context_truncated, response_truncated=response_truncated,
        fingerprint=fp,
    )
    meta_path.write_text(json.dumps({**meta, "file_stats": stats, "grounding_version": GROUNDING_VERSION}, indent=2))
    return Phase2Meta(**meta)


# -----------------------------------------------------------------------------
# Dataset containers and Phase-2 schedule
# -----------------------------------------------------------------------------

class ArrayDataset:
    def __init__(self, input_path, label_path, category_path=None, group_path=None, ram=False):
        x = np.load(input_path, mmap_mode="r")
        y = np.load(label_path, mmap_mode="r")
        if x.ndim != 2 or x.shape[1] != 128 or y.shape != x.shape:
            raise RuntimeError("bad dataset array shapes")
        c = np.load(category_path, mmap_mode="r") if category_path else None
        g = np.load(group_path, mmap_mode="r") if group_path else None
        if c is not None and c.shape != (x.shape[0],):
            raise RuntimeError("bad category shape")
        if g is not None and g.shape != (x.shape[0],):
            raise RuntimeError("bad group shape")
        self.inputs = np.asarray(x).copy() if ram else x
        self.labels = np.asarray(y).copy() if ram else y
        self.categories = np.asarray(c).copy() if ram and c is not None else c
        self.groups = np.asarray(g).copy() if ram and g is not None else g
        self.count = x.shape[0]

    def gather(self, indices, xdst, ydst, rows):
        if len(indices):
            xdst[rows] = self.inputs[indices]
            ydst[rows] = self.labels[indices]


def category_interleaved(categories, seed):
    groups = defaultdict(list)
    for i, c in enumerate(np.asarray(categories).tolist()):
        groups[int(c)].append(i)
    rng = random.Random(seed)
    qs = {}
    for c, inds in groups.items():
        rng.shuffle(inds)
        qs[c] = deque(inds)
    active, out = list(qs), []
    while active:
        rng.shuffle(active)
        nxt = []
        for c in active:
            if qs[c]:
                out.append(qs[c].popleft())
            if qs[c]:
                nxt.append(c)
        active = nxt
    return np.asarray(out, dtype=np.int64)


def grounding_epoch_indices(groups, epoch: int, per_persona: int, seed: int):
    by_group = defaultdict(list)
    for i, g in enumerate(np.asarray(groups).tolist()):
        by_group[int(g)].append(i)
    out = []
    for gid in sorted(by_group):
        inds = by_group[gid]
        # Rotate through the anchor bank across epochs, then shuffle the final row order.
        start = stable_hash(f"anchor|{seed}|{gid}|{epoch}") % len(inds)
        for j in range(min(per_persona, len(inds))):
            out.append(inds[(start + j) % len(inds)])
    rng = np.random.default_rng(seed + 70001 * (epoch + 1))
    out = np.asarray(out, dtype=np.int64)
    rng.shuffle(out)
    return out


@dataclass
class Schedule:
    domains: np.ndarray  # 0 grounding, 1 original SFT
    indices: np.ndarray
    grounding_rows: int
    original_rows: int
    total_rows: int
    batches: int
    batch_size: int


def build_schedule(epoch, ground: ArrayDataset, original: ArrayDataset, grounding_per_persona, batch_size, seed):
    gi = grounding_epoch_indices(ground.groups, epoch, grounding_per_persona, seed)
    oi = category_interleaved(original.categories, seed + 10007 * (epoch + 1))
    domains = np.concatenate([np.zeros(len(gi), np.uint8), np.ones(len(oi), np.uint8)])
    indices = np.concatenate([gi, oi])
    rng = np.random.default_rng(seed + 30013 * (epoch + 1))
    perm = rng.permutation(len(domains))
    domains = domains[perm]
    indices = indices[perm]
    return Schedule(
        domains, indices, len(gi), len(oi), len(domains),
        math.ceil(len(domains) / batch_size), batch_size,
    )


@dataclass
class Prepared:
    buffer_id: int
    x: torch.Tensor
    y: torch.Tensor
    real: int
    grounding: int
    original: int
    batch_index: int


class Prefetcher:
    def __init__(self, schedule, ground, original, batch_size, pad_id, start_batch, buffers=3):
        self.schedule = schedule
        self.ground = ground
        self.original = original
        self.bs = batch_size
        self.pad = pad_id
        self.start = start_batch
        self.buffers = []
        for _ in range(max(2, buffers)):
            x = torch.empty((batch_size, 128), dtype=torch.long, pin_memory=True)
            y = torch.empty_like(x)
            self.buffers.append((x, y, x.numpy(), y.numpy()))
        self.free = queue.Queue()
        self.ready = queue.Queue(maxsize=len(self.buffers))
        self.error = None
        for i in range(len(self.buffers)):
            self.free.put(i)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            for bi in range(self.start, self.schedule.batches):
                bid = self.free.get()
                x, y, xn, yn = self.buffers[bid]
                xn.fill(self.pad)
                yn.fill(-100)
                a = bi * self.bs
                b = min(a + self.bs, self.schedule.total_rows)
                dom = self.schedule.domains[a:b]
                inds = self.schedule.indices[a:b]
                rows = np.arange(len(dom), dtype=np.int64)
                gr = rows[dom == 0]
                sr = rows[dom == 1]
                if len(gr):
                    self.ground.gather(inds[dom == 0], xn, yn, gr)
                if len(sr):
                    self.original.gather(inds[dom == 1], xn, yn, sr)
                self.ready.put(Prepared(bid, x, y, len(dom), len(gr), len(sr), bi))
            self.ready.put(None)
        except BaseException as e:
            self.error = e
            self.ready.put(None)

    def __iter__(self):
        while True:
            item = self.ready.get()
            if item is None:
                if self.error:
                    raise self.error
                break
            yield item

    def release(self, item):
        self.free.put(item.buffer_id)


# -----------------------------------------------------------------------------
# Validation and greedy grounding probes
# -----------------------------------------------------------------------------

@torch.inference_mode()
def eval_masked(model, ds: ArrayDataset, batch_size, max_examples, dtype, device, cat_names=None):
    was = model.training
    model.eval()
    n = ds.count if max_examples <= 0 else min(ds.count, max_examples)
    loss_sum = 0.0
    rows = tok_ok = tok_n = exact = 0
    cat_loss = defaultdict(float)
    cat_exact = defaultdict(int)
    cat_n = defaultdict(int)
    for a in range(0, n, batch_size):
        b = min(a + batch_size, n)
        x = torch.as_tensor(np.asarray(ds.inputs[a:b], dtype=np.int64), dtype=torch.long, device=device)
        y = torch.as_tensor(np.asarray(ds.labels[a:b], dtype=np.int64), dtype=torch.long, device=device)
        with autocast(dtype):
            logits = model(x)
            tl = F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size), y.reshape(-1),
                ignore_index=-100, reduction="none"
            ).view_as(y)
        mask = y.ne(-100)
        counts = mask.sum(1).clamp_min(1)
        rl = tl.sum(1) / counts
        pred = logits.argmax(-1)
        ok = pred.eq(y) & mask
        rex = ((~mask) | pred.eq(y)).all(1)
        rlc = rl.float().cpu().numpy()
        rec = rex.cpu().numpy()
        loss_sum += float(rlc.sum())
        rows += b - a
        tok_ok += int(ok.sum())
        tok_n += int(mask.sum())
        exact += int(rex.sum())
        if ds.categories is not None:
            cats = np.asarray(ds.categories[a:b], dtype=np.int64)
            for i, c in enumerate(cats.tolist()):
                cat_loss[c] += float(rlc[i])
                cat_exact[c] += int(rec[i])
                cat_n[c] += 1
    result = dict(
        loss=loss_sum / max(1, rows),
        token_accuracy=tok_ok / max(1, tok_n),
        exact_accuracy=exact / max(1, rows),
        examples=rows,
        supervised_tokens=tok_n,
    )
    result["ppl"] = math.exp(min(result["loss"], 20.0))
    if ds.categories is not None:
        result["by_category"] = {
            (cat_names.get(c, str(c)) if cat_names else str(c)): {
                "loss": cat_loss[c] / cat_n[c],
                "exact_accuracy": cat_exact[c] / cat_n[c],
                "examples": cat_n[c],
            }
            for c in sorted(cat_n)
        }
    if was:
        model.train()
    return result


class LMValidation:
    def __init__(self, path, token_count):
        self.path = Path(path)
        self.n = int(token_count)
        self.data = np.memmap(self.path, mode="r", dtype=np.uint16, shape=(self.n,))
        self.blocks = (self.n - 1) // 128


@torch.inference_mode()
def eval_lm(model, ds: LMValidation, batch_size, max_batches, dtype, device):
    was = model.training
    model.eval()
    losses = []
    tokens = 0
    nb = min(max_batches, math.ceil(ds.blocks / batch_size))
    for bi in range(nb):
        first = bi * batch_size
        last = min(first + batch_size, ds.blocks)
        real = last - first
        if real <= 0:
            break
        x = np.empty((real, 128), np.int64)
        y = np.empty_like(x)
        for r, block in enumerate(range(first, last)):
            s = block * 128
            stream = ds.data[s:s + 129]
            x[r] = stream[:-1]
            y[r] = stream[1:]
        xt = torch.as_tensor(x, dtype=torch.long, device=device)
        yt = torch.as_tensor(y, dtype=torch.long, device=device)
        with autocast(dtype):
            logits = model(xt)
            loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), yt.reshape(-1))
        losses.append(loss.detach())
        tokens += real * 128
    mean = float(torch.stack(losses).mean())
    if was:
        model.train()
    return {"loss": mean, "ppl": math.exp(min(mean, 20.0)), "tokens": tokens}


def optional_lm_validations(workspace: Path):
    uo = tiny = None
    for name, mp in (("uo", workspace / "uo_curriculum/uo_corpus_meta.json"), ("tiny", workspace / "corpus_meta.json")):
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text())
            p = m.get("validation_bin")
            n = int(m.get("validation_tokens", 0))
            if p and Path(p).exists() and n > 128:
                if name == "uo":
                    uo = LMValidation(p, n)
                else:
                    tiny = LMValidation(p, n)
        except Exception:
            pass
    return uo, tiny


def load_jsonl_manifest(path: str):
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def supervised_prefix(ds: ArrayDataset, row: int):
    y = np.asarray(ds.labels[row], dtype=np.int64)
    nz = np.flatnonzero(y != -100)
    if len(nz) == 0:
        raise RuntimeError("row has no supervised target")
    first = int(nz[0])
    x = np.asarray(ds.inputs[row], dtype=np.int64)
    return x[:first + 1].tolist(), first


def decode_target(tok: Tokenizer, ds: ArrayDataset, row: int):
    y = np.asarray(ds.labels[row], dtype=np.int64)
    ids = [int(v) for v in y.tolist() if int(v) != -100]
    eos = tok.token_to_id("<EOS>")
    if eos in ids:
        ids = ids[:ids.index(eos)]
    return tok.decode(ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def greedy_generate(model, tok: Tokenizer, prefix_ids: Sequence[int], max_new_tokens: int, dtype, device):
    pad = tok.token_to_id("<PAD>")
    eos = tok.token_to_id("<EOS>")
    context = list(prefix_ids)
    out = []
    was = model.training
    model.eval()
    stop = "max_new_tokens"
    for _ in range(max_new_tokens):
        if not context:
            break
        if len(context) > model.config.max_seq_len:
            context = context[-model.config.max_seq_len:]
        x = torch.full((1, model.config.max_seq_len), pad, dtype=torch.long, device=device)
        real = len(context)
        x[0, :real] = torch.tensor(context, dtype=torch.long, device=device)
        with autocast(dtype):
            logits = model(x)
        nxt = int(logits[0, real - 1].argmax().item())
        if nxt == eos:
            stop = "EOS"
            break
        out.append(nxt)
        # A length-128 context can predict one token, but that token cannot be fed
        # back without exceeding the model context. Stop after recording it.
        if real >= model.config.max_seq_len:
            stop = "context_limit_after_token"
            break
        context.append(nxt)
    if was:
        model.train()
    return out, stop


def norm_for_check(text: str):
    return " ".join(re.sub(r"[^a-z0-9' -]+", " ", text.casefold()).split())


def profession_key(profession: str):
    words = re.findall(r"[a-zA-Z']+", profession.casefold())
    return words[-1] if words else profession.casefold()


def grounding_pass(rec, text: str):
    t = norm_for_check(text)
    at = rec["anchor_type"]
    name = norm_for_check(rec["name"])
    town = norm_for_check(rec["town"])
    prof = profession_key(rec["profession"])
    wrong_name = norm_for_check(rec.get("wrong_name", ""))
    wrong_town = norm_for_check(rec.get("wrong_town", ""))
    wrong_prof = profession_key(rec.get("wrong_profession", "")) if rec.get("wrong_profession") else ""
    if at == "name":
        return name in t
    if at == "identity":
        return name in t and prof in t and town in t
    if at == "profession":
        return prof in t
    if at == "hometown":
        return town in t
    if at == "false_identity":
        good = name in t and (prof in t or town in t)
        bad = (wrong_name and wrong_name in t) or (wrong_prof and wrong_prof in t and prof not in t)
        return good and not bad
    if at == "false_profession":
        return prof in t and not (wrong_prof and wrong_prof in t and prof not in t)
    if at == "false_hometown":
        return town in t and not (wrong_town and wrong_town in t and town not in t)
    return False


def select_probe_rows(manifest, count, seed, prefer_types=True):
    if not manifest:
        return []
    if prefer_types:
        by = defaultdict(list)
        for r in manifest:
            by[r.get("anchor_type", "")].append(r)
        chosen = []
        for at in sorted(by):
            chosen.append(by[at][stable_hash(f"probe|{seed}|{at}") % len(by[at])])
            if len(chosen) >= count:
                return chosen
        remaining = [r for r in manifest if r not in chosen]
    else:
        chosen = []
        remaining = list(manifest)
    remaining.sort(key=lambda r: stable_hash(f"probe|{seed}|{r.get('row',0)}|{r.get('category','')}"))
    chosen.extend(remaining[:max(0, count - len(chosen))])
    return chosen[:count]


def run_generation_probes(model, tok, ground_val, original_val, ground_manifest, original_manifest, args, dtype, device, label):
    print(f"\nGreedy conversational probes [{label}]", flush=True)
    print("---------------------------------------", flush=True)
    gp = select_probe_rows(ground_manifest, args.ground_probe_count, args.seed + 901, True)
    op = select_probe_rows(original_manifest, args.original_probe_count, args.seed + 1901, False)
    passed = 0
    probe_records = []
    for j, rec in enumerate(gp, 1):
        row = int(rec["row"])
        prefix, _ = supervised_prefix(ground_val, row)
        ids, stop = greedy_generate(model, tok, prefix, args.probe_max_new_tokens, dtype, device)
        text = tok.decode(ids, skip_special_tokens=True).strip()
        target = decode_target(tok, ground_val, row)
        ok = grounding_pass(rec, text)
        passed += int(ok)
        print(f"  G{j:02d} [{rec['anchor_type']}] {'PASS' if ok else 'MISS'}", flush=True)
        print(f"      STATE: {rec['system_text']}", flush=True)
        print(f"      PLAYER: {rec['user_text']}", flush=True)
        print(f"      MODEL:  {text}", flush=True)
        print(f"      TARGET: {target}", flush=True)
        probe_records.append({**rec, "model": text, "target": target, "pass": bool(ok), "stop": stop})
    for j, rec in enumerate(op, 1):
        row = int(rec["row"])
        prefix, _ = supervised_prefix(original_val, row)
        ids, stop = greedy_generate(model, tok, prefix, args.probe_max_new_tokens, dtype, device)
        text = tok.decode(ids, skip_special_tokens=True).strip()
        target = decode_target(tok, original_val, row)
        print(f"  O{j:02d} [{rec.get('category','original')}]", flush=True)
        print(f"      STATE: {rec.get('system','')}", flush=True)
        print(f"      PLAYER: {rec.get('user','')}", flush=True)
        print(f"      MODEL:  {text}", flush=True)
        print(f"      TARGET: {target}", flush=True)
        probe_records.append({**rec, "model": text, "target": target, "stop": stop, "kind": "original"})
    rate = passed / max(1, len(gp))
    print(f"  grounding probe score: {passed}/{len(gp)} = {rate*100:.1f}%", flush=True)
    return {"passed": passed, "count": len(gp), "pass_rate": rate, "records": probe_records}


def validate_all(model, ground_val, original_val, uo_val, tiny_val, cat_names, args, dtype, device):
    return {
        "grounding": eval_masked(model, ground_val, args.val_batch_size, args.ground_val_examples, dtype, device),
        "original_sft": eval_masked(model, original_val, args.val_batch_size, args.original_val_examples, dtype, device, cat_names),
        "uo_story": eval_lm(model, uo_val, args.lm_val_batch_size, args.uo_val_batches, dtype, device) if uo_val else None,
        "tinystories": eval_lm(model, tiny_val, args.lm_val_batch_size, args.tiny_val_batches, dtype, device) if tiny_val else None,
    }


def print_metric(name, m):
    if m is None:
        print(f"  {name}: unavailable", flush=True)
        return
    if "token_accuracy" in m:
        print(
            f"  {name}: loss={m['loss']:.4f} ppl={m['ppl']:.3f} "
            f"token_acc={m['token_accuracy']*100:.2f}% exact={m['exact_accuracy']*100:.2f}% "
            f"n={m['examples']:,}", flush=True
        )
    else:
        print(f"  {name}: loss={m['loss']:.4f} ppl={m['ppl']:.3f} tokens={m['tokens']:,}", flush=True)


def balanced_score(metrics, baseline, probe_rate, args):
    score = float(metrics["grounding"]["loss"])
    score += args.grounding_generation_penalty * (1.0 - float(probe_rate))
    for key, weight in (
        ("original_sft", args.original_loss_penalty),
        ("uo_story", args.uo_loss_penalty),
        ("tinystories", args.tiny_loss_penalty),
    ):
        if metrics.get(key) is not None and baseline.get(key) is not None:
            score += weight * max(0.0, float(metrics[key]["loss"]) - float(baseline[key]["loss"]))
    return score


# -----------------------------------------------------------------------------
# Optimizer / LR / checkpoints
# -----------------------------------------------------------------------------

def build_optimizer(model, max_lr, wd, embed_scale, geometry_scale):
    specs = {
        "core": dict(params=[], weight_decay=wd, lr_scale=1.0),
        "nodecay": dict(params=[], weight_decay=0.0, lr_scale=1.0),
        "embedding": dict(params=[], weight_decay=0.0, lr_scale=embed_scale),
        "geometry": dict(params=[], weight_decay=0.0, lr_scale=geometry_scale),
    }
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        low = name.lower()
        if name in {"token_embedding.weight", "lm_head.weight"}:
            key = "embedding"
        elif "raw_gain" in low:
            key = "geometry"
        elif p.ndim < 2 or "norm" in low:
            key = "nodecay"
        else:
            key = "core"
        if specs[key]["lr_scale"] > 0:
            specs[key]["params"].append(p)
        else:
            p.requires_grad_(False)
    groups = []
    for name, s in specs.items():
        if s["params"]:
            groups.append({
                "params": s["params"], "weight_decay": s["weight_decay"],
                "lr": max_lr * s["lr_scale"], "lr_scale": s["lr_scale"], "group_name": name,
            })
    try:
        opt = torch.optim.AdamW(groups, lr=max_lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
        print("Optimizer: fused AdamW", flush=True)
    except Exception as e:
        print(f"Fused AdamW unavailable ({e}); using foreach.", flush=True)
        opt = torch.optim.AdamW(groups, lr=max_lr, betas=(0.9, 0.95), eps=1e-8, foreach=True)
    print(
        f"LR scales: core=1.00x embedding={embed_scale:.2f}x geometry={geometry_scale:.2f}x",
        flush=True,
    )
    return opt


def lr_at(step, total, max_lr, min_lr, warm_frac):
    warm = min(max(5, round(total * warm_frac)), max(1, total // 4))
    if step < warm:
        return max_lr * (step + 1) / warm
    progress = min(1.0, max(0.0, (step - warm) / max(1, total - warm)))
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def set_lr(opt, base):
    for g in opt.param_groups:
        g["lr"] = base * float(g.get("lr_scale", 1.0))


def recursive_cpu(v):
    if torch.is_tensor(v):
        return v.detach().cpu()
    if isinstance(v, dict):
        return {k: recursive_cpu(x) for k, x in v.items()}
    if isinstance(v, list):
        return [recursive_cpu(x) for x in v]
    if isinstance(v, tuple):
        return tuple(recursive_cpu(x) for x in v)
    return v


def optimizer_to(opt, device):
    for state in opt.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def save_ckpt(path, model, opt, scaler, state, cfg, train_cfg, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model_state_dict": recursive_cpu(model.state_dict()),
        "optimizer_state_dict": recursive_cpu(opt.state_dict()),
        "scaler_state_dict": scaler.state_dict(),
        "state": state,
        "model_config": asdict(cfg),
        "training_config": train_cfg,
        "history": history,
        "rng": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
        },
    }, tmp)
    tmp.replace(path)


def restore_rng(payload):
    try:
        r = payload["rng"]
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        torch.set_rng_state(r["torch"])
        torch.cuda.set_rng_state_all(r["cuda"])
    except Exception:
        pass


def backup(path, folder):
    if folder:
        d = Path(folder)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, d / path.name)


def maybe_compile(loss_model, total_steps, args):
    if args.no_compile or total_steps < args.compile_min_steps:
        reason = "disabled" if args.no_compile else f"only {total_steps} steps"
        print(f"torch.compile skipped ({reason}).", flush=True)
        return loss_model, False
    try:
        print(f"Compiling with mode={args.compile_mode!r}...", flush=True)
        return torch.compile(loss_model, mode=args.compile_mode, dynamic=False), True
    except Exception as e:
        print(f"Compile failed; eager fallback: {type(e).__name__}: {e}", flush=True)
        return loss_model, False


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train(args, model, cfg, source_ckpt, tokenizer_path, tok, meta, uo_val, tiny_val, outdir, dtype):
    device = torch.device("cuda")
    ground_train = ArrayDataset(meta.ground_train_inputs, meta.ground_train_labels, group_path=meta.ground_train_groups, ram=True)
    ground_val = ArrayDataset(meta.ground_val_inputs, meta.ground_val_labels, group_path=meta.ground_val_groups, ram=True)
    original_train = ArrayDataset(meta.original_train_inputs, meta.original_train_labels, category_path=meta.original_train_categories, ram=True)
    original_val = ArrayDataset(meta.original_val_inputs, meta.original_val_labels, category_path=meta.original_val_categories, ram=True)
    ground_manifest = load_jsonl_manifest(meta.ground_val_manifest)
    original_manifest = load_jsonl_manifest(meta.original_val_manifest)

    batch_size = int(args.batch_size)
    schedules = [
        build_schedule(e, ground_train, original_train, args.grounding_per_persona, batch_size, args.seed)
        for e in range(args.epochs)
    ]
    total_steps = sum(s.batches for s in schedules)

    print("\nPhase-2 grounding plan\n----------------------", flush=True)
    print(f"source checkpoint:          {source_ckpt}", flush=True)
    print(f"groundable train personas:  {meta.train_personas:,}", flush=True)
    print(f"grounding anchors in bank:  {ground_train.count:,}", flush=True)
    print(f"anchors/persona/epoch:      {args.grounding_per_persona}", flush=True)
    print(f"original SFT rows/epoch:    {original_train.count:,}", flush=True)
    print("behavior/action replay:     0% (disabled by design)", flush=True)
    print(f"batch size:                 {batch_size}", flush=True)
    print(f"epochs:                     {args.epochs}", flush=True)
    print(f"optimizer steps:            {total_steps:,}", flush=True)
    print(f"core max LR:                {args.lr:.2e}", flush=True)
    print(f"core min LR:                {args.min_lr:.2e}", flush=True)
    for e, s in enumerate(schedules, 1):
        frac = s.grounding_rows / max(1, s.total_rows)
        print(
            f"  epoch {e}: grounding={s.grounding_rows:,} original={s.original_rows:,} "
            f"ground_share={frac*100:.1f}% batches={s.batches:,}", flush=True
        )

    opt = build_optimizer(model, args.lr, args.weight_decay, args.embedding_lr_scale, args.geometry_lr_scale)
    scaler = grad_scaler(dtype)
    ckdir = outdir / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    latest = ckdir / "latest.pt"
    best_ground = ckdir / "best_grounding.pt"
    best_bal = ckdir / "best_balanced.pt"
    final = ckdir / "final.pt"
    hist_path = outdir / "training_history.json"
    probes_path = outdir / "generation_probes.json"
    history = []
    all_probe_history = []
    state = dict(
        epoch_index=0, batch_in_epoch=0, global_step=0,
        grounding_rows_seen=0, original_rows_seen=0,
        best_ground_probe=-1.0, best_ground_loss=float("inf"), best_balanced_score=float("inf"),
    )
    train_cfg = dict(
        phase="grounded_conversation_phase2",
        grounding_version=GROUNDING_VERSION,
        source_checkpoint=str(source_ckpt), tokenizer=str(tokenizer_path),
        dataset_fingerprint=meta.fingerprint,
        epochs=args.epochs, batch_size=batch_size,
        grounding_per_persona=args.grounding_per_persona,
        behavior_replay_fraction=0.0,
        lr=args.lr, min_lr=args.min_lr, warmup_fraction=args.warmup_fraction,
        weight_decay=args.weight_decay,
        embedding_lr_scale=args.embedding_lr_scale,
        geometry_lr_scale=args.geometry_lr_scale,
        grad_clip=args.grad_clip, total_steps=total_steps,
    )

    resume = None
    if args.resume == "auto" and latest.exists():
        resume = torch.load(latest, map_location="cpu", weights_only=False)
    elif args.resume not in {"auto", "none"}:
        p = Path(args.resume)
        if not p.exists():
            raise FileNotFoundError(p)
        resume = torch.load(p, map_location="cpu", weights_only=False)
    if resume is not None:
        old = resume.get("training_config", {})
        if old.get("dataset_fingerprint") != meta.fingerprint:
            raise RuntimeError("Cannot resume: Phase-2 dataset/tokenizer changed")
        if int(old.get("batch_size", -1)) != batch_size:
            raise RuntimeError("Cannot resume with a different batch size")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        opt.load_state_dict(resume["optimizer_state_dict"])
        optimizer_to(opt, device)
        scaler.load_state_dict(resume.get("scaler_state_dict", {}))
        state.update(resume["state"])
        history = list(resume.get("history", []))
        restore_rng(resume)
        if probes_path.exists():
            try:
                all_probe_history = json.loads(probes_path.read_text())
            except Exception:
                all_probe_history = []
        print(
            f"Resuming epoch {int(state['epoch_index'])+1}, batch {state['batch_in_epoch']}, "
            f"global {state['global_step']:,}", flush=True
        )

    cat_names = {v: k for k, v in meta.category_map.items()}
    if int(state["global_step"]) == 0:
        print("\nPre-Phase-2 validation baseline:", flush=True)
        baseline = validate_all(model, ground_val, original_val, uo_val, tiny_val, cat_names, args, dtype, device)
        print_metric("Grounding val", baseline["grounding"])
        print_metric("Original SFT val", baseline["original_sft"])
        print_metric("UO story val", baseline["uo_story"])
        print_metric("TinyStories val", baseline["tinystories"])
        probe = run_generation_probes(
            model, tok, ground_val, original_val, ground_manifest, original_manifest,
            args, dtype, device, "baseline"
        )
        all_probe_history.append({"label": "baseline", **probe})
        probes_path.write_text(json.dumps(all_probe_history, indent=2, ensure_ascii=False))
        history.append({
            "event": "baseline", "global_step": 0, "metrics": baseline,
            "probe_pass_rate": probe["pass_rate"], "geometry": model.geometry_gain_summary(),
        })
        hist_path.write_text(json.dumps(history, indent=2))
    else:
        base = [h for h in history if h.get("event") == "baseline"]
        if not base:
            raise RuntimeError("Resume history has no validation baseline")
        # Recompute baseline losses only if the historical record lacks them.
        baseline = base[0]["metrics"]

    wrapper = SequenceBalancedLoss(model)
    train_model, compiled = maybe_compile(wrapper, total_steps, args)
    train_cfg["compile_used"] = compiled
    xgpu = torch.empty((batch_size, 128), dtype=torch.long, device=device)
    ygpu = torch.empty_like(xgpu)
    if compiled:
        print("Warming compiled graph...", flush=True)
        xgpu.fill_(tok.token_to_id("<PAD>"))
        ygpu.fill_(-100)
        xgpu[0].copy_(torch.as_tensor(np.asarray(ground_train.inputs[0], dtype=np.int64), dtype=torch.long, device=device))
        ygpu[0].copy_(torch.as_tensor(np.asarray(ground_train.labels[0], dtype=np.int64), dtype=torch.long, device=device))
        opt.zero_grad(set_to_none=True)
        with autocast(dtype):
            wl = train_model(xgpu, ygpu)
        scaler.scale(wl).backward()
        opt.zero_grad(set_to_none=True)
        del wl
        torch.cuda.synchronize()
        print("Compile warmup complete.", flush=True)

    pad = tok.token_to_id("<PAD>")
    start_epoch = int(state["epoch_index"])
    start_batch = int(state["batch_in_epoch"])
    global_step = int(state["global_step"])
    ground_seen = int(state["grounding_rows_seen"])
    original_seen = int(state["original_rows_seen"])
    best_probe = float(state["best_ground_probe"])
    best_ground_loss = float(state["best_ground_loss"])
    best_balanced = float(state["best_balanced_score"])
    loss_acc = torch.zeros((), dtype=torch.float32, device=device)
    loss_count = 0
    rows_log = 0
    log_start = time.perf_counter()
    model.train()

    for epoch in range(start_epoch, args.epochs):
        sched = schedules[epoch]
        sb = start_batch if epoch == start_epoch else 0
        print("\n" + "=" * 78, flush=True)
        print(
            f"PHASE-2 GROUNDING EPOCH {epoch+1}/{args.epochs} | "
            f"grounding={sched.grounding_rows:,} | original={sched.original_rows:,}", flush=True
        )
        print(f"Starting batch {sb:,}/{sched.batches:,}", flush=True)
        print("=" * 78, flush=True)
        pref = Prefetcher(sched, ground_train, original_train, batch_size, pad, sb, args.prefetch_buffers)
        epoch_ground = epoch_orig = 0
        epoch_start = time.perf_counter()
        for batch in pref:
            lr = lr_at(global_step, total_steps, args.lr, args.min_lr, args.warmup_fraction)
            set_lr(opt, lr)
            xgpu.copy_(batch.x, non_blocking=False)
            ygpu.copy_(batch.y, non_blocking=False)
            pref.release(batch)
            opt.zero_grad(set_to_none=True)
            if compiled and hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            with autocast(dtype):
                loss = train_model(xgpu, ygpu)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, foreach=True)
            scaler.step(opt)
            scaler.update()
            loss_acc.add_(loss.detach().float())
            del loss
            loss_count += 1
            global_step += 1
            ground_seen += batch.grounding
            original_seen += batch.original
            epoch_ground += batch.grounding
            epoch_orig += batch.original
            rows_log += batch.real
            next_batch = batch.batch_index + 1

            if global_step % args.log_every == 0:
                torch.cuda.synchronize()
                now = time.perf_counter()
                mean = float((loss_acc / max(1, loss_count)).item())
                elapsed = max(1e-9, now - log_start)
                print(
                    f"epoch={epoch+1}/{args.epochs} batch={next_batch:,}/{sched.batches:,} "
                    f"({100*next_batch/sched.batches:5.1f}%) global={global_step:,}/{total_steps:,} "
                    f"loss={mean:.4f} ppl={math.exp(min(mean,20)):.2f} lr={lr:.2e} "
                    f"rows/s={rows_log/elapsed:,.0f}", flush=True
                )
                loss_acc.zero_()
                loss_count = 0
                rows_log = 0
                log_start = now

            if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                torch.cuda.synchronize()
                st = dict(
                    epoch_index=epoch, batch_in_epoch=next_batch, global_step=global_step,
                    grounding_rows_seen=ground_seen, original_rows_seen=original_seen,
                    best_ground_probe=best_probe, best_ground_loss=best_ground_loss,
                    best_balanced_score=best_balanced,
                )
                save_ckpt(latest, model, opt, scaler, st, cfg, train_cfg, history)
                backup(latest, args.backup_dir)
                print(f"Saved resume checkpoint at step {global_step:,}", flush=True)

        torch.cuda.synchronize()
        elapsed = max(1e-9, time.perf_counter() - epoch_start)
        tail = None
        if loss_count:
            tail = float((loss_acc / loss_count).item())
            loss_acc.zero_()
            loss_count = 0
        print(
            f"\nEpoch {epoch+1} complete: grounding rows={epoch_ground:,} original rows={epoch_orig:,} "
            f"rows/s={(epoch_ground+epoch_orig)/elapsed:,.0f}" +
            (f" tail_loss={tail:.4f}" if tail is not None else ""), flush=True
        )
        print("Validation:", flush=True)
        metrics = validate_all(model, ground_val, original_val, uo_val, tiny_val, cat_names, args, dtype, device)
        print_metric("Grounding val", metrics["grounding"])
        print_metric("Original SFT val", metrics["original_sft"])
        print_metric("UO story val", metrics["uo_story"])
        print_metric("TinyStories val", metrics["tinystories"])
        probe = run_generation_probes(
            model, tok, ground_val, original_val, ground_manifest, original_manifest,
            args, dtype, device, f"epoch {epoch+1}"
        )
        all_probe_history.append({"label": f"epoch_{epoch+1}", **probe})
        probes_path.write_text(json.dumps(all_probe_history, indent=2, ensure_ascii=False))
        score = balanced_score(metrics, baseline, probe["pass_rate"], args)
        geom = model.geometry_gain_summary()
        print(
            f"  balanced score={score:.4f} grounding_probe={probe['pass_rate']*100:.1f}% "
            f"geometry_gain_mean={geom['mean']:.4f}", flush=True
        )

        record = {
            "event": "epoch_complete", "epoch": epoch + 1, "global_step": global_step,
            "metrics": metrics, "grounding_probe": {k: v for k, v in probe.items() if k != "records"},
            "balanced_score": score, "geometry": geom,
        }
        history.append(record)
        hist_path.write_text(json.dumps(history, indent=2))

        g_loss = float(metrics["grounding"]["loss"])
        better_ground = (
            probe["pass_rate"] > best_probe + 1e-12
            or (abs(probe["pass_rate"] - best_probe) <= 1e-12 and g_loss < best_ground_loss)
        )
        next_state = dict(
            epoch_index=epoch + 1, batch_in_epoch=0, global_step=global_step,
            grounding_rows_seen=ground_seen, original_rows_seen=original_seen,
            best_ground_probe=max(best_probe, probe["pass_rate"]),
            best_ground_loss=min(best_ground_loss, g_loss if better_ground else best_ground_loss),
            best_balanced_score=min(best_balanced, score),
        )
        ep_path = ckdir / f"epoch_{epoch+1}.pt"
        if better_ground:
            best_probe = probe["pass_rate"]
            best_ground_loss = g_loss
            next_state["best_ground_probe"] = best_probe
            next_state["best_ground_loss"] = best_ground_loss
            save_ckpt(best_ground, model, opt, scaler, next_state, cfg, train_cfg, history)
            backup(best_ground, args.backup_dir)
            print(
                f"New best grounding checkpoint: probe={best_probe*100:.1f}% loss={best_ground_loss:.4f}",
                flush=True,
            )
        if score < best_balanced:
            best_balanced = score
            next_state["best_balanced_score"] = best_balanced
            save_ckpt(best_bal, model, opt, scaler, next_state, cfg, train_cfg, history)
            backup(best_bal, args.backup_dir)
            print(f"New best balanced conversational score {best_balanced:.4f}", flush=True)
        save_ckpt(ep_path, model, opt, scaler, next_state, cfg, train_cfg, history)
        save_ckpt(latest, model, opt, scaler, next_state, cfg, train_cfg, history)
        backup(ep_path, args.backup_dir)
        backup(latest, args.backup_dir)
        start_batch = 0

    final_state = dict(
        epoch_index=args.epochs, batch_in_epoch=0, global_step=global_step,
        grounding_rows_seen=ground_seen, original_rows_seen=original_seen,
        best_ground_probe=best_probe, best_ground_loss=best_ground_loss,
        best_balanced_score=best_balanced,
    )
    save_ckpt(final, model, opt, scaler, final_state, cfg, train_cfg, history)
    backup(final, args.backup_dir)
    summary = {
        "source_checkpoint": str(source_ckpt),
        "model_config": asdict(cfg),
        "training_config": train_cfg,
        "final_state": final_state,
        "geometry": model.geometry_gain_summary(),
        "best_grounding_checkpoint": str(best_ground) if best_ground.exists() else None,
        "best_balanced_checkpoint": str(best_bal) if best_bal.exists() else None,
        "final_checkpoint": str(final),
        "history": str(hist_path),
        "generation_probes": str(probes_path),
    }
    (outdir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 78, flush=True)
    print("PHASE-2 GROUNDED CONVERSATION SFT COMPLETE", flush=True)
    print("=" * 78, flush=True)
    print(f"best grounding: {best_ground}", flush=True)
    print(f"best balanced:  {best_bal}", flush=True)
    print(f"final:          {final}", flush=True)
    print("Recommended first candidate: best_balanced.pt", flush=True)
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase-2 grounded conversational SFT for UO-Mind")
    p.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    p.add_argument("--sft-dir", default=str(DEFAULT_SFT_DIR))
    p.add_argument("--checkpoint", default="auto")
    p.add_argument("--tokenizer", default="auto")
    p.add_argument("--output-dir", default="")

    # More optimizer decisions than Phase 1, but substantially gentler updates.
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--grounding-per-persona", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.08)
    p.add_argument("--lr", type=float, default=1.2e-5)
    p.add_argument("--min-lr", type=float, default=1.2e-6)
    p.add_argument("--warmup-fraction", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--embedding-lr-scale", type=float, default=0.50)
    p.add_argument("--geometry-lr-scale", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--max-target-tokens", type=int, default=56)
    p.add_argument("--max-system-tokens", type=int, default=40)
    p.add_argument("--prefetch-buffers", type=int, default=3)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--compile-min-steps", type=int, default=600)
    p.add_argument(
        "--compile-mode", default="max-autotune-no-cudagraphs",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
    )
    p.add_argument("--no-compile", action="store_true")

    p.add_argument("--val-batch-size", type=int, default=256)
    p.add_argument("--ground-val-examples", type=int, default=0)
    p.add_argument("--original-val-examples", type=int, default=0)
    p.add_argument("--lm-val-batch-size", type=int, default=256)
    p.add_argument("--uo-val-batches", type=int, default=64)
    p.add_argument("--tiny-val-batches", type=int, default=32)

    p.add_argument("--ground-probe-count", type=int, default=8)
    p.add_argument("--original-probe-count", type=int, default=4)
    p.add_argument("--probe-max-new-tokens", type=int, default=40)

    # Checkpoint-selection penalties. No synthetic action/behavior metric exists here.
    p.add_argument("--grounding-generation-penalty", type=float, default=1.5)
    p.add_argument("--original-loss-penalty", type=float, default=0.35)
    p.add_argument("--uo-loss-penalty", type=float, default=0.10)
    p.add_argument("--tiny-loss-penalty", type=float, default=0.05)

    p.add_argument("--resume", default="auto")
    p.add_argument("--backup-dir", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.grounding_per_persona <= 0:
        raise ValueError("grounding-per-persona must be positive")
    if not 0 < args.val_fraction < 0.5:
        raise ValueError("val_fraction must be in (0, 0.5)")
    if not 0 < args.min_lr <= args.lr:
        raise ValueError("require 0 < min_lr <= lr")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    workspace = Path(args.workspace)
    root = Path(args.sft_dir)
    outdir = Path(args.output_dir) if args.output_dir else workspace / "sft_phase2_grounding"
    outdir.mkdir(parents=True, exist_ok=True)
    print(
        f"Runtime\n-------\ntorch:  {torch.__version__}\ncuda:   {torch.version.cuda}\n"
        f"gpu:    {torch.cuda.get_device_name(0)}", flush=True
    )
    ckpt = find_checkpoint(args.checkpoint, workspace)
    tok_path = find_tokenizer(args.tokenizer, workspace, ckpt)
    tok = Tokenizer.from_file(str(tok_path))
    validate_tokenizer(tok)
    model, _, cfg = load_model(ckpt, tok, torch.device("cuda"))
    files = discover_jsonl(root)
    # Same split seed as Phase 1: base seed 42 + 5000 = 5042 by default.
    meta = prepare_phase2(
        root, files, tok, tok_path, cfg.max_seq_len, args.val_fraction,
        args.max_target_tokens, args.max_system_tokens, outdir, args.seed + 5000,
    )
    if meta.response_truncated:
        print(f"WARNING: {meta.response_truncated:,} responses clipped", flush=True)
    if meta.context_truncated:
        print(f"Context trimming occurred in {meta.context_truncated:,} encoded examples", flush=True)
    if meta.persona_parse_failures:
        print(
            f"NOTE: {meta.persona_parse_failures:,} conversations lacked a conservative "
            "'Your name is ... You are a/an ... from ...' persona parse and therefore contribute "
            "only their original SFT row, not synthetic grounding anchors.", flush=True
        )
    uo_val, tiny_val = optional_lm_validations(workspace)
    dtype = amp_dtype()
    print("\nPhase-2 source summary\n----------------------", flush=True)
    print(f"checkpoint:             {ckpt}", flush=True)
    print(f"tokenizer:              {tok_path}", flush=True)
    print(f"categories:             {len(meta.category_map):,}", flush=True)
    print(f"conversations:          {meta.conversations:,}", flush=True)
    print(f"original train targets: {meta.original_train_examples:,}", flush=True)
    print(f"original val targets:   {meta.original_val_examples:,}", flush=True)
    print(f"ground train bank:      {meta.ground_train_examples:,}", flush=True)
    print(f"ground val bank:        {meta.ground_val_examples:,}", flush=True)
    print(f"train personas:         {meta.train_personas:,}", flush=True)
    print(f"val personas:           {meta.val_personas:,}", flush=True)
    print("runtime contract:       <BOS><STATE>...<PLAYER>...<SAY> -> speech -> <EOS>", flush=True)
    print("action-policy replay:   NONE", flush=True)
    train(args, model, cfg, ckpt, tok_path, tok, meta, uo_val, tiny_val, outdir, dtype)


if __name__ == "__main__":
    main()
