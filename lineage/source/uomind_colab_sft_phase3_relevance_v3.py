"""
uomind_colab_sft_phase3_relevance_v3.py

Revised Phase-3 relevance / answerability trainer for the 128-token UO-Mind
conversational NPC.

This version deliberately BACKTRACKS to the Phase-2 grounded checkpoint.  It does
not continue from the first Phase-3 run.  The first Phase-3 experiment successfully
learned fact retrieval and explicit uncertainty, but its refusal examples were too
strong and too templated: on unseen prompts the model sometimes said "not my trade"
even when STATE explicitly made it the correct expert.

Runtime contract (unchanged):
  <BOS><STATE> NPC facts/persona <PLAYER> current player speech <SAY> response <EOS>

V3 fixes the regression with contrastive routing:
  * every practical question is trained in positive/in-trade settings much more often
    than it is trained as an out-of-trade refusal;
  * the SAME practical intents appear under both matching and non-matching roles;
  * role descriptions are paraphrased ("experienced farmer near...", "keep a small
    inn...", "work as a tanner...") so routing cannot depend on one persona syntax;
  * ordinary town/social questions are positive examples for all roles, teaching that
    not every non-craft question should be refused;
  * uncertainty questions are paired with KNOWN-state contrasts using the SAME player
    wording, so "I know not" must depend on STATE rather than on the question alone;
  * checkpoint selection includes anti-false-refusal regression probes.

No action-policy tokens, hidden numeric state, or multi-turn objective are used.

Default epoch mix (16,000 rows):
  37.5% relevance/routing bank (about 5 positive rows : 1 refusal row)
  18.75% explicit STATE fact retrieval
  18.75% paired answerability bank (half unknown, half known contrast)
  10.0% Phase-2 identity grounding replay
  15.0% conservatively filtered original SFT conversation

Defaults:
  * source: sft_phase2_grounding/checkpoints/best_balanced.pt
  * output: sft_phase3_relevance_v3/
  * 2 epochs, batch 24
  * 4e-6 core max LR -> 4e-7 cosine floor
  * embedding 0.5x LR, geometry 0.025x LR
  * fixed greedy probes plus explicit anti-false-refusal regression probes

Run:
  !python uomind_colab_sft_phase3_relevance_v3.py
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

# -----------------------------------------------------------------------------
# Phase-3 relevance / answerability curriculum
# -----------------------------------------------------------------------------

import re

PHASE3_VERSION = "phase3-relevance-v3-contrastive-routing"
DEFAULT_PHASE3_OUT = DEFAULT_WORKSPACE / "sft_phase3_relevance_v3"


def find_checkpoint(requested: str, workspace: Path) -> Path:
    if requested != "auto":
        p = Path(requested)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    candidates = [
        workspace / "sft_phase2_grounding/checkpoints/best_balanced.pt",
        workspace / "sft_phase2_grounding/checkpoints/best_grounding.pt",
        workspace / "sft_phase2_grounding/checkpoints/final.pt",
        workspace / "sft_phase2_grounding/checkpoints/latest.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No Phase-2 grounded checkpoint found. Expected "
        "/content/uomind_tinystories/sft_phase2_grounding/checkpoints/best_balanced.pt "
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
class SyntheticRow:
    group_id: int
    kind: str
    subtype: str
    category: str
    system_text: str
    user_text: str
    response: str
    required_any: List[str]
    required_all: List[str]
    forbidden: List[str]
    name: str
    profession: str
    town: str


@dataclass
class Phase3Meta:
    relevance_train_inputs: str
    relevance_train_labels: str
    relevance_train_groups: str
    relevance_val_inputs: str
    relevance_val_labels: str
    relevance_val_groups: str
    fact_train_inputs: str
    fact_train_labels: str
    fact_train_groups: str
    fact_val_inputs: str
    fact_val_labels: str
    fact_val_groups: str
    unknown_train_inputs: str
    unknown_train_labels: str
    unknown_train_groups: str
    unknown_val_inputs: str
    unknown_val_labels: str
    unknown_val_groups: str
    ground_train_inputs: str
    ground_train_labels: str
    ground_train_groups: str
    ground_val_inputs: str
    ground_val_labels: str
    ground_val_groups: str
    clean_train_inputs: str
    clean_train_labels: str
    clean_train_categories: str
    clean_val_inputs: str
    clean_val_labels: str
    clean_val_categories: str
    full_val_inputs: str
    full_val_labels: str
    full_val_categories: str
    relevance_train_manifest: str
    relevance_val_manifest: str
    fact_train_manifest: str
    fact_val_manifest: str
    unknown_train_manifest: str
    unknown_val_manifest: str
    ground_train_manifest: str
    ground_val_manifest: str
    clean_train_manifest: str
    clean_val_manifest: str
    full_val_manifest: str
    category_map: Dict[str, int]
    relevance_train_examples: int
    relevance_val_examples: int
    fact_train_examples: int
    fact_val_examples: int
    unknown_train_examples: int
    unknown_val_examples: int
    ground_train_examples: int
    ground_val_examples: int
    clean_train_examples: int
    clean_val_examples: int
    full_val_examples: int
    train_personas: int
    val_personas: int
    conversations: int
    duplicates: int
    invalid: int
    persona_parse_failures: int
    original_rows_considered: int
    original_rows_kept: int
    original_rows_rejected: int
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
                    invalid += 1; st["invalid"] += 1; continue
                msgs = obj.get("messages")
                if not isinstance(msgs, list):
                    invalid += 1; st["invalid"] += 1; continue
                clean, assistants, ok = [], 0, True
                for m in msgs:
                    if not isinstance(m, dict):
                        ok = False; break
                    role, content = m.get("role"), m.get("content")
                    if role not in allowed or not isinstance(content, str):
                        ok = False; break
                    content = norm_text(content)
                    if not content:
                        ok = False; break
                    clean.append({"role": role, "content": content})
                    assistants += int(role == "assistant")
                if not ok or assistants == 0:
                    invalid += 1; st["invalid"] += 1; continue
                fp = conv_fingerprint(clean)
                if fp in seen:
                    duplicates += 1; st["duplicates"] += 1; continue
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


def compact_state(p: Persona, extra: str = "") -> str:
    # Phase-3 fact/uncertainty rows must keep the *entire* supplied fact inside the
    # same 40-token STATE budget used by the original conversational SFT.  When
    # extra knowledge is present, use a deliberately compact but still natural
    # persona prefix.  Phase-2 grounding replay separately preserves the full
    # "Your name is ... You are ... from ..." form.
    if extra:
        return norm_text(f"{p.name}, {p.profession} from {p.town}. {extra}")
    return norm_text(f"Your name is {p.name}. You are {p.article} {p.profession} from {p.town}.")


def encode_content(tok: Tokenizer, text: str) -> List[int]:
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
            if m["role"] == "assistant":
                rows.append((ci, conv, conv.messages[:mi], m["content"]))
    return rows


# Practical profiles deliberately use broad, ordinary craft knowledge rather than
# setting-specific lore.  Each role has multiple player phrasings so the model learns
# the underlying intent rather than a single lexical trigger.
PRACTICAL_PROFILES = {
    "blacksmith": [
        ("My sword has a small chip near the tip. Can it be mended?", "A small chip can be ground smooth if the steel is not cracked through.", ["chip", "ground", "steel", "crack"]),
        ("I nicked my blade on a stone. Is the point worth repairing?", "Aye, if the blade is sound. A small nick can be ground smooth.", ["blade", "nick", "ground", "smooth"]),
    ],
    "baker": [
        ("My dough sticks to my hands. What should I do?", "Dust thy hands with a little flour and fold the dough gently.", ["flour", "dough", "fold"]),
        ("The dough clings to my fingers whenever I shape it. Any counsel?", "Use a little flour on thy hands and work the dough gently.", ["flour", "hands", "dough", "gently"]),
    ],
    "brewer": [
        ("My ale is cloudy. What should I do with it?", "Let it settle in a clean cask before thou servest it.", ["settle", "cask", "clean"]),
        ("This ale will not clear. Should I serve it yet?", "Nay. Give it time to settle in a clean cask first.", ["settle", "cask", "clean"]),
    ],
    "fisherman": [
        ("When is a good time to cast a line?", "Try the quiet water near dawn, when the fish begin to feed.", ["dawn", "water", "fish"]),
        ("Wouldst thou fish at noon or early morning?", "Early morning is better; the water is quiet and the fish are feeding.", ["morning", "water", "fish"]),
    ],
    "farmer": [
        ("Black clouds are coming. Should I leave cut hay outside?", "Nay. Bring the hay under cover before the rain comes.", ["hay", "cover", "rain", "bring"]),
        ("The wind is cold and rain looks near. Shall I leave the cut hay in the field?", "Nay. Move the hay under cover before the rain reaches it.", ["hay", "cover", "rain", "move"]),
        ("Storm clouds are rolling in. What should I do with the hay I cut?", "Get the hay into a barn or beneath cover while it is still dry.", ["hay", "barn", "cover", "dry"]),
    ],
    "gardener": [
        ("My herb bed is wilting in the heat. What should I do?", "Water the roots early and give the tender plants some shade.", ["water", "roots", "shade"]),
        ("The herbs droop every hot afternoon. How can I help them?", "Water at the roots early and shade the tender plants from the worst heat.", ["water", "roots", "shade"]),
    ],
    "carpenter": [
        ("This chair joint is loose. Can it be repaired?", "Aye. Clean the joint, set it true, and clamp it while the glue dries.", ["joint", "clamp", "glue"]),
        ("A chair wobbles where two pieces meet. How wouldst thou mend it?", "Clean and reset the joint, then clamp it until the glue dries.", ["joint", "clamp", "glue"]),
    ],
    "mason": [
        ("A stone in my wall has come loose. What should I do?", "Reset the stone in sound mortar and keep the joint square.", ["stone", "mortar", "joint"]),
        ("One wall stone shifts when I touch it. How is that repaired?", "Lift it out, bed it in sound mortar, and set the joint true.", ["stone", "mortar", "joint"]),
    ],
    "tanner": [
        ("My leather got soaked in the rain. How should I dry it?", "Dry it slowly in the air, not hard against a hot fire.", ["dry", "air", "fire", "leather"]),
        ("This hide was drenched on the road. Should I put it beside the hearth?", "Nay. Let the leather dry slowly in the air, away from fierce heat.", ["leather", "dry", "air", "heat"]),
    ],
    "weaver": [
        ("There is a tear in this cloth. Can it be mended?", "Aye. Catch the loose threads and weave the patch into sound cloth.", ["threads", "weave", "patch", "cloth"]),
        ("My cloak is torn along the weave. Is it salvageable?", "Aye. Catch the loose threads and work a patch into the sound cloth.", ["threads", "patch", "cloth"]),
    ],
    "cobbler": [
        ("The sole of my boot is worn through. Is it worth mending?", "Aye, if the upper leather is sound. A new sole should serve thee well.", ["sole", "leather", "boot"]),
        ("My boot has a hole through the sole. Must I replace the whole thing?", "Not if the upper is sound. A cobbler can fit a new sole.", ["sole", "upper", "boot"]),
    ],
    "miner": [
        ("I hear stone cracking above the tunnel. Should I keep digging?", "Nay. Leave the face and see that the roof is made safe first.", ["leave", "roof", "safe", "stone"]),
        ("The mine roof has begun to crack. Is it safe to work beneath it?", "Nay. Get clear and make the roof safe before any more digging.", ["roof", "safe", "digging", "clear"]),
    ],
    "cook": [
        ("The stew is too thin. How can I thicken it?", "Let it simmer longer, or stir in a little meal until it thickens.", ["simmer", "meal", "thicken", "stew"]),
        ("My pot of stew is watery. What can save it?", "Let it simmer down, or stir in a little meal to thicken it.", ["simmer", "meal", "thicken"]),
    ],
    "merchant": [
        ("I have little coin. How should I bargain for a fair price?", "Ask the price plainly, offer what thou canst pay, and haggle without insult.", ["price", "offer", "haggle", "pay"]),
        ("A seller asks more coin than I can spare. How should I bargain?", "Offer a fair price plainly and haggle without giving insult.", ["offer", "price", "haggle"]),
    ],
    "shopkeeper": [
        ("How can I tell whether this measure is fair?", "See that the weights balance true before any coin changes hands.", ["weights", "balance", "coin"]),
        ("How do I know a merchant's scales are honest?", "Watch the weights and see that the scales balance true.", ["weights", "scales", "balance"]),
    ],
    "shepherd": [
        ("One sheep is limping. What should I look at first?", "Keep it apart from the flock and look closely at the hoof.", ["hoof", "flock", "apart", "sheep"]),
        ("A ewe favors one leg. Where should I look for trouble?", "Set her apart and inspect the hoof first.", ["apart", "hoof"]),
    ],
    "scribe": [
        ("I smudged fresh ink on the page. What can I do?", "Let the ink dry before touching it again, and scrape only with care.", ["ink", "dry", "scrape"]),
        ("Fresh ink smeared across my writing. Can the page be saved?", "Let it dry first; scrape the worst mark only with great care.", ["dry", "scrape", "ink"]),
    ],
    "healer": [
        ("A traveler is feverish and thirsty. What simple care would you give?", "Give water, rest, and a cool cloth, and watch that the fever does not worsen.", ["water", "rest", "cool", "fever"]),
        ("Someone has a fever and cannot keep comfortable. What should I do first?", "Give water and rest, use a cool cloth, and watch the fever closely.", ["water", "rest", "cool", "fever"]),
    ],
    "cooper": [
        ("This barrel leaks at one stave. Can it be saved?", "Aye. Set the stave true and tighten the hoops before filling it again.", ["stave", "hoops", "barrel"]),
        ("Water seeps between the staves of my barrel. How wouldst thou mend it?", "Set the loose stave true and tighten the hoops before filling it again.", ["stave", "hoops", "barrel"]),
    ],
    "innkeeper": [
        ("I have only three coppers. What can fill my belly?", "A bowl of porridge and a heel of bread should fill thee cheaply.", ["porridge", "bread", "bowl", "food"]),
        ("My purse is nearly empty. What cheap food canst thou serve?", "Porridge and bread are cheap, warm, and filling.", ["porridge", "bread", "cheap", "filling"]),
        ("What is the cheapest filling meal in the house?", "A bowl of porridge with bread will fill thee for little coin.", ["porridge", "bread", "bowl", "coin"]),
    ],
}

ROLE_ALIASES = {
    "blacksmith": ("blacksmith", "smith"),
    "baker": ("baker",), "brewer": ("brewer",), "fisherman": ("fisherman", "fisher"),
    "farmer": ("farmer",), "gardener": ("gardener",), "carpenter": ("carpenter",),
    "mason": ("mason",), "tanner": ("tanner",), "weaver": ("weaver",),
    "cobbler": ("cobbler",), "miner": ("miner",), "cook": ("cook",),
    "merchant": ("merchant", "trader"), "shopkeeper": ("shopkeeper", "shop keeper"),
    "shepherd": ("shepherd",), "scribe": ("scribe",), "healer": ("healer",),
    "cooper": ("cooper",), "innkeeper": ("innkeeper", "inn keeper", "tavern keeper", "hosteler"),
}

GENERAL_CONVERSATION = [
    ("How are you feeling?", "I am well enough, friend. The day has treated me kindly.", ["well", "kindly", "fine", "hale"]),
    ("What do you hear in town at dawn?", "Bells, carts, and birds stir with the dawn.", ["bell", "cart", "bird", "dawn"]),
    ("What thinkest thou of all this rain?", "Rain is welcome when the roofs hold and the fields need it.", ["rain", "roof", "field"]),
    ("Does the road turn muddy after a hard rain?", "Aye. Hard rain can turn packed earth to mud quickly.", ["rain", "mud", "earth", "road"]),
    ("That red moon is unsettling, is it not?", "Aye. A red moon can make even steady folk uneasy.", ["red", "moon", "uneasy"]),
    ("Canst thou spare a kind word?", "Aye. Keep thy courage; hard days do pass.", ["courage", "hard", "pass"]),
]

FALSE_REFUSAL_TERMS = ["not my trade", "ask a", "outside my craft", "better asked of", "would know that better"]


def profession_key(text: str) -> Optional[str]:
    low = text.casefold()
    for key, aliases in ROLE_ALIASES.items():
        if any(alias in low for alias in aliases):
            return key
    return None


def article_for(role: str) -> str:
    return "an" if role[:1].casefold() in "aeiou" else "a"


def role_state(p: Persona, role: str, variant: int) -> str:
    a = article_for(role)
    v = variant % 6
    if role == "innkeeper":
        opts = [
            f"Your name is {p.name}. You are {a} innkeeper from {p.town}.",
            f"Your name is {p.name}. You keep a small inn in {p.town}.",
            f"Your name is {p.name}. You run a tavern in {p.town}.",
            f"{p.name}, innkeeper in {p.town}.",
        ]
        return opts[v % len(opts)]
    if role == "farmer" and v in {1, 4}:
        return f"Your name is {p.name}. You are an experienced farmer near {p.town}."
    opts = [
        f"Your name is {p.name}. You are {a} {role} from {p.town}.",
        f"Your name is {p.name}. You are {a} seasoned {role} near {p.town}.",
        f"Your name is {p.name}. You work as {a} {role} near {p.town}.",
        f"{p.name}, {role} from {p.town}.",
        f"Your name is {p.name}. Your trade is {role}. You live near {p.town}.",
        f"Your name is {p.name}. You make thy living as {a} {role} in {p.town}.",
    ]
    return opts[v]


def choose_other_role(role: str, seed_text: str) -> str:
    keys = sorted(PRACTICAL_PROFILES)
    start = stable_hash(seed_text) % len(keys)
    for j in range(len(keys)):
        k = keys[(start + j) % len(keys)]
        if k != role:
            return k
    raise RuntimeError("need at least two practical roles")


def refusal_answer(target_role: str, variant: int) -> str:
    a = article_for(target_role)
    forms = [
        f"That is better asked of {a} {target_role}.",
        f"I know little of that craft; ask {a} {target_role}.",
        f"A {target_role} would know that better than I.",
        f"That lies outside my craft. Seek {a} {target_role}.",
    ]
    return forms[variant % len(forms)]


def build_personas(conversations: Sequence[Conversation]):
    personas, failures = [], 0
    for ci, conv in enumerate(conversations):
        st = first_system(conv.messages)
        p = parse_persona(st, ci) if st else None
        if p is None:
            failures += 1
        else:
            personas.append(p)
    return personas, failures


def make_relevance_bank(conversations: Sequence[Conversation], seed: int):
    personas, failures = build_personas(conversations)
    rows: List[SyntheticRow] = []
    role_keys = sorted(PRACTICAL_PROFILES)
    for gid, p in enumerate(personas):
        cat = conversations[p.conversation_index].category
        base = stable_hash(f"route|{seed}|{p.name}|{p.profession}|{p.town}")

        # 1) Preserve a positive example for the NPC's actual parsed profession when
        # that profession is in our practical-role set.
        actual = profession_key(p.profession)
        if actual:
            variants = PRACTICAL_PROFILES[actual]
            q, a, req = variants[base % len(variants)]
            rows.append(SyntheticRow(
                gid, "relevance", "practical", cat, p.system_text, q, a,
                req, [], FALSE_REFUSAL_TERMS, p.name, actual, p.town,
            ))
        else:
            rows.append(SyntheticRow(
                gid, "relevance", "practical", cat, p.system_text,
                "What sort of work do folk bring to you?",
                f"They come to me for {p.profession} work.",
                [p.profession], [], FALSE_REFUSAL_TERMS, p.name, p.profession, p.town,
            ))

        # 2-3) Uniform synthetic role coverage.  The same role receives two distinct
        # state phrasings and, where available, two distinct user phrasings.
        target_role = role_keys[base % len(role_keys)]
        variants = PRACTICAL_PROFILES[target_role]
        q0 = (base // 17) % len(variants)
        state0 = (base // 31) % 6
        for j in range(2):
            q, a, req = variants[(q0 + j) % len(variants)]
            st = role_state(p, target_role, (state0 + j) % 6)
            rows.append(SyntheticRow(
                gid, "relevance", "practical_paraphrase", cat, st, q, a,
                req, [], FALSE_REFUSAL_TERMS, p.name, target_role, p.town,
            ))

        # 4-5) Ordinary conversation is answerable by any NPC. These rows explicitly
        # oppose the learned shortcut "non-craft question => refuse".
        for j in range(2):
            q, a, req = GENERAL_CONVERSATION[(base // (43 + 13*j) + j) % len(GENERAL_CONVERSATION)]
            st = role_state(p, target_role, (base // (59 + 17*j) + 2 + j) % 6)
            rows.append(SyntheticRow(
                gid, "relevance", "general", cat, st, q, a,
                req, [], FALSE_REFUSAL_TERMS, p.name, target_role, p.town,
            ))

        # 6) Only one refusal for five positive relevance rows. Crucially, its player
        # question is drawn from the exact same practical intent family used by
        # positive examples elsewhere; only STATE's profession changes.
        ask_role = role_keys[(base // 71 + 5) % len(role_keys)]
        current_role = choose_other_role(ask_role, f"current|{seed}|{gid}|{base}")
        q, _, _ = PRACTICAL_PROFILES[ask_role][(base // 83) % len(PRACTICAL_PROFILES[ask_role])]
        st = role_state(p, current_role, (base // 97) % 6)
        ans = refusal_answer(ask_role, base // 101)
        rows.append(SyntheticRow(
            gid, "relevance", "out_of_trade", cat, st, q, ans,
            [ask_role], [], [], p.name, current_role, p.town,
        ))
    return rows, personas, failures


FACT_TEMPLATES = [
    (
        "East gate: {value}.",
        "What knowest thou of the east gate?",
        ["closed until morning", "open until sundown", "closed for repairs"],
        lambda v: f"The east gate is {v}.",
    ),
    (
        "Ferry leaves {value}.",
        "When does the ferry leave?",
        ["at dawn", "at noon", "at sundown"],
        lambda v: f"The ferry leaves {v}.",
    ),
    (
        "Market closes {value}.",
        "When does the market close?",
        ["at noon", "at sundown", "after the evening bell"],
        lambda v: f"The market closes {v}.",
    ),
    (
        "North road: {value} today.",
        "How fares the north road today?",
        ["flooded", "clear", "blocked by a fallen tree"],
        lambda v: f"The north road is {v} today.",
    ),
    (
        "Silver Hart: {value} tonight.",
        "Can I find lodging at the Silver Hart tonight?",
        ["rooms to spare", "one room left", "no rooms left"],
        lambda v: f"The Silver Hart has {v} tonight.",
    ),
    (
        "Mara the healer: {value} today.",
        "Where can I find Mara the healer today?",
        ["at the shrine", "at the market", "visiting the farms"],
        lambda v: f"Mara the healer is {v} today.",
    ),
    (
        "Bread costs {value} today.",
        "What does bread cost today?",
        ["two coppers", "three coppers", "four coppers"],
        lambda v: f"Bread costs {v} today.",
    ),
    (
        "Weather before dusk: {value}.",
        "What weather is expected before dusk?",
        ["heavy rain", "a cold wind", "clear weather"],
        lambda v: f"{v.capitalize()} is expected before dusk.",
    ),
]


def make_fact_bank(conversations: Sequence[Conversation], seed: int):
    personas, failures = build_personas(conversations)
    rows: List[SyntheticRow] = []
    for gid, p in enumerate(personas):
        cat = conversations[p.conversation_index].category
        base = stable_hash(f"fact|{seed}|{p.name}|{p.profession}|{p.town}")
        # Two different fact families per persona. Values vary independently so the
        # same question type has different correct answers across examples.
        tids = [base % len(FACT_TEMPLATES), (base // 17 + 3) % len(FACT_TEMPLATES)]
        if tids[1] == tids[0]:
            tids[1] = (tids[1] + 1) % len(FACT_TEMPLATES)
        for j, ti in enumerate(tids):
            state_tpl, q, values, answer_fn = FACT_TEMPLATES[ti]
            vi = stable_hash(f"value|{seed}|{gid}|{ti}|{j}") % len(values)
            value = values[vi]
            extra = state_tpl.format(value=value)
            st = compact_state(p, extra)
            ans = answer_fn(value)
            req = [value]
            rows.append(SyntheticRow(
                gid, "fact", f"fact_{ti}", cat, st, q, ans,
                req, [], [], p.name, p.profession, p.town,
            ))
    return rows, personas, failures


ANSWERABILITY_TEMPLATES = [
    (
        "You have never travelled far and do not know distant roads.",
        "Road to Serpent Isle: take the east road to the ferry at dawn.",
        "Canst thou tell me the road to Serpent Isle?",
        "I know not. I have never travelled so far.",
        "Aye. Take the east road to the ferry at dawn.",
        ["know not", "do not know", "never travelled"],
        ["east road", "ferry", "dawn"],
    ),
    (
        "You know nothing of magic.",
        "Recall spell: it requires a marked rune and steady focus.",
        "Can you teach me the Recall spell?",
        "I know not. Magic is beyond my learning.",
        "The Recall spell requires a marked rune and steady focus.",
        ["know not", "do not know", "magic"],
        ["marked rune", "focus"],
    ),
    (
        "You know little of medicine and are not a healer.",
        "For a simple fever: give water, rest, and a cool cloth.",
        "What simple care should I give for a fever?",
        "I know not. Ask a healer who knows such things.",
        "Give water and rest, and use a cool cloth while watching the fever.",
        ["know not", "do not know", "healer"],
        ["water", "rest", "cool"],
    ),
    (
        "You know little of the law.",
        "Local law: theft cases are heard by the magistrate.",
        "Who hears a theft case here?",
        "I know not. Ask the magistrate or the town watch.",
        "The magistrate hears theft cases here.",
        ["know not", "do not know", "magistrate", "watch"],
        ["magistrate"],
    ),
    (
        "You have never sailed and know little of navigation.",
        "Night navigation: use the north star to hold a steady course.",
        "Which star should guide me far out at sea?",
        "I know not. I have never sailed those waters.",
        "Use the north star to hold a steady course at night.",
        ["know not", "do not know", "never sailed"],
        ["north star", "course"],
    ),
    (
        "You do not know the old histories here.",
        "Ruined keep: it was built by the old town watch.",
        "Who built the ruined keep beyond the hill?",
        "I know not who built it.",
        "The old town watch built the ruined keep.",
        ["know not", "do not know"],
        ["town watch", "built"],
    ),
]


def make_unknown_bank(conversations: Sequence[Conversation], seed: int):
    personas, failures = build_personas(conversations)
    rows: List[SyntheticRow] = []
    for gid, p in enumerate(personas):
        cat = conversations[p.conversation_index].category
        base = stable_hash(f"answerability|{seed}|{p.name}|{p.profession}|{p.town}")
        tids = [base % len(ANSWERABILITY_TEMPLATES), (base // 23 + 2) % len(ANSWERABILITY_TEMPLATES)]
        if tids[1] == tids[0]:
            tids[1] = (tids[1] + 1) % len(ANSWERABILITY_TEMPLATES)
        for ti in tids:
            unknown_extra, known_extra, q, unknown_a, known_a, unknown_req, known_req = ANSWERABILITY_TEMPLATES[ti]
            rows.append(SyntheticRow(
                gid, "unknown", f"unknown_{ti}", cat, compact_state(p, unknown_extra), q, unknown_a,
                unknown_req, [], [], p.name, p.profession, p.town,
            ))
            # Exact same PLAYER wording; STATE now supplies the answer.  This is the
            # anti-shortcut example that prevents "Serpent Isle" or "Recall spell"
            # from becoming unconditional triggers for "I know not".
            rows.append(SyntheticRow(
                gid, "unknown", f"known_contrast_{ti}", cat, compact_state(p, known_extra), q, known_a,
                known_req, [], ["know not", "do not know", "i don't know", "i cannot say"], p.name, p.profession, p.town,
            ))
    return rows, personas, failures


def choose_wrong_persona(personas: Sequence[Persona], i: int, seed: int) -> Optional[Persona]:
    if len(personas) < 2:
        return None
    start = stable_hash(f"wrong|{seed}|{personas[i].name}|{personas[i].profession}|{personas[i].town}") % len(personas)
    for k in range(1, len(personas) + 1):
        q = personas[(start + k) % len(personas)]
        p = personas[i]
        if q.name.casefold() != p.name.casefold() and q.profession.casefold() != p.profession.casefold() and q.town.casefold() != p.town.casefold():
            return q
    return personas[(i + 1) % len(personas)]


def make_grounding_bank(conversations: Sequence[Conversation], seed: int):
    personas, failures = build_personas(conversations)
    rows: List[SyntheticRow] = []
    for gid, p in enumerate(personas):
        cat = conversations[p.conversation_index].category
        base = [
            ("name", "What is your name?", f"My name is {p.name}.", [p.name], []),
            ("identity", "Who are you?", f"I am {p.name}, {p.article} {p.profession} from {p.town}.", [p.name, p.profession, p.town], []),
            ("profession", "What work do you do?", f"I am {p.article} {p.profession}.", [p.profession], []),
            ("hometown", "Where are you from?", f"I am from {p.town}.", [p.town], []),
        ]
        for subtype, q, a, req, forb in base:
            rows.append(SyntheticRow(gid, "ground", subtype, cat, p.system_text, q, a, req, [], forb, p.name, p.profession, p.town))
        wrong = choose_wrong_persona(personas, gid, seed)
        if wrong:
            rows.extend([
                SyntheticRow(gid, "ground", "false_identity", cat, p.system_text,
                             f"Are you {wrong.name}, {wrong.article} {wrong.profession} from {wrong.town}?",
                             f"Nay. I am {p.name}, {p.article} {p.profession} from {p.town}.",
                             [p.name, p.profession], [], [wrong.name, wrong.profession, wrong.town], p.name, p.profession, p.town),
                SyntheticRow(gid, "ground", "false_profession", cat, p.system_text,
                             f"Are you {wrong.article} {wrong.profession}?",
                             f"Nay. I am {p.article} {p.profession}.",
                             [p.profession], [], [wrong.profession], p.name, p.profession, p.town),
                SyntheticRow(gid, "ground", "false_hometown", cat, p.system_text,
                             f"Are you from {wrong.town}?",
                             f"Nay. I am from {p.town}.",
                             [p.town], [], [wrong.town], p.name, p.profession, p.town),
            ])
    return rows, personas, failures


STOPWORDS = {
    "the","a","an","and","or","but","if","then","to","of","in","on","at","for","from","with","by","as",
    "is","are","was","were","be","been","being","do","does","did","have","has","had","i","you","he","she",
    "it","we","they","me","my","your","thy","thee","thou","his","her","their","our","this","that","these",
    "those","what","where","when","why","how","who","can","could","would","should","will","shall","may","might",
    "yes","no","aye","nay","not","very","some","any","all","there","here","about","tell","say","said","friend",
}


def word_stem(w: str) -> str:
    w = re.sub(r"[^a-z0-9']+", "", w.casefold())
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith("es"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s"):
        w = w[:-1]
    return w


def content_words(text: str):
    out = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z'-]*", text):
        w = word_stem(raw)
        if len(w) >= 3 and w not in STOPWORDS:
            out.add(w)
    return out


def _proper_key(x: str) -> str:
    return re.sub(r"(?:'s|’s)$", "", x.casefold())


def new_proper_nouns(prompt: str, response: str):
    # Only flag capitalized words that occur *inside* a sentence. Sentence-initial
    # capitalization is ordinary English and was far too aggressive in early tests.
    known = {_proper_key(x) for x in re.findall(r"\b[A-Z][A-Za-z'’-]+\b", prompt)}
    ignored = {"I","Aye","Nay","The","A","An","If","When","Where","What","Who","How","Some","Many","Every","Only","My","Thou","This","That"}
    out = []
    for mt in re.finditer(r"\b[A-Z][A-Za-z'’-]+\b", response):
        x = mt.group(0)
        if x in ignored:
            continue
        prefix = response[:mt.start()].rstrip()
        if not prefix or prefix[-1:] in ".!?":
            continue
        if _proper_key(x) not in known:
            out.append(x)
    return out


def clean_original_row(row) -> Tuple[bool, str]:
    _, conv, preceding, response = row
    # Phase 3 is single-turn. Do not replay multi-turn examples here.
    users = [m for m in preceding if m["role"] == "user"]
    assistants = [m for m in preceding if m["role"] == "assistant"]
    if len(users) != 1 or assistants:
        return False, "multi_turn"
    system = first_system(preceding) or ""
    user = users[-1]["content"]
    prompt = system + " " + user
    if len(response.split()) > 32:
        return False, "long_response"
    # Avoid targets that introduce obvious hidden named entities or exact numeric facts
    # absent from the input. This is intentionally conservative.
    if new_proper_nouns(prompt, response):
        return False, "new_proper_noun"
    nums_in = set(re.findall(r"\b\d+\b", prompt))
    nums_out = set(re.findall(r"\b\d+\b", response))
    if nums_out - nums_in:
        return False, "new_number"
    uw = content_words(user)
    rw = content_words(response)
    sw = content_words(system)
    overlap = uw & rw
    persona_overlap = sw & rw
    # Question/answer pairs with at least one shared topical word, or responses that
    # explicitly reuse a supplied STATE fact, are safer replay material.
    if not overlap and not persona_overlap:
        return False, "no_supported_overlap"
    # Reject extreme lexical parroting or degenerate very-short targets.
    if len(rw) < 2 and len(response.split()) < 4:
        return False, "too_short"
    return True, "kept"


def filter_original_rows(rows):
    kept, reasons = [], defaultdict(int)
    for row in rows:
        ok, reason = clean_original_row(row)
        reasons[reason] += 1
        if ok:
            kept.append(row)
    return kept, dict(reasons)


def source_fingerprint(files, tokenizer_path, val_fraction, max_target, max_system, seed):
    h = hashlib.sha256()
    h.update(PHASE3_VERSION.encode())
    h.update(sha256(tokenizer_path).encode())
    h.update(f"{val_fraction}|{max_target}|{max_system}|{seed}".encode())
    for p in files:
        h.update(str(p).encode()); h.update(sha256(p).encode())
    return h.hexdigest()


def system_fits(tok: Tokenizer, text: str, max_system: int) -> bool:
    ids = [tok.token_to_id("<STATE>")] + encode_content(tok, text)
    return len(ids) <= max_system


def _fit_synthetic_state(tok: Tokenizer, row: SyntheticRow, max_system: int) -> str:
    """Return a semantically equivalent STATE that fits the SFT system-token cap.

    Never token-truncate fact/unknown STATE because the target may depend on the
    final words.  Instead shorten only redundant persona wording.
    """
    if system_fits(tok, row.system_text, max_system):
        return row.system_text

    # compact_state() uses this prefix for fact/unknown rows.  Preserve everything
    # after it verbatim and progressively shorten only the identity prefix.
    compact_prefix = norm_text(f"{row.name}, {row.profession} from {row.town}.")
    extra = row.system_text
    if extra.startswith(compact_prefix):
        extra = extra[len(compact_prefix):].strip()

    candidates = [
        norm_text(f"{row.name}, {row.profession}, {row.town}. {extra}"),
        norm_text(f"{row.profession} from {row.town}. {extra}"),
        norm_text(f"{row.name} from {row.town}. {extra}"),
        norm_text(extra),
    ]
    for cand in candidates:
        if cand and system_fits(tok, cand, max_system):
            return cand

    raise RuntimeError(
        f"Synthetic STATE cannot fit max-system-tokens={max_system} without "
        f"truncating required content: {row.system_text!r}"
    )


def _save_rows(prefix: str, rows, tok, seq_len, max_target, max_system, outdir, category_map=None):
    n = len(rows)
    x = np.empty((n, seq_len), dtype=np.uint16)
    y = np.empty((n, seq_len), dtype=np.int16)
    c = np.empty((n,), dtype=np.int16) if category_map is not None else None
    groups = np.empty((n,), dtype=np.int32) if rows and isinstance(rows[0], SyntheticRow) else None
    manifest_path = outdir / f"{prefix}_manifest.jsonl"
    context_truncated = response_truncated = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, row in enumerate(rows):
            if isinstance(row, SyntheticRow):
                preceding = [
                    {"role": "system", "content": row.system_text},
                    {"role": "user", "content": row.user_text},
                ]
                response = row.response
                fitted_state = _fit_synthetic_state(tok, row, max_system)
                if fitted_state != row.system_text:
                    preceding[0]["content"] = fitted_state
                xx, yy, ct, rt = encode_assistant(tok, preceding, response, seq_len, max_target, max_system)
                x[i], y[i] = xx, yy
                groups[i] = row.group_id
                rec = asdict(row); rec["row"] = i
                rec["encoded_system_text"] = preceding[0]["content"]
            else:
                ci, conv, preceding, response = row
                xx, yy, ct, rt = encode_assistant(tok, preceding, response, seq_len, max_target, max_system)
                x[i], y[i] = xx, yy
                if c is not None:
                    c[i] = category_map[conv.category]
                last_user = ""
                for m in reversed(preceding):
                    if m["role"] == "user":
                        last_user = m["content"]; break
                rec = {
                    "row": i, "conversation_index": ci, "category": conv.category,
                    "source": conv.source, "line": conv.line,
                    "system": first_system(preceding) or "", "user": last_user,
                    "response": response,
                }
            context_truncated += int(ct)
            response_truncated += int(rt)
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    xp = outdir / f"{prefix}_inputs.npy"; yp = outdir / f"{prefix}_labels.npy"
    np.save(xp, x); np.save(yp, y)
    cp = gp = None
    if c is not None:
        cp = outdir / f"{prefix}_categories.npy"; np.save(cp, c)
    if groups is not None:
        gp = outdir / f"{prefix}_groups.npy"; np.save(gp, groups)
    return xp, yp, cp, gp, manifest_path, {
        "examples": n, "context_truncated": context_truncated,
        "response_truncated": response_truncated,
    }


def prepare_phase3(root, files, tok, tokenizer_path, seq_len, val_fraction, max_target, max_system, outdir, seed):
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = outdir / "phase3_dataset_meta.json"
    fp = source_fingerprint(files, tokenizer_path, val_fraction, max_target, max_system, seed)
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text())
            path_keys = [k for k in Phase3Meta.__dataclass_fields__ if k.endswith(("_inputs","_labels","_groups","_categories","_manifest"))]
            required = [Path(old[k]) for k in path_keys]
            if old.get("fingerprint") == fp and all(p.exists() for p in required):
                print("Phase-3 dataset cache valid; skipping rebuild.", flush=True)
                return Phase3Meta(**{k: old[k] for k in Phase3Meta.__dataclass_fields__})
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

    rel_train, train_personas, rf1 = make_relevance_bank(train_conv, seed + 101)
    rel_val, val_personas, rf2 = make_relevance_bank(val_conv, seed + 103)
    fact_train, _, ff1 = make_fact_bank(train_conv, seed + 211)
    fact_val, _, ff2 = make_fact_bank(val_conv, seed + 223)
    unk_train, _, uf1 = make_unknown_bank(train_conv, seed + 307)
    unk_val, _, uf2 = make_unknown_bank(val_conv, seed + 311)
    grd_train, _, gf1 = make_grounding_bank(train_conv, seed + 401)
    grd_val, _, gf2 = make_grounding_bank(val_conv, seed + 409)
    parse_fail = max(rf1, ff1, uf1, gf1) + max(rf2, ff2, uf2, gf2)

    full_train = expand_conversation_targets(train_conv)
    full_val = expand_conversation_targets(val_conv)
    clean_train, train_filter = filter_original_rows(full_train)
    clean_val, val_filter = filter_original_rows(full_val)
    if not clean_train:
        raise RuntimeError("Conservative original-SFT filter kept zero training rows")
    if not clean_val:
        print("WARNING: clean original validation bank is empty; full original validation remains available.", flush=True)

    print(
        f"SFT conversations={len(conversations):,} train={len(train_conv):,} val={len(val_conv):,} "
        f"duplicates={dup:,} invalid={invalid:,}", flush=True
    )
    print(
        f"Groundable personas: train={len(train_personas):,}/{len(train_conv):,} "
        f"val={len(val_personas):,}/{len(val_conv):,} parse_failures={parse_fail:,}", flush=True
    )
    print(
        f"Synthetic banks: relevance={len(rel_train):,}/{len(rel_val):,} "
        f"fact={len(fact_train):,}/{len(fact_val):,} unknown={len(unk_train):,}/{len(unk_val):,} "
        f"ground={len(grd_train):,}/{len(grd_val):,}", flush=True
    )
    print(
        f"Clean original SFT: train={len(clean_train):,}/{len(full_train):,} "
        f"val={len(clean_val):,}/{len(full_val):,}", flush=True
    )
    print(f"  train filter reasons: {json.dumps(train_filter, sort_keys=True)}", flush=True)
    print(f"  val filter reasons:   {json.dumps(val_filter, sort_keys=True)}", flush=True)

    rt = _save_rows("relevance_train", rel_train, tok, seq_len, max_target, max_system, outdir)
    rv = _save_rows("relevance_val", rel_val, tok, seq_len, max_target, max_system, outdir)
    ft = _save_rows("fact_train", fact_train, tok, seq_len, max_target, max_system, outdir)
    fv = _save_rows("fact_val", fact_val, tok, seq_len, max_target, max_system, outdir)
    ut = _save_rows("unknown_train", unk_train, tok, seq_len, max_target, max_system, outdir)
    uv = _save_rows("unknown_val", unk_val, tok, seq_len, max_target, max_system, outdir)
    gt = _save_rows("ground_train", grd_train, tok, seq_len, max_target, max_system, outdir)
    gv = _save_rows("ground_val", grd_val, tok, seq_len, max_target, max_system, outdir)
    ct = _save_rows("clean_train", clean_train, tok, seq_len, max_target, max_system, outdir, category_map)
    cv = _save_rows("clean_val", clean_val, tok, seq_len, max_target, max_system, outdir, category_map)
    av = _save_rows("full_val", full_val, tok, seq_len, max_target, max_system, outdir, category_map)

    packs = [rt,rv,ft,fv,ut,uv,gt,gv,ct,cv,av]
    context_truncated = sum(z[-1]["context_truncated"] for z in packs)
    response_truncated = sum(z[-1]["response_truncated"] for z in packs)
    meta = dict(
        relevance_train_inputs=str(rt[0]), relevance_train_labels=str(rt[1]), relevance_train_groups=str(rt[3]),
        relevance_val_inputs=str(rv[0]), relevance_val_labels=str(rv[1]), relevance_val_groups=str(rv[3]),
        fact_train_inputs=str(ft[0]), fact_train_labels=str(ft[1]), fact_train_groups=str(ft[3]),
        fact_val_inputs=str(fv[0]), fact_val_labels=str(fv[1]), fact_val_groups=str(fv[3]),
        unknown_train_inputs=str(ut[0]), unknown_train_labels=str(ut[1]), unknown_train_groups=str(ut[3]),
        unknown_val_inputs=str(uv[0]), unknown_val_labels=str(uv[1]), unknown_val_groups=str(uv[3]),
        ground_train_inputs=str(gt[0]), ground_train_labels=str(gt[1]), ground_train_groups=str(gt[3]),
        ground_val_inputs=str(gv[0]), ground_val_labels=str(gv[1]), ground_val_groups=str(gv[3]),
        clean_train_inputs=str(ct[0]), clean_train_labels=str(ct[1]), clean_train_categories=str(ct[2]),
        clean_val_inputs=str(cv[0]), clean_val_labels=str(cv[1]), clean_val_categories=str(cv[2]),
        full_val_inputs=str(av[0]), full_val_labels=str(av[1]), full_val_categories=str(av[2]),
        relevance_train_manifest=str(rt[4]), relevance_val_manifest=str(rv[4]),
        fact_train_manifest=str(ft[4]), fact_val_manifest=str(fv[4]),
        unknown_train_manifest=str(ut[4]), unknown_val_manifest=str(uv[4]),
        ground_train_manifest=str(gt[4]), ground_val_manifest=str(gv[4]),
        clean_train_manifest=str(ct[4]), clean_val_manifest=str(cv[4]), full_val_manifest=str(av[4]),
        category_map=category_map,
        relevance_train_examples=rt[-1]["examples"], relevance_val_examples=rv[-1]["examples"],
        fact_train_examples=ft[-1]["examples"], fact_val_examples=fv[-1]["examples"],
        unknown_train_examples=ut[-1]["examples"], unknown_val_examples=uv[-1]["examples"],
        ground_train_examples=gt[-1]["examples"], ground_val_examples=gv[-1]["examples"],
        clean_train_examples=ct[-1]["examples"], clean_val_examples=cv[-1]["examples"],
        full_val_examples=av[-1]["examples"], train_personas=len(train_personas), val_personas=len(val_personas),
        conversations=len(conversations), duplicates=dup, invalid=invalid, persona_parse_failures=parse_fail,
        original_rows_considered=len(full_train), original_rows_kept=len(clean_train),
        original_rows_rejected=len(full_train)-len(clean_train),
        context_truncated=context_truncated, response_truncated=response_truncated, fingerprint=fp,
    )
    meta_path.write_text(json.dumps({
        **meta, "phase3_version": PHASE3_VERSION, "file_stats": stats,
        "clean_train_filter": train_filter, "clean_val_filter": val_filter,
    }, indent=2))
    return Phase3Meta(**meta)

# -----------------------------------------------------------------------------
# Dataset containers and deterministic epoch schedule
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


def deterministic_sample(count: int, want: int, seed: int):
    if count <= 0 or want <= 0:
        return np.empty((0,), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if want <= count:
        return rng.choice(count, size=want, replace=False).astype(np.int64)
    # Synthetic banks are deliberately larger than defaults, but this fallback keeps
    # custom row-count settings usable without silently changing proportions.
    first = rng.permutation(count).astype(np.int64)
    rest = rng.choice(count, size=want-count, replace=True).astype(np.int64)
    return np.concatenate([first, rest])


def compute_epoch_targets(rows_per_epoch: int, clean_count: int):
    # Outer mix: 75% new Phase-3 core, 10% grounding replay, 15% clean original.
    # Inner core mix: 50% relevance, 25% fact retrieval, 25% uncertainty.
    if rows_per_epoch < 100:
        raise ValueError("rows_per_epoch must be at least 100")
    clean = round(rows_per_epoch * 0.15)
    ground = round(rows_per_epoch * 0.10)
    core = rows_per_epoch - clean - ground
    relevance = round(core * 0.50)
    fact = round(core * 0.25)
    unknown = core - relevance - fact
    # Never duplicate curated originals merely to hit a ratio. Reallocate any
    # shortfall to the three synthetic core banks in 2:1:1 proportions.
    if clean_count < clean:
        short = clean - clean_count
        clean = clean_count
        relevance += short // 2
        fact += (short - short // 2) // 2
        unknown += short - short // 2 - ((short - short // 2) // 2)
    return dict(relevance=relevance, fact=fact, unknown=unknown, ground=ground, clean=clean)


@dataclass
class Schedule:
    domains: np.ndarray  # 0 relevance, 1 fact, 2 unknown, 3 grounding, 4 clean original
    indices: np.ndarray
    counts: Dict[str, int]
    total_rows: int
    batches: int
    batch_size: int


def build_schedule(epoch, datasets: Dict[str, ArrayDataset], rows_per_epoch: int, batch_size: int, seed: int):
    target = compute_epoch_targets(rows_per_epoch, datasets["clean"].count)
    order = ["relevance", "fact", "unknown", "ground", "clean"]
    dom_parts, idx_parts = [], []
    for dom, key in enumerate(order):
        want = target[key]
        if key == "clean":
            # Category interleaving keeps the curated replay broad across domains.
            full = category_interleaved(datasets[key].categories, seed + 997 * (epoch + 1))
            idx = full[:want]
        else:
            idx = deterministic_sample(datasets[key].count, want, seed + 10007 * (epoch + 1) + dom * 911)
        dom_parts.append(np.full(len(idx), dom, dtype=np.uint8))
        idx_parts.append(idx)
    domains = np.concatenate(dom_parts)
    indices = np.concatenate(idx_parts)
    rng = np.random.default_rng(seed + 30013 * (epoch + 1))
    perm = rng.permutation(len(domains))
    domains, indices = domains[perm], indices[perm]
    counts = {k: int((domains == i).sum()) for i, k in enumerate(order)}
    return Schedule(domains, indices, counts, len(domains), math.ceil(len(domains)/batch_size), batch_size)


@dataclass
class Prepared:
    buffer_id: int
    x: torch.Tensor
    y: torch.Tensor
    real: int
    counts: Dict[str, int]
    batch_index: int


class Prefetcher:
    def __init__(self, schedule, datasets, batch_size, pad_id, start_batch, buffers=3):
        self.schedule = schedule
        self.datasets = datasets
        self.bs = batch_size
        self.pad = pad_id
        self.start = start_batch
        self.keys = ["relevance", "fact", "unknown", "ground", "clean"]
        self.buffers = []
        for _ in range(max(2, buffers)):
            x = torch.empty((batch_size, 128), dtype=torch.long, pin_memory=True)
            y = torch.empty_like(x)
            self.buffers.append((x, y, x.numpy(), y.numpy()))
        self.free = queue.Queue(); self.ready = queue.Queue(maxsize=len(self.buffers)); self.error = None
        for i in range(len(self.buffers)):
            self.free.put(i)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            for bi in range(self.start, self.schedule.batches):
                bid = self.free.get()
                x, y, xn, yn = self.buffers[bid]
                xn.fill(self.pad); yn.fill(-100)
                a = bi * self.bs; b = min(a + self.bs, self.schedule.total_rows)
                dom = self.schedule.domains[a:b]; inds = self.schedule.indices[a:b]
                rows = np.arange(len(dom), dtype=np.int64)
                bc = {}
                for di, key in enumerate(self.keys):
                    rr = rows[dom == di]
                    bc[key] = len(rr)
                    if len(rr):
                        self.datasets[key].gather(inds[dom == di], xn, yn, rr)
                self.ready.put(Prepared(bid, x, y, len(dom), bc, bi))
            self.ready.put(None)
        except BaseException as e:
            self.error = e; self.ready.put(None)

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
# Validation and generation probes
# -----------------------------------------------------------------------------

@torch.inference_mode()
def eval_masked(model, ds: ArrayDataset, batch_size, max_examples, dtype, device, cat_names=None):
    was = model.training; model.eval()
    n = ds.count if max_examples <= 0 else min(ds.count, max_examples)
    if n <= 0:
        if was: model.train()
        return None
    loss_sum = 0.0; rows = tok_ok = tok_n = exact = 0
    cat_loss = defaultdict(float); cat_exact = defaultdict(int); cat_n = defaultdict(int)
    for a in range(0, n, batch_size):
        b = min(a + batch_size, n)
        x = torch.as_tensor(np.asarray(ds.inputs[a:b], dtype=np.int64), dtype=torch.long, device=device)
        y = torch.as_tensor(np.asarray(ds.labels[a:b], dtype=np.int64), dtype=torch.long, device=device)
        with autocast(dtype):
            logits = model(x)
            tl = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), y.reshape(-1), ignore_index=-100, reduction="none").view_as(y)
        mask = y.ne(-100); counts = mask.sum(1).clamp_min(1); rl = tl.sum(1) / counts
        pred = logits.argmax(-1); ok = pred.eq(y) & mask; rex = ((~mask) | pred.eq(y)).all(1)
        rlc = rl.float().cpu().numpy(); rec = rex.cpu().numpy()
        loss_sum += float(rlc.sum()); rows += b-a; tok_ok += int(ok.sum()); tok_n += int(mask.sum()); exact += int(rex.sum())
        if ds.categories is not None:
            cats = np.asarray(ds.categories[a:b], dtype=np.int64)
            for i, c in enumerate(cats.tolist()):
                cat_loss[c] += float(rlc[i]); cat_exact[c] += int(rec[i]); cat_n[c] += 1
    result = dict(loss=loss_sum/max(1,rows), token_accuracy=tok_ok/max(1,tok_n), exact_accuracy=exact/max(1,rows), examples=rows, supervised_tokens=tok_n)
    result["ppl"] = math.exp(min(result["loss"], 20.0))
    if ds.categories is not None:
        result["by_category"] = {
            (cat_names.get(c, str(c)) if cat_names else str(c)): {
                "loss": cat_loss[c]/cat_n[c], "exact_accuracy": cat_exact[c]/cat_n[c], "examples": cat_n[c]
            } for c in sorted(cat_n)
        }
    if was: model.train()
    return result


class LMValidation:
    def __init__(self, path, token_count):
        self.path = Path(path); self.n = int(token_count)
        self.data = np.memmap(self.path, mode="r", dtype=np.uint16, shape=(self.n,))
        self.blocks = (self.n - 1) // 128


@torch.inference_mode()
def eval_lm(model, ds: LMValidation, batch_size, max_batches, dtype, device):
    was = model.training; model.eval(); losses = []; tokens = 0
    nb = min(max_batches, math.ceil(ds.blocks / batch_size))
    for bi in range(nb):
        first = bi*batch_size; last = min(first+batch_size, ds.blocks); real = last-first
        if real <= 0: break
        x = np.empty((real,128), np.int64); y = np.empty_like(x)
        for r, block in enumerate(range(first,last)):
            s = block*128; stream = ds.data[s:s+129]; x[r] = stream[:-1]; y[r] = stream[1:]
        xt = torch.as_tensor(x, dtype=torch.long, device=device); yt = torch.as_tensor(y, dtype=torch.long, device=device)
        with autocast(dtype):
            logits = model(xt); loss = F.cross_entropy(logits.reshape(-1,model.config.vocab_size), yt.reshape(-1))
        losses.append(loss.detach()); tokens += real*128
    mean = float(torch.stack(losses).mean())
    if was: model.train()
    return {"loss":mean,"ppl":math.exp(min(mean,20.0)),"tokens":tokens}


def optional_lm_validations(workspace: Path):
    uo = tiny = None
    for name, mp in (("uo", workspace/"uo_curriculum/uo_corpus_meta.json"), ("tiny", workspace/"corpus_meta.json")):
        if not mp.exists(): continue
        try:
            m = json.loads(mp.read_text()); p = m.get("validation_bin"); n = int(m.get("validation_tokens",0))
            if p and Path(p).exists() and n > 128:
                if name == "uo": uo = LMValidation(p,n)
                else: tiny = LMValidation(p,n)
        except Exception:
            pass
    return uo, tiny


def load_jsonl_manifest(path: str):
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out


def supervised_prefix(ds: ArrayDataset, row: int):
    y = np.asarray(ds.labels[row], dtype=np.int64)
    nz = np.flatnonzero(y != -100)
    if len(nz) == 0: raise RuntimeError("row has no supervised target")
    start = int(nz[0]); x = np.asarray(ds.inputs[row], dtype=np.int64)
    return x[:start+1].tolist(), start


def decode_target(tok: Tokenizer, ds: ArrayDataset, row: int):
    y = np.asarray(ds.labels[row], dtype=np.int64); ids = y[y != -100].tolist()
    eos = tok.token_to_id("<EOS>")
    if eos in ids: ids = ids[:ids.index(eos)]
    return tok.decode(ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def greedy_generate(model, tok: Tokenizer, prefix_ids: Sequence[int], max_new_tokens: int, dtype, device):
    was = model.training; model.eval()
    pad = tok.token_to_id("<PAD>"); eos = tok.token_to_id("<EOS>")
    context = list(map(int, prefix_ids)); out = []; stop = "max_new_tokens"
    for _ in range(max_new_tokens):
        if len(context) > model.config.max_seq_len:
            stop = "context_limit"; break
        x = torch.full((1,128), pad, dtype=torch.long, device=device)
        x[0,:len(context)] = torch.tensor(context, dtype=torch.long, device=device)
        with autocast(dtype): logits = model(x)
        nxt = int(logits[0,len(context)-1].argmax().item())
        if nxt == eos:
            stop = "EOS"; break
        out.append(nxt)
        if len(context) == model.config.max_seq_len:
            stop = "context_limit_after_token"; break
        context.append(nxt)
    if was: model.train()
    return out, stop


def norm_for_check(text: str):
    return " ".join(re.sub(r"[^a-z0-9']+", " ", text.casefold()).split())


def synthetic_pass(rec, text: str):
    t = norm_for_check(text)
    any_terms = [norm_for_check(x) for x in rec.get("required_any",[]) if x]
    all_terms = [norm_for_check(x) for x in rec.get("required_all",[]) if x]
    forb = [norm_for_check(x) for x in rec.get("forbidden",[]) if x]
    any_ok = True if not any_terms else any(x in t for x in any_terms)
    all_ok = all(x in t for x in all_terms)
    forb_ok = not any(x in t for x in forb)
    return bool(any_ok and all_ok and forb_ok)


def select_manifest_rows(manifest, count, seed, subtype=None, subtypes=None):
    allowed = set(subtypes) if subtypes is not None else None
    pool = [
        r for r in manifest
        if (subtype is None or r.get("subtype") == subtype)
        and (allowed is None or r.get("subtype") in allowed)
    ]
    if not pool or count <= 0:
        return []
    pool = sorted(pool, key=lambda r: stable_hash(f"probe|{seed}|{r.get('kind','')}|{r.get('subtype','')}|{r.get('row',0)}"))
    return pool[:min(count, len(pool))]


def raw_probe_prefix(tok: Tokenizer, system_text: str, user_text: str, max_system: int = 40):
    preceding = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    prompt, _ = build_prompt(tok, preceding, 128, max_system)
    return prompt


REGRESSION_PROBES = [
    {
        "name": "innkeeper_food",
        "system": "Your name is Nessa. You run a modest inn in Britain.",
        "user": "My purse holds only a few coppers. What cheap meal could fill me?",
        "required_any": ["bread", "porridge", "stew", "soup", "meal", "food", "bowl"],
        "forbidden": ["not my trade", "ask a", "outside my craft"],
        "positive": True,
    },
    {
        "name": "farmer_hay",
        "system": "Your name is Alden. You are a seasoned farmer near Vesper.",
        "user": "Rain is nearly upon us. Must the mown hay come under shelter?",
        "required_any": ["barn", "inside", "cover", "dry", "rain", "bring", "move", "hay"],
        "forbidden": ["not my trade", "ask a farmer", "outside my craft"],
        "positive": True,
    },
    {
        "name": "tanner_paraphrase",
        "system": "Your name is Bryn. Your trade is tanner. You live near Britain.",
        "user": "A hide was soaked in a storm. Would strong hearth heat dry it safely?",
        "required_any": ["dry", "air", "heat", "fire", "leather"],
        "forbidden": ["not my trade", "ask a tanner", "outside my craft"],
        "positive": True,
    },
    {
        "name": "general_dawn",
        "system": "Your name is Lysa. You are a baker from Cove.",
        "user": "What sounds wake the town when morning first breaks?",
        "required_any": ["rooster", "cockcrow", "bell", "hammer", "cart", "shutter", "well", "bird"],
        "forbidden": ["not my trade", "ask a", "outside my craft"],
        "positive": True,
    },
    {
        "name": "cooper_should_defer_hay",
        "system": "Your name is Corin. You are a cooper from Jhelom.",
        "user": "A storm is close. Should mown hay stay out in the field?",
        "required_any": ["farmer"],
        "forbidden": [],
        "positive": False,
    },
    {
        "name": "baker_should_defer_leather",
        "system": "Your name is Wren. You are a baker from Minoc.",
        "user": "A leather cloak is drenched. How ought it be dried?",
        "required_any": ["tanner"],
        "forbidden": [],
        "positive": False,
    },
]


def run_generation_probes(model, tok, val_sets, manifests, args, dtype, device, label):
    print(f"\nGreedy revised Phase-3 conversational probes [{label}]\n" + "-"*52, flush=True)
    groups = [
        ("grounding", "ground", None, None, args.probe_each),
        ("in_trade", "relevance", None, ["practical", "practical_paraphrase"], args.probe_each),
        ("general", "relevance", "general", None, args.probe_each),
        ("out_of_trade", "relevance", "out_of_trade", None, args.probe_each),
        ("fact", "fact", None, None, args.probe_each),
        ("uncertainty", "unknown", None, [s for s in {r.get('subtype') for r in manifests['unknown']} if str(s).startswith('unknown_')], args.probe_each),
        ("known_contrast", "unknown", None, [s for s in {r.get('subtype') for r in manifests['unknown']} if str(s).startswith('known_contrast_')], args.probe_each),
    ]
    records = []; scores = {}; serial = 1
    for label_name, key, subtype, subtypes, count in groups:
        chosen = select_manifest_rows(manifests[key], count, args.seed + 1901 + serial*31, subtype=subtype, subtypes=subtypes)
        passed = 0
        for rec in chosen:
            row = int(rec["row"]); prefix,_ = supervised_prefix(val_sets[key], row)
            ids, stop = greedy_generate(model, tok, prefix, args.probe_max_new_tokens, dtype, device)
            text = tok.decode(ids, skip_special_tokens=True).strip(); target = decode_target(tok, val_sets[key], row)
            ok = synthetic_pass(rec, text); passed += int(ok)
            print(f"  P{serial:02d} [{label_name}/{rec.get('subtype','')}] {'PASS' if ok else 'MISS'}", flush=True)
            print(f"      STATE:  {rec.get('encoded_system_text', rec.get('system_text',''))}", flush=True)
            print(f"      PLAYER: {rec.get('user_text','')}", flush=True)
            print(f"      MODEL:  {text}", flush=True)
            print(f"      TARGET: {target}", flush=True)
            records.append({**rec,"probe_group":label_name,"model":text,"target":target,"pass":bool(ok),"stop":stop})
            serial += 1
        rate = passed/max(1,len(chosen)); scores[label_name] = {"passed":passed,"count":len(chosen),"pass_rate":rate}
        print(f"  {label_name} score: {passed}/{len(chosen)} = {rate*100:.1f}%", flush=True)

    print("\n  Anti-false-refusal regression probes", flush=True)
    reg_pass = reg_pos_pass = reg_neg_pass = 0
    reg_pos_n = reg_neg_n = 0
    for rp in REGRESSION_PROBES:
        prefix = raw_probe_prefix(tok, rp["system"], rp["user"], args.max_system_tokens)
        ids, stop = greedy_generate(model, tok, prefix, args.probe_max_new_tokens, dtype, device)
        text = tok.decode(ids, skip_special_tokens=True).strip()
        rec = {"required_any":rp["required_any"],"required_all":[],"forbidden":rp["forbidden"]}
        ok = synthetic_pass(rec, text)
        reg_pass += int(ok)
        if rp["positive"]:
            reg_pos_n += 1; reg_pos_pass += int(ok)
        else:
            reg_neg_n += 1; reg_neg_pass += int(ok)
        print(f"  R{serial:02d} [{rp['name']}] {'PASS' if ok else 'MISS'}", flush=True)
        print(f"      STATE:  {rp['system']}", flush=True)
        print(f"      PLAYER: {rp['user']}", flush=True)
        print(f"      MODEL:  {text}", flush=True)
        records.append({**rp,"probe_group":"regression","model":text,"pass":bool(ok),"stop":stop})
        serial += 1
    reg_n = len(REGRESSION_PROBES)
    reg_rate = reg_pass/max(1,reg_n)
    reg_pos_rate = reg_pos_pass/max(1,reg_pos_n)
    reg_neg_rate = reg_neg_pass/max(1,reg_neg_n)
    print(f"  regression score: {reg_pass}/{reg_n} = {reg_rate*100:.1f}%", flush=True)
    print(f"    positive/no-false-refusal: {reg_pos_pass}/{reg_pos_n} = {reg_pos_rate*100:.1f}%", flush=True)
    print(f"    negative/defer-correctly:  {reg_neg_pass}/{reg_neg_n} = {reg_neg_rate*100:.1f}%", flush=True)

    chosen = select_manifest_rows(manifests["clean"], args.original_probe_count, args.seed + 8881)
    for rec in chosen:
        row = int(rec["row"]); prefix,_ = supervised_prefix(val_sets["clean"], row)
        ids, stop = greedy_generate(model, tok, prefix, args.probe_max_new_tokens, dtype, device)
        text = tok.decode(ids, skip_special_tokens=True).strip(); target = decode_target(tok, val_sets["clean"], row)
        print(f"  P{serial:02d} [clean_original/{rec.get('category','')}]", flush=True)
        print(f"      STATE:  {rec.get('system','')}", flush=True)
        print(f"      PLAYER: {rec.get('user','')}", flush=True)
        print(f"      MODEL:  {text}", flush=True)
        print(f"      TARGET: {target}", flush=True)
        records.append({**rec,"probe_group":"clean_original","model":text,"target":target,"stop":stop})
        serial += 1

    core_names = ["in_trade","general","out_of_trade","fact","uncertainty","known_contrast"]
    numer = sum(scores[k]["passed"] for k in core_names)
    denom = sum(scores[k]["count"] for k in core_names)
    core_rate = numer/max(1,denom)
    ground_rate = scores["grounding"]["pass_rate"]
    positive_names = ["in_trade","general","known_contrast"]
    pos_num = sum(scores[k]["passed"] for k in positive_names)
    pos_den = sum(scores[k]["count"] for k in positive_names)
    positive_rate = pos_num/max(1,pos_den)
    routing_floor = min(scores["in_trade"]["pass_rate"], scores["general"]["pass_rate"], scores["out_of_trade"]["pass_rate"])
    selection_rate = 0.40*core_rate + 0.25*positive_rate + 0.25*reg_rate + 0.10*routing_floor
    print(f"  CORE revised Phase-3 probe score: {numer}/{denom} = {core_rate*100:.1f}%", flush=True)
    print(f"  positive/no-refusal score:        {pos_num}/{pos_den} = {positive_rate*100:.1f}%", flush=True)
    print(f"  grounding preservation:           {scores['grounding']['passed']}/{scores['grounding']['count']} = {ground_rate*100:.1f}%", flush=True)
    print(f"  routing floor:                    {routing_floor*100:.1f}%", flush=True)
    print(f"  checkpoint selection rate:        {selection_rate*100:.1f}%", flush=True)
    return {
        "groups":scores,"core_pass_rate":core_rate,"ground_pass_rate":ground_rate,
        "positive_pass_rate":positive_rate,"routing_floor":routing_floor,
        "regression_pass_rate":reg_rate,"regression_positive_pass_rate":reg_pos_rate,
        "regression_negative_pass_rate":reg_neg_rate,"selection_rate":selection_rate,
        "records":records,
    }


def validate_all(model, val_sets, full_val, uo_val, tiny_val, cat_names, args, dtype, device):
    return {
        "relevance": eval_masked(model, val_sets["relevance"], args.val_batch_size, args.synthetic_val_examples, dtype, device),
        "fact": eval_masked(model, val_sets["fact"], args.val_batch_size, args.synthetic_val_examples, dtype, device),
        "unknown": eval_masked(model, val_sets["unknown"], args.val_batch_size, args.synthetic_val_examples, dtype, device),
        "ground": eval_masked(model, val_sets["ground"], args.val_batch_size, args.ground_val_examples, dtype, device),
        "clean_original": eval_masked(model, val_sets["clean"], args.val_batch_size, args.clean_val_examples, dtype, device, cat_names),
        "full_original": eval_masked(model, full_val, args.val_batch_size, args.full_original_val_examples, dtype, device, cat_names),
        "uo_story": eval_lm(model, uo_val, args.lm_val_batch_size, args.uo_val_batches, dtype, device) if uo_val else None,
        "tinystories": eval_lm(model, tiny_val, args.lm_val_batch_size, args.tiny_val_batches, dtype, device) if tiny_val else None,
    }


def print_metric(name, m):
    if m is None:
        print(f"  {name}: unavailable", flush=True); return
    if "token_accuracy" in m:
        print(f"  {name}: loss={m['loss']:.4f} ppl={m['ppl']:.3f} token_acc={m['token_accuracy']*100:.2f}% exact={m['exact_accuracy']*100:.2f}% n={m['examples']:,}", flush=True)
    else:
        print(f"  {name}: loss={m['loss']:.4f} ppl={m['ppl']:.3f} tokens={m['tokens']:,}", flush=True)


def core_validation_loss(metrics):
    vals = [metrics[k]["loss"] for k in ("relevance","fact","unknown") if metrics.get(k) is not None]
    return float(sum(vals)/max(1,len(vals)))


def balanced_score(metrics, baseline, probe, args):
    score = core_validation_loss(metrics)
    score += args.core_generation_penalty * (1.0 - probe["core_pass_rate"])
    # Strongly punish the exact regression seen in the first Phase-3 run: a correct
    # expert or ordinary conversational question being falsely rejected as "not my trade".
    score += args.false_refusal_penalty * (1.0 - probe["positive_pass_rate"])
    score += args.regression_generation_penalty * (1.0 - probe["regression_pass_rate"])
    score += args.routing_floor_penalty * (1.0 - probe["routing_floor"])
    score += args.grounding_generation_penalty * (1.0 - probe["ground_pass_rate"])
    for key, weight in (
        ("clean_original", args.clean_original_loss_penalty),
        ("full_original", args.full_original_loss_penalty),
        ("uo_story", args.uo_loss_penalty),
        ("tinystories", args.tiny_loss_penalty),
    ):
        if metrics.get(key) is not None and baseline.get(key) is not None:
            score += weight * max(0.0, float(metrics[key]["loss"]) - float(baseline[key]["loss"]))
    return score

# -----------------------------------------------------------------------------
# Optimizer, LR schedule, checkpoints
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
        seen.add(id(p)); low = name.lower()
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
                "lr": max_lr*s["lr_scale"], "lr_scale": s["lr_scale"], "group_name": name,
            })
    try:
        opt = torch.optim.AdamW(groups, lr=max_lr, betas=(0.9,0.95), eps=1e-8, fused=True)
        print("Optimizer: fused AdamW", flush=True)
    except Exception as e:
        print(f"Fused AdamW unavailable ({e}); using foreach.", flush=True)
        opt = torch.optim.AdamW(groups, lr=max_lr, betas=(0.9,0.95), eps=1e-8, foreach=True)
    print(f"LR scales: core=1.00x embedding={embed_scale:.2f}x geometry={geometry_scale:.2f}x", flush=True)
    return opt


def lr_at(step, total, max_lr, min_lr, warm_frac):
    warm = min(max(5, round(total*warm_frac)), max(1,total//4))
    if step < warm:
        return max_lr*(step+1)/warm
    progress = min(1.0,max(0.0,(step-warm)/max(1,total-warm)))
    return min_lr + (max_lr-min_lr)*0.5*(1+math.cos(math.pi*progress))


def set_lr(opt, base):
    for g in opt.param_groups:
        g["lr"] = base*float(g.get("lr_scale",1.0))


def recursive_cpu(v):
    if torch.is_tensor(v): return v.detach().cpu()
    if isinstance(v, dict): return {k:recursive_cpu(x) for k,x in v.items()}
    if isinstance(v, list): return [recursive_cpu(x) for x in v]
    if isinstance(v, tuple): return tuple(recursive_cpu(x) for x in v)
    return v


def optimizer_to(opt, device):
    for state in opt.state.values():
        for k,v in list(state.items()):
            if torch.is_tensor(v): state[k] = v.to(device)


def save_ckpt(path, model, opt, scaler, state, cfg, train_cfg, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    torch.save({
        "model_state_dict": recursive_cpu(model.state_dict()),
        "optimizer_state_dict": recursive_cpu(opt.state_dict()),
        "scaler_state_dict": scaler.state_dict(),
        "state": state,
        "model_config": asdict(cfg),
        "training_config": train_cfg,
        "history": history,
        "rng": {
            "python":random.getstate(), "numpy":np.random.get_state(),
            "torch":torch.get_rng_state(), "cuda":torch.cuda.get_rng_state_all(),
        },
    }, tmp)
    tmp.replace(path)


def restore_rng(payload):
    try:
        r=payload["rng"]; random.setstate(r["python"]); np.random.set_state(r["numpy"])
        torch.set_rng_state(r["torch"]); torch.cuda.set_rng_state_all(r["cuda"])
    except Exception:
        pass


def backup(path, folder):
    if folder:
        d=Path(folder); d.mkdir(parents=True, exist_ok=True); shutil.copy2(path,d/path.name)


def maybe_compile(loss_model, total_steps, args):
    if args.no_compile or total_steps < args.compile_min_steps:
        reason = "disabled" if args.no_compile else f"only {total_steps} steps"
        print(f"torch.compile skipped ({reason}).", flush=True); return loss_model, False
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
    train_sets = {
        "relevance": ArrayDataset(meta.relevance_train_inputs, meta.relevance_train_labels, group_path=meta.relevance_train_groups, ram=True),
        "fact": ArrayDataset(meta.fact_train_inputs, meta.fact_train_labels, group_path=meta.fact_train_groups, ram=True),
        "unknown": ArrayDataset(meta.unknown_train_inputs, meta.unknown_train_labels, group_path=meta.unknown_train_groups, ram=True),
        "ground": ArrayDataset(meta.ground_train_inputs, meta.ground_train_labels, group_path=meta.ground_train_groups, ram=True),
        "clean": ArrayDataset(meta.clean_train_inputs, meta.clean_train_labels, category_path=meta.clean_train_categories, ram=True),
    }
    val_sets = {
        "relevance": ArrayDataset(meta.relevance_val_inputs, meta.relevance_val_labels, group_path=meta.relevance_val_groups, ram=True),
        "fact": ArrayDataset(meta.fact_val_inputs, meta.fact_val_labels, group_path=meta.fact_val_groups, ram=True),
        "unknown": ArrayDataset(meta.unknown_val_inputs, meta.unknown_val_labels, group_path=meta.unknown_val_groups, ram=True),
        "ground": ArrayDataset(meta.ground_val_inputs, meta.ground_val_labels, group_path=meta.ground_val_groups, ram=True),
        "clean": ArrayDataset(meta.clean_val_inputs, meta.clean_val_labels, category_path=meta.clean_val_categories, ram=True),
    }
    full_val = ArrayDataset(meta.full_val_inputs, meta.full_val_labels, category_path=meta.full_val_categories, ram=True)
    manifests = {
        "relevance": load_jsonl_manifest(meta.relevance_val_manifest),
        "fact": load_jsonl_manifest(meta.fact_val_manifest),
        "unknown": load_jsonl_manifest(meta.unknown_val_manifest),
        "ground": load_jsonl_manifest(meta.ground_val_manifest),
        "clean": load_jsonl_manifest(meta.clean_val_manifest),
    }

    schedules = [build_schedule(e, train_sets, args.rows_per_epoch, args.batch_size, args.seed) for e in range(args.epochs)]
    total_steps = sum(s.batches for s in schedules)
    print("\nRevised Phase-3 contrastive relevance plan\n------------------------------------------", flush=True)
    print(f"source checkpoint:       {source_ckpt}", flush=True)
    print(f"rows per epoch:          {args.rows_per_epoch:,}", flush=True)
    print(f"batch size:              {args.batch_size}", flush=True)
    print(f"epochs:                  {args.epochs}", flush=True)
    print(f"optimizer steps:         {total_steps:,}", flush=True)
    print(f"core max LR:             {args.lr:.2e}", flush=True)
    print(f"core min LR:             {args.min_lr:.2e}", flush=True)
    print("action-policy replay:    NONE", flush=True)
    for e,s in enumerate(schedules,1):
        print(
            f"  epoch {e}: relevance={s.counts['relevance']:,} fact={s.counts['fact']:,} "
            f"unknown={s.counts['unknown']:,} grounding={s.counts['ground']:,} "
            f"clean_original={s.counts['clean']:,} batches={s.batches:,}", flush=True
        )

    opt = build_optimizer(model,args.lr,args.weight_decay,args.embedding_lr_scale,args.geometry_lr_scale)
    scaler = grad_scaler(dtype)
    ckdir = outdir/"checkpoints"; ckdir.mkdir(parents=True, exist_ok=True)
    latest=ckdir/"latest.pt"; best_phase3=ckdir/"best_phase3.pt"; best_bal=ckdir/"best_balanced.pt"; final=ckdir/"final.pt"
    hist_path=outdir/"training_history.json"; probes_path=outdir/"generation_probes.json"
    history=[]; all_probe_history=[]
    state=dict(epoch_index=0,batch_in_epoch=0,global_step=0,rows_seen={k:0 for k in train_sets},best_core_probe=-1.0,best_core_loss=float("inf"),best_balanced_score=float("inf"))
    train_cfg=dict(
        phase="conversation_phase3_relevance_v3", phase3_version=PHASE3_VERSION,
        source_checkpoint=str(source_ckpt), tokenizer=str(tokenizer_path), dataset_fingerprint=meta.fingerprint,
        runtime_contract="<BOS><STATE>...<PLAYER>...<SAY> -> speech -> <EOS>",
        epochs=args.epochs,batch_size=args.batch_size,rows_per_epoch=args.rows_per_epoch,
        mix={"core_phase3":0.75,"grounding":0.10,"clean_original":0.15,"core_inner":{"relevance":0.50,"fact":0.25,"answerability":0.25},"relevance_bank_ratio":"~5 positive : 1 out_of_trade","answerability_bank":"1 unknown : 1 known contrast"},
        action_policy_replay=0.0, lr=args.lr,min_lr=args.min_lr,warmup_fraction=args.warmup_fraction,
        weight_decay=args.weight_decay,embedding_lr_scale=args.embedding_lr_scale,geometry_lr_scale=args.geometry_lr_scale,
        grad_clip=args.grad_clip,total_steps=total_steps,
    )

    resume=None
    if args.resume=="auto" and latest.exists():
        resume=torch.load(latest,map_location="cpu",weights_only=False)
    elif args.resume not in {"auto","none"}:
        p=Path(args.resume)
        if not p.exists(): raise FileNotFoundError(p)
        resume=torch.load(p,map_location="cpu",weights_only=False)
    if resume is not None:
        old=resume.get("training_config",{})
        if old.get("dataset_fingerprint") != meta.fingerprint:
            raise RuntimeError("Cannot resume: Phase-3 dataset/tokenizer changed")
        if int(old.get("batch_size",-1)) != args.batch_size or int(old.get("rows_per_epoch",-1)) != args.rows_per_epoch:
            raise RuntimeError("Cannot resume with different batch-size or rows-per-epoch")
        model.load_state_dict(resume["model_state_dict"],strict=True); opt.load_state_dict(resume["optimizer_state_dict"]); optimizer_to(opt,device)
        scaler.load_state_dict(resume.get("scaler_state_dict",{})); state.update(resume["state"]); history=list(resume.get("history",[])); restore_rng(resume)
        if probes_path.exists():
            try: all_probe_history=json.loads(probes_path.read_text())
            except Exception: all_probe_history=[]
        print(f"Resuming epoch {int(state['epoch_index'])+1}, batch {state['batch_in_epoch']}, global {state['global_step']:,}", flush=True)

    cat_names={v:k for k,v in meta.category_map.items()}
    if int(state["global_step"])==0:
        print("\nPre-Phase-3 validation baseline:", flush=True)
        baseline=validate_all(model,val_sets,full_val,uo_val,tiny_val,cat_names,args,dtype,device)
        for n,k in (("Relevance val","relevance"),("Fact retrieval val","fact"),("Answerability val","unknown"),("Grounding val","ground"),("Clean original val","clean_original"),("Full original SFT val","full_original"),("UO story val","uo_story"),("TinyStories val","tinystories")):
            print_metric(n,baseline[k])
        probe=run_generation_probes(model,tok,val_sets,manifests,args,dtype,device,"baseline")
        all_probe_history.append({"label":"baseline",**probe}); probes_path.write_text(json.dumps(all_probe_history,indent=2,ensure_ascii=False))
        history.append({"event":"baseline","global_step":0,"metrics":baseline,"probe":{"core":probe["core_pass_rate"],"ground":probe["ground_pass_rate"],"positive":probe["positive_pass_rate"],"regression":probe["regression_pass_rate"],"selection":probe["selection_rate"]},"geometry":model.geometry_gain_summary()})
        hist_path.write_text(json.dumps(history,indent=2))
    else:
        base=[h for h in history if h.get("event")=="baseline"]
        if not base: raise RuntimeError("Resume history has no validation baseline")
        baseline=base[0]["metrics"]

    wrapper=SequenceBalancedLoss(model); train_model,compiled=maybe_compile(wrapper,total_steps,args); train_cfg["compile_used"]=compiled
    xgpu=torch.empty((args.batch_size,128),dtype=torch.long,device=device); ygpu=torch.empty_like(xgpu)
    if compiled:
        print("Warming compiled graph...", flush=True)
        xgpu.fill_(tok.token_to_id("<PAD>")); ygpu.fill_(-100)
        xgpu[0].copy_(torch.as_tensor(np.asarray(train_sets["relevance"].inputs[0],dtype=np.int64),dtype=torch.long,device=device))
        ygpu[0].copy_(torch.as_tensor(np.asarray(train_sets["relevance"].labels[0],dtype=np.int64),dtype=torch.long,device=device))
        opt.zero_grad(set_to_none=True)
        with autocast(dtype): wl=train_model(xgpu,ygpu)
        scaler.scale(wl).backward(); opt.zero_grad(set_to_none=True); del wl; torch.cuda.synchronize()
        print("Compile warmup complete.", flush=True)

    pad=tok.token_to_id("<PAD>"); start_epoch=int(state["epoch_index"]); start_batch=int(state["batch_in_epoch"]); global_step=int(state["global_step"])
    rows_seen={k:int(state.get("rows_seen",{}).get(k,0)) for k in train_sets}
    best_core=float(state["best_core_probe"]); best_core_loss=float(state["best_core_loss"]); best_balanced=float(state["best_balanced_score"])
    loss_acc=torch.zeros((),dtype=torch.float32,device=device); loss_count=0; rows_log=0; log_start=time.perf_counter(); model.train()

    for epoch in range(start_epoch,args.epochs):
        sched=schedules[epoch]; sb=start_batch if epoch==start_epoch else 0
        print("\n"+"="*78, flush=True)
        print(f"PHASE-3 RELEVANCE EPOCH {epoch+1}/{args.epochs} | rows={sched.total_rows:,}", flush=True)
        print(f"Starting batch {sb:,}/{sched.batches:,}", flush=True); print("="*78, flush=True)
        pref=Prefetcher(sched,train_sets,args.batch_size,pad,sb,args.prefetch_buffers)
        epoch_counts={k:0 for k in train_sets}; epoch_start=time.perf_counter()
        for batch in pref:
            lr=lr_at(global_step,total_steps,args.lr,args.min_lr,args.warmup_fraction); set_lr(opt,lr)
            xgpu.copy_(batch.x,non_blocking=False); ygpu.copy_(batch.y,non_blocking=False); pref.release(batch)
            opt.zero_grad(set_to_none=True)
            if compiled and hasattr(torch,"compiler") and hasattr(torch.compiler,"cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            with autocast(dtype): loss=train_model(xgpu,ygpu)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip,foreach=True); scaler.step(opt); scaler.update()
            loss_acc.add_(loss.detach().float()); del loss; loss_count+=1; global_step+=1; rows_log+=batch.real
            for k,v in batch.counts.items(): rows_seen[k]+=v; epoch_counts[k]+=v
            next_batch=batch.batch_index+1
            if global_step % args.log_every == 0:
                torch.cuda.synchronize(); now=time.perf_counter(); mean=float((loss_acc/max(1,loss_count)).item()); elapsed=max(1e-9,now-log_start)
                print(f"epoch={epoch+1}/{args.epochs} batch={next_batch:,}/{sched.batches:,} ({100*next_batch/sched.batches:5.1f}%) global={global_step:,}/{total_steps:,} loss={mean:.4f} ppl={math.exp(min(mean,20)):.2f} lr={lr:.2e} rows/s={rows_log/elapsed:,.0f}", flush=True)
                loss_acc.zero_(); loss_count=0; rows_log=0; log_start=now
            if args.checkpoint_every>0 and global_step%args.checkpoint_every==0:
                torch.cuda.synchronize(); st=dict(epoch_index=epoch,batch_in_epoch=next_batch,global_step=global_step,rows_seen=rows_seen,best_core_probe=best_core,best_core_loss=best_core_loss,best_balanced_score=best_balanced)
                save_ckpt(latest,model,opt,scaler,st,cfg,train_cfg,history); backup(latest,args.backup_dir); print(f"Saved resume checkpoint at step {global_step:,}", flush=True)

        torch.cuda.synchronize(); elapsed=max(1e-9,time.perf_counter()-epoch_start); tail=None
        if loss_count:
            tail=float((loss_acc/loss_count).item()); loss_acc.zero_(); loss_count=0
        print(f"\nEpoch {epoch+1} complete: rows={sum(epoch_counts.values()):,} rows/s={sum(epoch_counts.values())/elapsed:,.0f}" + (f" tail_loss={tail:.4f}" if tail is not None else ""), flush=True)
        print("  rows by pool: " + ", ".join(f"{k}={v:,}" for k,v in epoch_counts.items()), flush=True)
        print("Validation:", flush=True)
        metrics=validate_all(model,val_sets,full_val,uo_val,tiny_val,cat_names,args,dtype,device)
        for n,k in (("Relevance val","relevance"),("Fact retrieval val","fact"),("Answerability val","unknown"),("Grounding val","ground"),("Clean original val","clean_original"),("Full original SFT val","full_original"),("UO story val","uo_story"),("TinyStories val","tinystories")):
            print_metric(n,metrics[k])
        probe=run_generation_probes(model,tok,val_sets,manifests,args,dtype,device,f"epoch {epoch+1}")
        all_probe_history.append({"label":f"epoch_{epoch+1}",**probe}); probes_path.write_text(json.dumps(all_probe_history,indent=2,ensure_ascii=False))
        score=balanced_score(metrics,baseline,probe,args); geom=model.geometry_gain_summary(); core_loss=core_validation_loss(metrics)
        print(f"  balanced score={score:.4f} core_probe={probe['core_pass_rate']*100:.1f}% positive={probe['positive_pass_rate']*100:.1f}% regression={probe['regression_pass_rate']*100:.1f}% grounding={probe['ground_pass_rate']*100:.1f}% geometry_gain_mean={geom['mean']:.4f}", flush=True)
        record={"event":"epoch_complete","epoch":epoch+1,"global_step":global_step,"metrics":metrics,"probe":{"core":probe["core_pass_rate"],"ground":probe["ground_pass_rate"],"positive":probe["positive_pass_rate"],"regression":probe["regression_pass_rate"],"selection":probe["selection_rate"],"groups":probe["groups"]},"balanced_score":score,"geometry":geom}
        history.append(record); hist_path.write_text(json.dumps(history,indent=2))

        better_core = probe["selection_rate"] > best_core + 1e-12 or (abs(probe["selection_rate"]-best_core)<=1e-12 and core_loss < best_core_loss)
        next_state=dict(epoch_index=epoch+1,batch_in_epoch=0,global_step=global_step,rows_seen=rows_seen,best_core_probe=max(best_core,probe["selection_rate"]),best_core_loss=min(best_core_loss,core_loss if better_core else best_core_loss),best_balanced_score=min(best_balanced,score))
        ep_path=ckdir/f"epoch_{epoch+1}.pt"
        if better_core:
            best_core=probe["selection_rate"]; best_core_loss=core_loss; next_state["best_core_probe"]=best_core; next_state["best_core_loss"]=best_core_loss
            save_ckpt(best_phase3,model,opt,scaler,next_state,cfg,train_cfg,history); backup(best_phase3,args.backup_dir)
            print(f"New best revised Phase-3 checkpoint: selection={best_core*100:.1f}% core_loss={best_core_loss:.4f}", flush=True)
        if score < best_balanced:
            best_balanced=score; next_state["best_balanced_score"]=best_balanced
            save_ckpt(best_bal,model,opt,scaler,next_state,cfg,train_cfg,history); backup(best_bal,args.backup_dir)
            print(f"New best balanced score {best_balanced:.4f}", flush=True)
        save_ckpt(ep_path,model,opt,scaler,next_state,cfg,train_cfg,history); save_ckpt(latest,model,opt,scaler,next_state,cfg,train_cfg,history)
        backup(ep_path,args.backup_dir); backup(latest,args.backup_dir); start_batch=0

    final_state=dict(epoch_index=args.epochs,batch_in_epoch=0,global_step=global_step,rows_seen=rows_seen,best_core_probe=best_core,best_core_loss=best_core_loss,best_balanced_score=best_balanced)
    save_ckpt(final,model,opt,scaler,final_state,cfg,train_cfg,history); backup(final,args.backup_dir)
    summary={"source_checkpoint":str(source_ckpt),"model_config":asdict(cfg),"training_config":train_cfg,"final_state":final_state,"geometry":model.geometry_gain_summary(),"best_phase3_checkpoint":str(best_phase3) if best_phase3.exists() else None,"best_balanced_checkpoint":str(best_bal) if best_bal.exists() else None,"final_checkpoint":str(final),"history":str(hist_path),"generation_probes":str(probes_path)}
    (outdir/"training_summary.json").write_text(json.dumps(summary,indent=2))
    print("\n"+"="*78, flush=True); print("REVISED PHASE-3 CONTRASTIVE RELEVANCE SFT COMPLETE", flush=True); print("="*78, flush=True)
    print(f"best Phase-3: {best_phase3}", flush=True); print(f"best balanced: {best_bal}", flush=True); print(f"final:         {final}", flush=True)
    print("Recommended first candidate: best_balanced.pt", flush=True)
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p=argparse.ArgumentParser(description="Revised Phase-3 contrastive relevance / answerability SFT for UO-Mind")
    p.add_argument("--workspace",default=str(DEFAULT_WORKSPACE)); p.add_argument("--sft-dir",default=str(DEFAULT_SFT_DIR)); p.add_argument("--checkpoint",default="auto"); p.add_argument("--tokenizer",default="auto"); p.add_argument("--output-dir",default="")
    p.add_argument("--epochs",type=int,default=2); p.add_argument("--batch-size",type=int,default=24); p.add_argument("--rows-per-epoch",type=int,default=16000)
    p.add_argument("--val-fraction",type=float,default=0.08); p.add_argument("--lr",type=float,default=4e-6); p.add_argument("--min-lr",type=float,default=4e-7); p.add_argument("--warmup-fraction",type=float,default=0.03); p.add_argument("--weight-decay",type=float,default=0.003); p.add_argument("--embedding-lr-scale",type=float,default=0.50); p.add_argument("--geometry-lr-scale",type=float,default=0.025); p.add_argument("--grad-clip",type=float,default=1.0)
    p.add_argument("--max-target-tokens",type=int,default=56); p.add_argument("--max-system-tokens",type=int,default=40); p.add_argument("--prefetch-buffers",type=int,default=3); p.add_argument("--log-every",type=int,default=50); p.add_argument("--checkpoint-every",type=int,default=250)
    p.add_argument("--compile-min-steps",type=int,default=600); p.add_argument("--compile-mode",default="default",choices=("default","reduce-overhead","max-autotune","max-autotune-no-cudagraphs")); p.add_argument("--no-compile",action="store_true")
    p.add_argument("--val-batch-size",type=int,default=256); p.add_argument("--synthetic-val-examples",type=int,default=0); p.add_argument("--ground-val-examples",type=int,default=0); p.add_argument("--clean-val-examples",type=int,default=0); p.add_argument("--full-original-val-examples",type=int,default=0)
    p.add_argument("--lm-val-batch-size",type=int,default=256); p.add_argument("--uo-val-batches",type=int,default=64); p.add_argument("--tiny-val-batches",type=int,default=32)
    p.add_argument("--probe-each",type=int,default=5); p.add_argument("--original-probe-count",type=int,default=5); p.add_argument("--probe-max-new-tokens",type=int,default=40)
    p.add_argument("--core-generation-penalty",type=float,default=1.5); p.add_argument("--false-refusal-penalty",type=float,default=3.0); p.add_argument("--regression-generation-penalty",type=float,default=2.0); p.add_argument("--routing-floor-penalty",type=float,default=1.0); p.add_argument("--grounding-generation-penalty",type=float,default=0.75); p.add_argument("--clean-original-loss-penalty",type=float,default=0.30); p.add_argument("--full-original-loss-penalty",type=float,default=0.15); p.add_argument("--uo-loss-penalty",type=float,default=0.05); p.add_argument("--tiny-loss-penalty",type=float,default=0.02)
    p.add_argument("--resume",default="auto"); p.add_argument("--backup-dir",default=""); p.add_argument("--seed",type=int,default=42)
    return p.parse_args()


def main():
    args=parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    if args.epochs<=0 or args.batch_size<=0 or args.rows_per_epoch<=0: raise ValueError("epochs, batch-size, and rows-per-epoch must be positive")
    if not 0<args.val_fraction<0.5: raise ValueError("val_fraction must be in (0,0.5)")
    if not 0<args.min_lr<=args.lr: raise ValueError("require 0 < min_lr <= lr")
    if args.probe_each<=0: raise ValueError("probe-each must be positive")
    torch.backends.cudnn.benchmark=True; torch.backends.cuda.matmul.allow_tf32=True; torch.set_float32_matmul_precision("high"); set_seed(args.seed)
    workspace=Path(args.workspace); root=Path(args.sft_dir); outdir=Path(args.output_dir) if args.output_dir else workspace/"sft_phase3_relevance_v3"; outdir.mkdir(parents=True,exist_ok=True)
    print(f"Runtime\n-------\ntorch:  {torch.__version__}\ncuda:   {torch.version.cuda}\ngpu:    {torch.cuda.get_device_name(0)}", flush=True)
    ckpt=find_checkpoint(args.checkpoint,workspace); tok_path=find_tokenizer(args.tokenizer,workspace,ckpt); tok=Tokenizer.from_file(str(tok_path)); validate_tokenizer(tok)
    model,_,cfg=load_model(ckpt,tok,torch.device("cuda")); files=discover_jsonl(root)
    meta=prepare_phase3(root,files,tok,tok_path,cfg.max_seq_len,args.val_fraction,args.max_target_tokens,args.max_system_tokens,outdir,args.seed+5000)
    if meta.response_truncated: print(f"WARNING: {meta.response_truncated:,} encoded responses clipped", flush=True)
    if meta.context_truncated: print(f"Context trimming occurred in {meta.context_truncated:,} encoded examples", flush=True)
    if meta.persona_parse_failures: print(f"NOTE: {meta.persona_parse_failures:,} conversations could not generate synthetic Phase-3 examples but remain available to original validation where applicable.", flush=True)
    uo_val,tiny_val=optional_lm_validations(workspace); dtype=amp_dtype()
    print("\nRevised Phase-3 source summary\n------------------------------", flush=True)
    print(f"checkpoint:                {ckpt}", flush=True); print(f"tokenizer:                 {tok_path}", flush=True); print(f"categories:                {len(meta.category_map):,}", flush=True); print(f"conversations:             {meta.conversations:,}", flush=True)
    print(f"relevance bank train/val:  {meta.relevance_train_examples:,}/{meta.relevance_val_examples:,}", flush=True); print(f"fact bank train/val:       {meta.fact_train_examples:,}/{meta.fact_val_examples:,}", flush=True); print(f"unknown bank train/val:    {meta.unknown_train_examples:,}/{meta.unknown_val_examples:,}", flush=True); print(f"ground bank train/val:     {meta.ground_train_examples:,}/{meta.ground_val_examples:,}", flush=True)
    print(f"clean original train/val:  {meta.clean_train_examples:,}/{meta.clean_val_examples:,}", flush=True); print(f"full original val:         {meta.full_val_examples:,}", flush=True); print(f"runtime contract:          <BOS><STATE>...<PLAYER>...<SAY> -> speech -> <EOS>", flush=True); print("action-policy replay:      NONE", flush=True)
    train(args,model,cfg,ckpt,tok_path,tok,meta,uo_val,tiny_val,outdir,dtype)


if __name__ == "__main__":
    main()
