#!/usr/bin/env python3
"""
NPCMicro13M local inference and integrity verification utility.

This script is intentionally self-contained: it embeds the exact architecture
needed to load the stripped final SFT checkpoint.

Examples
--------
1) Run the built-in local sanity/generalization suite:
    python deployment/uomind_infer.py --bundle . --verify-only

2) Interactive single-turn NPC conversation:
    python deployment/uomind_infer.py --bundle . --interactive

3) One-shot generation:
    python deployment/uomind_infer.py --bundle . \
        --state "Your name is Marta. You are a baker from Britain." \
        --player "Who are you?"

4) Verify the extracted freeze hashes:
    python deployment/uomind_infer.py --bundle . --verify-only

If --bundle is omitted, the script searches the current directory and its
immediate descendants for an extracted NPCMicro13M deployment bundle.

Dependencies:
    pip install torch tokenizers
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required.\n"
        "Install it first, for example:\n"
        "  python -m pip install torch\n"
        "For NVIDIA CUDA builds, use the installer command recommended at "
        "https://pytorch.org/get-started/locally/"
    ) from exc

try:
    from tokenizers import Tokenizer
except ImportError as exc:
    raise SystemExit(
        "The Hugging Face 'tokenizers' package is required.\n"
        "Install it with:\n"
        "  python -m pip install tokenizers"
    ) from exc

from uomind_grounded_response import grounded_response


EXPECTED_PARAMETER_COUNT = 13_640_064
REQUIRED_MARKERS = ("<PAD>", "<BOS>", "<EOS>", "<STATE>", "<PLAYER>", "<SAY>")


# ---------------------------------------------------------------------------
# Exact frozen NPCMicro13M architecture
# ---------------------------------------------------------------------------

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
            raise ValueError("one geometry scale is required per head")

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
            raise ValueError(
                f"NPCMicro13M expects a static sequence length of "
                f"{self.max_seq_len}, received {t}"
            )

        qkv = self.qkv_proj(x).view(
            b, t, 3, self.n_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        gain = self.gains().to(q.dtype).view(1, self.n_heads, 1, 1)
        bias = gain * self.base_geometry.to(q.dtype) + self.causal_bias.to(q.dtype)

        if self.backend == "sdpa":
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
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
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GeometryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    def num_parameters(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    def geometry_gain_summary(self) -> dict:
        g = torch.stack([b.attn.gains() for b in self.blocks])
        return {
            "mean": float(g.mean().detach().cpu()),
            "min": float(g.min().detach().cpu()),
            "max": float(g.max().detach().cpu()),
        }


# ---------------------------------------------------------------------------
# Bundle discovery / integrity verification
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def looks_like_bundle(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "model").is_dir()
        and (path / "tokenizer").is_dir()
        and (path / "tokenizer" / "tokenizer.json").is_file()
    )


def discover_bundle(start: Path) -> Path:
    start = start.expanduser().resolve()

    if start.is_file():
        if start.suffix.lower() == ".zip":
            raise SystemExit(
                f"{start} is still a ZIP. Extract it first, then point --bundle "
                "at the extracted folder."
            )
        raise SystemExit(f"--bundle must be a directory, not: {start}")

    if looks_like_bundle(start):
        return start

    # Common case: user points at the parent directory produced by extraction.
    direct_children = [p for p in start.iterdir() if p.is_dir()] if start.is_dir() else []
    matches = [p for p in direct_children if looks_like_bundle(p)]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise SystemExit(
            "Multiple NPCMicro13M bundles found. Pass --bundle with the exact "
            "extracted freeze directory:\n  " +
            "\n  ".join(str(p) for p in matches)
        )

    # One deeper level is enough for normal ZIP extraction layouts.
    deeper = []
    for parent in direct_children:
        try:
            deeper.extend(p for p in parent.iterdir() if p.is_dir() and looks_like_bundle(p))
        except OSError:
            pass
    if len(deeper) == 1:
        return deeper[0].resolve()

    raise SystemExit(
        f"Could not find an extracted NPCMicro13M deployment under:\n  {start}\n\n"
        "Expected a directory containing:\n"
        "  model/npcmicro13m_sft.pt\n"
        "  tokenizer/tokenizer.json\n"
        "  manifest.json (normally present)\n"
    )


def discover_checkpoint(bundle: Path, override: Optional[str]) -> Path:
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

    preferred = [
        bundle / "model" / "npcmicro13m_sft.pt",
        bundle / "model" / "v9_base_finetune.pt",
        bundle / "model" / "v9_grounded.pt",
        bundle / "model" / "best_balanced.pt",
        bundle / "model" / "best_phase3.pt",
        bundle / "model" / "final.pt",
    ]
    for p in preferred:
        if p.is_file():
            return p.resolve()

    pts = sorted((bundle / "model").glob("*.pt"))
    if len(pts) == 1:
        return pts[0].resolve()

    raise SystemExit(
        f"Could not uniquely identify a checkpoint under {bundle / 'model'}"
    )


def discover_tokenizer(bundle: Path, override: Optional[str]) -> Path:
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"Tokenizer not found: {p}")
        return p
    p = bundle / "tokenizer" / "tokenizer.json"
    if not p.is_file():
        raise SystemExit(f"Tokenizer not found: {p}")
    return p.resolve()


def verify_freeze(bundle: Path) -> bool:
    sums = bundle / "SHA256SUMS.txt"
    if not sums.is_file():
        print("SHA256SUMS.txt is not present; skipping full freeze verification.")
        return True

    rows = []
    for raw in sums.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            digest, rel = raw.split(None, 1)
        except ValueError:
            print(f"Malformed SHA256SUMS line: {raw!r}")
            return False
        rows.append((digest.lower(), rel.strip()))

    print(f"Verifying {len(rows)} frozen artifacts...")
    ok = True
    for i, (expected, rel) in enumerate(rows, 1):
        p = bundle / rel
        if not p.is_file():
            print(f"  [{i:02d}/{len(rows):02d}] MISSING  {rel}")
            ok = False
            continue
        actual = sha256_file(p)
        match = actual.lower() == expected
        print(f"  [{i:02d}/{len(rows):02d}] {'OK' if match else 'FAIL':7s}  {rel}")
        if not match:
            print(f"      expected {expected}")
            print(f"      actual   {actual}")
            ok = False

    print("Freeze integrity:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------------------
# Tokenization / model loading
# ---------------------------------------------------------------------------

def validate_tokenizer(tok: Tokenizer) -> None:
    missing = [m for m in REQUIRED_MARKERS if tok.token_to_id(m) is None]
    if missing:
        raise RuntimeError(f"Tokenizer is missing required markers: {missing}")

    non_atomic = [
        m for m in REQUIRED_MARKERS
        if tok.encode(m).ids != [tok.token_to_id(m)]
    ]
    if non_atomic:
        raise RuntimeError(
            f"Control markers are not atomic tokenizer tokens: {non_atomic}"
        )


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        dev = torch.device(requested)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")
        if dev.type == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise SystemExit("MPS was requested but is not available.")
        return dev

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_precision(device: torch.device, requested: str) -> torch.dtype:
    if requested == "fp32":
        return torch.float32
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16

    # auto
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type in {"cuda", "mps"} and dtype in {torch.float16, torch.bfloat16}:
        try:
            return torch.autocast(device_type=device.type, dtype=dtype)
        except Exception:
            return contextlib.nullcontext()
    return contextlib.nullcontext()


def load_model(
    checkpoint: Path,
    tok: Tokenizer,
    device: torch.device,
    precision: torch.dtype,
    attention_backend: Optional[str],
):
    print(f"Loading checkpoint: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions that do not know weights_only.
        payload = torch.load(checkpoint, map_location="cpu")

    if "model_config" not in payload or "model_state_dict" not in payload:
        raise RuntimeError(
            "Checkpoint does not contain model_config/model_state_dict in the "
            "expected NPCMicro13M checkpoint format."
        )

    cfg_dict = dict(payload["model_config"])
    if attention_backend:
        cfg_dict["attention_backend"] = attention_backend
    cfg = ModelConfig(**cfg_dict)

    if cfg.vocab_size != tok.get_vocab_size():
        raise RuntimeError(
            f"Checkpoint vocab={cfg.vocab_size}, tokenizer vocab={tok.get_vocab_size()}"
        )

    model = UOMindLM(cfg)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    # lm_head is tied to token_embedding in UOMindLM, so the deployment
    # checkpoint stores that tensor only once.
    missing = [
        k for k in incompatible.missing_keys
        if "causal_bias" not in k and k != "lm_head.weight"
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch. missing={missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    n_params = model.num_parameters()
    if n_params != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Unexpected parameter count: {n_params:,}; "
            f"expected {EXPECTED_PARAMETER_COUNT:,}"
        )

    model.to(device)
    model.eval()

    geom = model.geometry_gain_summary()
    print(f"parameters:          {n_params:,}")
    print(f"context:             {cfg.max_seq_len}")
    print(f"device:              {device}")
    print(f"precision:           {str(precision).replace('torch.', '')}")
    print(f"attention backend:   {cfg.attention_backend}")
    print(
        "geometry gain:       "
        f"mean={geom['mean']:.4f} min={geom['min']:.4f} max={geom['max']:.4f}"
    )

    return model, payload, cfg


# ---------------------------------------------------------------------------
# Exact runtime serialization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split()).strip()


def encode_content(tok: Tokenizer, text: str) -> list[int]:
    # Exact training convention: natural text gets one leading ASCII space.
    return tok.encode(" " + normalize_text(text)).ids


def build_runtime_prompt(
    tok: Tokenizer,
    state: str,
    player: str,
    max_seq_len: int = 128,
    max_system_tokens: int = 40,
) -> tuple[list[int], bool, bool]:
    bos = tok.token_to_id("<BOS>")
    state_id = tok.token_to_id("<STATE>")
    player_id = tok.token_to_id("<PLAYER>")
    say = tok.token_to_id("<SAY>")

    state_tokens_full = [state_id] + encode_content(tok, state)
    state_tokens = state_tokens_full[:max_system_tokens]
    state_trimmed = len(state_tokens) < len(state_tokens_full)

    player_tokens = [player_id] + encode_content(tok, player)

    # Reserve BOS and SAY. Preserve STATE first, then the tail of the player's
    # current utterance exactly as the training prompt builder did.
    remaining = max_seq_len - 2
    keep_state = state_tokens[:remaining]
    remaining -= len(keep_state)

    player_trimmed = len(player_tokens) > remaining
    keep_player = player_tokens[-remaining:] if remaining > 0 else []

    prompt = [bos] + keep_state + keep_player + [say]
    if len(prompt) > max_seq_len:
        raise RuntimeError("Internal prompt packing overflow")
    return prompt, state_trimmed, player_trimmed


def top_p_sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits.float() / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        # Always keep at least the most probable token.
        remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        denom = sorted_probs.sum()
        if float(denom) > 0:
            sorted_probs = sorted_probs / denom
        choice = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_idx[choice].item())

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.inference_mode()
def generate(
    model: UOMindLM,
    tok: Tokenizer,
    state: str,
    player: str,
    device: torch.device,
    precision: torch.dtype,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_system_tokens: int = 40,
):
    prompt, state_trimmed, player_trimmed = build_runtime_prompt(
        tok,
        state,
        player,
        max_seq_len=model.config.max_seq_len,
        max_system_tokens=max_system_tokens,
    )

    pad = tok.token_to_id("<PAD>")
    eos = tok.token_to_id("<EOS>")
    context = list(prompt)
    out: list[int] = []
    stop = "max_new_tokens"

    start = time.perf_counter()

    for _ in range(max_new_tokens):
        if len(context) > model.config.max_seq_len:
            stop = "context_limit"
            break

        x = torch.full(
            (1, model.config.max_seq_len),
            pad,
            dtype=torch.long,
            device=device,
        )
        x[0, :len(context)] = torch.tensor(
            context, dtype=torch.long, device=device
        )

        with autocast_context(device, precision):
            logits = model(x)

        next_logits = logits[0, len(context) - 1]
        nxt = top_p_sample(next_logits, temperature=temperature, top_p=top_p)

        if nxt == eos:
            stop = "EOS"
            break

        out.append(nxt)

        if len(context) == model.config.max_seq_len:
            stop = "context_limit_after_token"
            break

        context.append(nxt)

    elapsed = time.perf_counter() - start
    text = tok.decode(out, skip_special_tokens=True).strip()
    tps = len(out) / elapsed if elapsed > 0 else 0.0

    return {
        "text": text,
        "tokens": len(out),
        "stop": stop,
        "seconds": elapsed,
        "tokens_per_second": tps,
        "prompt_tokens": len(prompt),
        "state_trimmed": state_trimmed,
        "player_trimmed": player_trimmed,
    }


# ---------------------------------------------------------------------------
# Built-in local sanity/generalization suite
# ---------------------------------------------------------------------------

TESTS = [
    {
        "name": "identity",
        "state": "Your name is Elara. You are a weaver from Yew.",
        "player": "Tell me plainly, who art thou?",
        "required_all": ["elara", "weaver"],
        "forbidden": [],
    },
    {
        "name": "hometown",
        "state": "Your name is Corin. You are a mason from Trinsic.",
        "player": "What town dost thou call home?",
        "required_any": ["trinsic"],
        "forbidden": [],
    },
    {
        "name": "false_identity",
        "state": "Your name is Nessa. You are a baker from Britain.",
        "player": "Art thou Mara the fisherwoman from Cove?",
        "required_any": ["nessa", "baker", "britain"],
        "forbidden": ["i am mara", "i'm mara"],
    },
    {
        "name": "innkeeper_food",
        "state": "Your name is Talia. You keep a small inn in Britain.",
        "player": "I've but a few coppers. What cheap fare could fill me?",
        "required_any": ["bread", "porridge", "stew", "soup", "meal", "food", "bowl"],
        "forbidden": ["not my trade", "ask a gardener", "ask a farmer"],
    },
    {
        "name": "farmer_weather",
        "state": "Your name is Alden. You are an experienced farmer near Vesper.",
        "player": "Rain is coming fast. Should the cut hay remain in the field?",
        "required_any": ["cover", "barn", "inside", "bring", "move", "rain", "hay"],
        "forbidden": ["ask a farmer", "not my trade"],
    },
    {
        "name": "state_fact",
        "state": "Your name is Pip. You are a cooper from Moonglow. Bread costs seven coppers today.",
        "player": "How many coppers for a loaf today?",
        "required_all": ["seven", "copper"],
        "forbidden": [],
    },
    {
        "name": "unknown_road",
        "state": "Your name is Polly. You are a shopkeeper from Minoc. You have never travelled far and do not know distant roads.",
        "player": "How do I reach Serpent Isle?",
        "required_any": ["know not", "i know not", "never", "cannot", "can't", "do not know"],
        "forbidden": [],
    },
    {
        "name": "known_road_contrast",
        "state": "Your name is Polly. You are a shopkeeper from Minoc. Road to Serpent Isle: take the east road to the ferry at dawn.",
        "player": "How do I reach Serpent Isle?",
        "required_any": ["east road", "ferry", "dawn"],
        "forbidden": ["know not", "do not know", "never travelled"],
    },
    {
        "name": "general_feeling",
        "state": "Your name is Brina. You are a brewer from Cove. You are cheerful today.",
        "player": "How fare thee this morning?",
        "required_any": ["well", "good", "fine", "cheer", "kindly"],
        "forbidden": ["not my trade"],
    },
    {
        "name": "dawn_sounds",
        "state": "Your name is Toren. You are a baker from Moonglow. You watch the town quietly.",
        "player": "What sounds greet thee when morning breaks?",
        "required_any": ["bell", "bird", "cart", "rooster", "cock", "hammer", "shutter"],
        "forbidden": ["not my trade"],
    },
    {
        "name": "blacksmith_paraphrase",
        "state": "Your name is Garrick. You earn thy living at the forge in Minoc.",
        "player": "My blade caught a stone and took a nick near the point. Is it worth mending?",
        "required_any": ["blade", "nick", "ground", "smooth", "steel", "mend", "repair"],
        "forbidden": ["ask a blacksmith", "not my trade"],
    },
    {
        "name": "false_hometown",
        "state": "Your name is Mira. You are a tanner from Jhelom.",
        "player": "I could have sworn thou wert from Yew.",
        "required_any": ["jhelom"],
        "forbidden": ["i am from yew", "from yew"],
    },
]


def norm_for_check(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9']+", " ", text.casefold()).split())


def score_test(test: dict, text: str) -> tuple[bool, list[str]]:
    normalized = norm_for_check(text)
    required_any = [norm_for_check(x) for x in test.get("required_any", [])]
    required_all = [norm_for_check(x) for x in test.get("required_all", [])]
    forbidden = [norm_for_check(x) for x in test.get("forbidden", [])]

    reasons = []
    ok = True

    if required_any and not any(x in normalized for x in required_any):
        ok = False
        reasons.append("missing any of: " + ", ".join(test["required_any"]))

    missing_all = [
        original for original, normalized_term
        in zip(test.get("required_all", []), required_all)
        if normalized_term not in normalized
    ]
    if missing_all:
        ok = False
        reasons.append("missing: " + ", ".join(missing_all))

    hits = [
        original for original, normalized_term
        in zip(test.get("forbidden", []), forbidden)
        if normalized_term in normalized
    ]
    if hits:
        ok = False
        reasons.append("forbidden: " + ", ".join(hits))

    return ok, reasons


def run_test_suite(
    model,
    tok,
    device,
    precision,
    args,
) -> int:
    print()
    print("=" * 78)
    print("NPCMICRO13M LOCAL SANITY / GENERALIZATION SUITE")
    print("=" * 78)
    print(
        "These prompts are a local behavioral smoke test. "
        "They are not a substitute for the full Colab V4 benchmark."
    )
    print()

    passed = 0
    total_tokens = 0
    total_seconds = 0.0

    for i, test in enumerate(TESTS, 1):
        result = generate(
            model,
            tok,
            state=test["state"],
            player=test["player"],
            device=device,
            precision=precision,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_system_tokens=args.max_system_tokens,
        )
        ok, reasons = score_test(test, result["text"])
        passed += int(ok)
        total_tokens += result["tokens"]
        total_seconds += result["seconds"]

        print(f"T{i:02d} [{test['name']}] {'PASS' if ok else 'MISS'}")
        print(f"  STATE:  {test['state']}")
        print(f"  PLAYER: {test['player']}")
        print(f"  MODEL:  {result['text']}")
        if reasons:
            print(f"  CHECK:  {'; '.join(reasons)}")
        print(
            f"  stop={result['stop']} generated={result['tokens']} "
            f"latency={result['seconds']*1000:.1f}ms "
            f"tok/s={result['tokens_per_second']:.1f}"
        )
        if result["state_trimmed"] or result["player_trimmed"]:
            print(
                f"  WARNING: prompt trimmed "
                f"(state={result['state_trimmed']}, player={result['player_trimmed']})"
            )
        print()

    rate = passed / len(TESTS)
    overall_tps = total_tokens / total_seconds if total_seconds > 0 else 0.0

    print("-" * 78)
    print(f"score:              {passed}/{len(TESTS)} = {rate*100:.1f}%")
    print(f"generated tokens:   {total_tokens}")
    print(f"aggregate tok/s:    {overall_tps:.1f}")
    print("-" * 78)

    return 0 if passed == len(TESTS) else 2


def interactive_loop(model, tok, device, precision, args) -> None:
    print()
    print("=" * 78)
    print("NPCMICRO13M INTERACTIVE LOCAL INFERENCE")
    print("=" * 78)
    print("Runtime is stateless/single-turn: each PLAYER line is answered from STATE.")
    print("Commands: /state  /show  /quit")
    print()

    state = normalize_text(args.state or "")
    if not state:
        state = input("NPC STATE> ").strip()

    while True:
        try:
            player = input("PLAYER> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not player:
            continue

        if player.casefold() in {"/quit", "/exit", "quit", "exit"}:
            break

        if player.casefold() == "/state":
            state = input("NEW STATE> ").strip()
            continue

        if player.casefold() == "/show":
            print("STATE:", state)
            continue

        result = generate(
            model,
            tok,
            state=state,
            player=player,
            device=device,
            precision=precision,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_system_tokens=args.max_system_tokens,
        )
        raw_text = result["text"]
        text = raw_text if args.raw_model else grounded_response(
            state, player, lambda _state, _player: raw_text
        )
        print("NPC>", text)
        if not args.raw_model and text != raw_text:
            print("     [grounded response layer applied]")
        print(
            f"     [{result['tokens']} tok, {result['seconds']*1000:.1f} ms, "
            f"{result['tokens_per_second']:.1f} tok/s, stop={result['stop']}]"
        )
        if result["state_trimmed"] or result["player_trimmed"]:
            print(
                "     [prompt trimming: "
                f"state={result['state_trimmed']} player={result['player_trimmed']}]"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the frozen NPCMicro13M model locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--bundle",
        default=".",
        help="NPCMicro13M deployment directory, or its parent directory.",
    )
    p.add_argument("--checkpoint", default=None, help="Override checkpoint path.")
    p.add_argument("--tokenizer", default=None, help="Override tokenizer.json path.")
    p.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, mps, etc.",
    )
    p.add_argument(
        "--precision",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
    )
    p.add_argument(
        "--attention-backend",
        choices=["sdpa", "manual"],
        default=None,
        help="Override checkpoint attention backend. Try manual if SDPA is unsupported.",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-system-tokens", type=int, default=40)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy, matching the main evaluation setup.",
    )
    p.add_argument("--top-p", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=5042)
    p.add_argument(
        "--raw-model",
        action="store_true",
        help="Bypass the grounded response layer and show the neural model output.",
    )

    action = p.add_mutually_exclusive_group()
    action.add_argument("--test", action="store_true", help="Run built-in local test suite.")
    action.add_argument("--interactive", action="store_true", help="Interactive NPC loop.")
    action.add_argument("--verify-only", action="store_true", help="Verify freeze hashes and exit.")

    p.add_argument("--verify", action="store_true", help="Verify freeze hashes before inference.")
    p.add_argument("--state", default=None, help="NPC STATE for one-shot/interactive mode.")
    p.add_argument("--player", default=None, help="PLAYER utterance for one-shot mode.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be >= 1")
    if args.max_system_tokens < 1:
        raise SystemExit("--max-system-tokens must be >= 1")
    if args.temperature < 0:
        raise SystemExit("--temperature must be >= 0")
    if not (0 < args.top_p <= 1.0):
        raise SystemExit("--top-p must be in (0, 1]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    bundle = discover_bundle(Path(args.bundle))
    print("=" * 78)
    print("NPCMICRO13M LOCAL INFERENCE")
    print("=" * 78)
    print(f"bundle:              {bundle}")

    if args.verify or args.verify_only:
        if not verify_freeze(bundle):
            return 3
        if args.verify_only:
            return 0

    checkpoint = discover_checkpoint(bundle, args.checkpoint)
    tokenizer_path = discover_tokenizer(bundle, args.tokenizer)
    print(f"checkpoint:          {checkpoint}")
    print(f"tokenizer:           {tokenizer_path}")

    tok = Tokenizer.from_file(str(tokenizer_path))
    validate_tokenizer(tok)
    print(f"tokenizer vocab:     {tok.get_vocab_size()}")

    device = choose_device(args.device)
    precision = choose_precision(device, args.precision)

    try:
        model, payload, cfg = load_model(
            checkpoint,
            tok,
            device,
            precision,
            args.attention_backend,
        )
    except RuntimeError as exc:
        # Helpful MPS/older torch guidance.
        if args.attention_backend is None:
            print()
            print("Model load failed:", exc)
            print(
                "If your local PyTorch build has trouble with scaled-dot-product "
                "attention, retry with:\n"
                "  --attention-backend manual"
            )
        raise

    print(
        "runtime contract:    "
        "<BOS><STATE> NPC facts/persona <PLAYER> current speech <SAY> response <EOS>"
    )

    # One-shot takes precedence when both state and player are supplied.
    if args.state is not None and args.player is not None:
        result = generate(
            model,
            tok,
            state=args.state,
            player=args.player,
            device=device,
            precision=precision,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_system_tokens=args.max_system_tokens,
        )
        raw_text = result["text"]
        text = raw_text if args.raw_model else grounded_response(
            args.state, args.player, lambda _state, _player: raw_text
        )
        print()
        print("STATE: ", normalize_text(args.state))
        print("PLAYER:", normalize_text(args.player))
        print("NPC:   ", text)
        if not args.raw_model and text != raw_text:
            print("mode:   grounded response layer applied")
        print(
            f"generated={result['tokens']} stop={result['stop']} "
            f"latency={result['seconds']*1000:.1f}ms "
            f"tok/s={result['tokens_per_second']:.1f}"
        )
        return 0

    if args.interactive:
        interactive_loop(model, tok, device, precision, args)
        return 0

    # Default action is the built-in test suite.
    return run_test_suite(model, tok, device, precision, args)


if __name__ == "__main__":
    raise SystemExit(main())
