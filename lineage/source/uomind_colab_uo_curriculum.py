#!/usr/bin/env python3
"""
uomind_colab_uo_curriculum.py

Continuation pretraining for the 128-token UO-Mind model.

This script is designed to run AFTER the TinyStories pretraining module.
It loads the existing TinyStories-trained checkpoint/tokenizer, discovers
all Ultima Online story files in /content/uo_stories, packs them with the
EXISTING tokenizer, and runs a five-stage domain curriculum:

    Stage 1: 20% UO / 80% TinyStories
    Stage 2: 40% UO / 60% TinyStories
    Stage 3: 60% UO / 40% TinyStories
    Stage 4: 80% UO / 20% TinyStories
    Stage 5: 100% UO

Each stage consumes every UO TRAINING block exactly once. Therefore the UO
training corpus receives exactly five passes by default.

The ratio is measured in 128-token training sequences. Because every real
sequence has the same length, sequence ratio == token ratio.

Default Colab layout
--------------------
TinyStories workspace:
    /content/uomind_tinystories/
        tokenizer.json
        train.bin
        validation.bin
        corpus_meta.json
        checkpoints/
            best.pt
            final.pt
            latest.pt

UO input:
    /content/uo_stories/
        *.txt

Output:
    /content/uomind_tinystories/uo_curriculum/
        uo_train.bin
        uo_validation.bin
        uo_corpus_meta.json
        curriculum_history.json
        checkpoints/
            stage_1_20pct_uo.pt
            ...
            stage_5_100pct_uo.pt
            latest.pt
            best_uo.pt
            final.pt

Fast-path design
----------------
- Never retrains or changes the tokenizer.
- TinyStories Parquet/network/tokenization are absent from the hot path.
- UO text is tokenized exactly once.
- Both domains are memory-mapped as uint16.
- Each training batch is mixed at row level.
- CPU batch assembly is background-prefetched into pinned memory.
- GPU tensor addresses are static.
- FP16 AMP + fused AdamW are used on CUDA.
- No per-step .item() / CUDA synchronization.
- Validation occurs only before curriculum start and after each stage.
- Mid-stage checkpoints are resumable and reconstruct the exact stage schedule.

Important
---------
This script starts a FRESH optimizer/schedule from the pretrained model weights.
That is intentional for domain-adaptive continuation training. It does NOT
continue the fully-decayed TinyStories optimizer schedule.

Typical run
-----------
    !python uomind_colab_uo_curriculum.py

If your TinyStories checkpoint is in Drive:
    !python uomind_colab_uo_curriculum.py \
        --checkpoint /content/drive/MyDrive/uomind_tinystories/best.pt \
        --tokenizer /content/drive/MyDrive/uomind_tinystories/tokenizer.json \
        --backup-dir /content/drive/MyDrive/uomind_uo_curriculum

To train every UO story instead of reserving validation:
    !python uomind_colab_uo_curriculum.py --uo-val-fraction 0
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def ensure_packages() -> None:
    required = {
        "huggingface_hub": "huggingface_hub>=0.34",
        "tokenizers": "tokenizers>=0.21",
        "pyarrow": "pyarrow>=15",
    }

    missing = [
        requirement
        for module_name, requirement in required.items()
        if importlib.util.find_spec(module_name) is None
    ]

    if not missing:
        return

    print("Installing missing dependencies:", " ".join(missing), flush=True)

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            *missing,
        ]
    )


ensure_packages()

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_WORKSPACE = Path("/content/uomind_tinystories")
DEFAULT_UO_DIR = Path("/content/uo_stories")
DATASET_REPO = "roneneldan/TinyStories"

DEFAULT_UO_RATIOS = (
    0.20,
    0.40,
    0.60,
    0.80,
    1.00,
)


# ---------------------------------------------------------------------------
# Model definition
#
# This intentionally mirrors the previous Colab pretraining module so its
# checkpoint state_dict loads without source-code dependencies.
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
    geometry_scales: Tuple[float, ...] = (
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
        256.0,
    )
    geometry_gain_init: float = 1.0
    geometry_gain_max: float = 4.0
    init_std: float = 0.02
    attention_backend: str = "sdpa"

    def __post_init__(self) -> None:
        if isinstance(self.geometry_scales, list):
            self.geometry_scales = tuple(self.geometry_scales)

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if len(self.geometry_scales) != self.n_heads:
            raise ValueError("one fixed geometry scale is required per head")

        if self.max_seq_len != 128:
            raise ValueError(
                "this continuation module is intentionally fixed at context 128"
            )

        if self.vocab_size > 65535:
            raise ValueError("vocab must fit uint16 packed storage")

        if self.attention_backend not in {"sdpa", "manual"}:
            raise ValueError("attention_backend must be sdpa or manual")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        normed = xf * torch.rsqrt(
            xf.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normed.to(dtype) * self.weight


class GeometryAttention(nn.Module):
    """
    NoPE attention with fixed physical-length-scale relational geometry.

        score = QK^T / sqrt(dh) + g_h * (-|i-j| / tau_h)

    Token state receives no additive positional vector.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.backend = config.attention_backend
        self.gain_max = float(config.geometry_gain_max)

        self.qkv_proj = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=False,
        )

        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )

        scales = torch.tensor(
            config.geometry_scales,
            dtype=torch.float32,
        )

        positions = torch.arange(
            config.max_seq_len,
            dtype=torch.float32,
        )

        distance = torch.abs(
            positions[:, None] - positions[None, :]
        )

        base_geometry = (
            -distance.unsqueeze(0) / scales[:, None, None]
        ).to(torch.float16)

        # [1, H, T, T]
        self.register_buffer(
            "base_geometry",
            base_geometry.unsqueeze(0),
            persistent=True,
        )

        causal = torch.zeros(
            config.max_seq_len,
            config.max_seq_len,
            dtype=torch.float16,
        )

        causal = causal.masked_fill(
            torch.triu(
                torch.ones_like(causal, dtype=torch.bool),
                diagonal=1,
            ),
            float("-inf"),
        )

        # [1, 1, T, T]
        self.register_buffer(
            "causal_bias",
            causal.unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        init_ratio = (
            config.geometry_gain_init
            / config.geometry_gain_max
        )

        raw_init = math.log(
            init_ratio / (1.0 - init_ratio)
        )

        self.raw_gain = nn.Parameter(
            torch.full(
                (config.n_heads,),
                raw_init,
                dtype=torch.float32,
            )
        )

    def gains(self) -> torch.Tensor:
        return self.gain_max * torch.sigmoid(self.raw_gain)

    def _geometry_bias(
        self,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        gains = self.gains().to(dtype=dtype).view(
            1,
            self.n_heads,
            1,
            1,
        )

        geometry = gains * self.base_geometry.to(dtype=dtype)

        return geometry + self.causal_bias.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        if seq_len != self.max_seq_len:
            raise ValueError(
                f"static graph expects seq_len={self.max_seq_len}, got {seq_len}"
            )

        qkv = self.qkv_proj(x)

        qkv = qkv.view(
            batch,
            seq_len,
            3,
            self.n_heads,
            self.head_dim,
        )

        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        bias = self._geometry_bias(q.dtype)

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
            scores = torch.matmul(
                q,
                k.transpose(-2, -1),
            )

            scores = scores * (
                1.0 / math.sqrt(self.head_dim)
            )

            scores = scores + bias
            weights = torch.softmax(scores, dim=-1)
            y = torch.matmul(weights, v)

        y = y.transpose(1, 2).contiguous().view(
            batch,
            seq_len,
            self.d_model,
        )

        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.fc1 = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=False,
        )

        self.fc2 = nn.Linear(
            config.d_ff,
            config.d_model,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(
            F.gelu(
                self.fc1(x),
                approximate="tanh",
            )
        )


class UOMindBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.attn_norm = RMSNorm(
            config.d_model,
            config.norm_eps,
        )

        self.attn = GeometryAttention(config)

        self.mlp_norm = RMSNorm(
            config.d_model,
            config.norm_eps,
        )

        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x)
        )

        x = x + self.mlp(
            self.mlp_norm(x)
        )

        return x


class UOMindLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.blocks = nn.ModuleList(
            [
                UOMindBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.final_norm = RMSNorm(
            config.d_model,
            config.norm_eps,
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self.apply(self._init_weights)

        # Exact weight tying used in the original pretraining module.
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.init_std,
            )
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.init_std,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        # -100 is used for padded rows of the final static batch.
        loss = F.cross_entropy(
            logits.reshape(
                -1,
                self.config.vocab_size,
            ),
            targets.reshape(-1),
            ignore_index=-100,
        )

        return logits, loss

    def num_parameters(self) -> int:
        seen = set()
        total = 0

        for parameter in self.parameters():
            pid = id(parameter)

            if pid in seen:
                continue

            seen.add(pid)
            total += parameter.numel()

        return total

    def geometry_gain_summary(self) -> Dict[str, object]:
        gains = torch.stack(
            [
                block.attn.gains()
                for block in self.blocks
            ],
            dim=0,
        )

        return {
            "mean": float(
                gains.mean().detach().cpu().item()
            ),
            "min": float(
                gains.min().detach().cpu().item()
            ),
            "max": float(
                gains.max().detach().cpu().item()
            ),
            "matrix": gains.detach().cpu().tolist(),
        }

    def parameter_groups(
        self,
        weight_decay: float,
    ) -> List[Dict[str, object]]:
        decay: List[torch.Tensor] = []
        no_decay: List[torch.Tensor] = []
        seen = set()

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue

            pid = id(parameter)

            if pid in seen:
                continue

            seen.add(pid)

            if (
                parameter.ndim < 2
                or "norm" in name.lower()
                or "raw_gain" in name
            ):
                no_decay.append(parameter)
            else:
                decay.append(parameter)

        return [
            {
                "params": decay,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ]


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_context():
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    )


def make_grad_scaler():
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
    except Exception:
        return torch.cuda.amp.GradScaler(
            enabled=True,
        )


def gpu_memory_line() -> str:
    free, total = torch.cuda.mem_get_info()

    return (
        f"{free / 2**30:.2f} GiB free / "
        f"{total / 2**30:.2f} GiB total"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def parse_ratios(text: str) -> Tuple[float, ...]:
    values = tuple(
        float(part.strip())
        for part in text.split(",")
        if part.strip()
    )

    if not values:
        raise ValueError("no UO ratios supplied")

    for value in values:
        if not (0.0 < value <= 1.0):
            raise ValueError(
                f"UO ratios must be in (0,1], got {value}"
            )

    return values


# ---------------------------------------------------------------------------
# Checkpoint / tokenizer discovery
# ---------------------------------------------------------------------------


def find_checkpoint(
    requested: str,
    workspace: Path,
) -> Path:
    if requested != "auto":
        path = Path(requested)

        if not path.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {path}"
            )

        return path

    candidates = [
        workspace / "checkpoints" / "best.pt",
        workspace / "checkpoints" / "final.pt",
        workspace / "checkpoints" / "latest.pt",
        Path("/content/drive/MyDrive/uomind_tinystories/best.pt"),
        Path("/content/drive/MyDrive/uomind_tinystories/final.pt"),
        Path("/content/drive/MyDrive/uomind_tinystories/latest.pt"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the TinyStories-trained checkpoint automatically. "
        "Pass --checkpoint /path/to/best.pt"
    )


def find_tokenizer(
    requested: str,
    workspace: Path,
    checkpoint_path: Path,
) -> Path:
    candidates: List[Path] = []

    if requested != "auto":
        candidates.append(Path(requested))

    candidates.extend(
        [
            workspace / "tokenizer.json",
            checkpoint_path.parent / "tokenizer.json",
            checkpoint_path.parent.parent / "tokenizer.json",
            Path("/content/drive/MyDrive/uomind_tinystories/tokenizer.json"),
        ]
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find tokenizer.json. The exact pretrained tokenizer is "
        "required; it must not be retrained for continuation training."
    )


def load_pretrained_model(
    checkpoint_path: Path,
    tokenizer: Tokenizer,
    device: torch.device,
) -> Tuple[UOMindLM, Dict[str, object], ModelConfig]:
    print(
        f"Loading pretrained checkpoint: {checkpoint_path}",
        flush=True,
    )

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_config" not in payload:
        raise RuntimeError(
            "checkpoint does not contain model_config"
        )

    saved_config = dict(payload["model_config"])

    # The runtime attention backend can safely be overridden later if needed,
    # but default to the exact backend recorded in pretraining.
    model_config = ModelConfig(**saved_config)

    tokenizer_vocab = tokenizer.get_vocab_size()

    if tokenizer_vocab != model_config.vocab_size:
        raise RuntimeError(
            f"Tokenizer/model vocab mismatch: "
            f"tokenizer={tokenizer_vocab}, model={model_config.vocab_size}"
        )

    model = UOMindLM(model_config)

    state_dict = payload.get("model_state_dict")

    if state_dict is None:
        raise RuntimeError(
            "checkpoint does not contain model_state_dict"
        )

    incompatible = model.load_state_dict(
        state_dict,
        strict=False,
    )

    # causal_bias is non-persistent and therefore expected not to exist in the
    # checkpoint. Everything else should be exact.
    unexpected = list(incompatible.unexpected_keys)

    missing = [
        key
        for key in incompatible.missing_keys
        if "causal_bias" not in key
    ]

    if unexpected or missing:
        raise RuntimeError(
            "checkpoint/model mismatch.\n"
            f"missing={missing}\n"
            f"unexpected={unexpected}"
        )

    model = model.to(device)

    print(
        f"Loaded {model.num_parameters():,} parameters; "
        f"geometry gain mean={model.geometry_gain_summary()['mean']:.4f}",
        flush=True,
    )

    return model, payload, model_config


# ---------------------------------------------------------------------------
# UO story parsing / cleaning
# ---------------------------------------------------------------------------


_MOJIBAKE_REPLACEMENTS = {
    "â": "—",
    "â": "–",
    "â": "’",
    "â": "‘",
    "â": "“",
    "â": "”",
    "â¦": "…",
    "Â ": " ",
    "Â": "",
}


def repair_obvious_mojibake(text: str) -> str:
    for bad, good in _MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)

    return text


def read_text_robust(path: Path) -> str:
    raw = path.read_bytes()

    encodings = (
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    )

    last_error: Optional[Exception] = None

    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            return repair_obvious_mojibake(text)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"could not decode {path}: {last_error}",
    )


def split_story_file(path: Path) -> List[str]:
    """
    Primary format:
        one story paragraph, blank line, next story paragraph.

    If a file has no blank separators but contains many non-empty lines,
    each non-empty line is treated as a story.
    """

    text = read_text_robust(path)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not text:
        return []

    blocks = [
        block.strip()
        for block in re.split(
            r"\n\s*\n+",
            text,
        )
        if block.strip()
    ]

    if len(blocks) <= 1:
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if len(lines) > 1:
            blocks = lines

    stories: List[str] = []

    for block in blocks:
        # Preserve punctuation and wording, but unwrap formatting newlines.
        story = re.sub(
            r"\s+",
            " ",
            block,
        ).strip()

        if story:
            stories.append(story)

    return stories


def discover_uo_files(
    uo_dir: Path,
    required_files: int,
    allow_any_count: bool,
) -> List[Path]:
    if not uo_dir.exists():
        raise FileNotFoundError(
            f"UO story folder not found: {uo_dir}"
        )

    files = sorted(
        path
        for path in uo_dir.rglob("*.txt")
        if path.is_file()
    )

    if not files:
        raise RuntimeError(
            f"No .txt files found under {uo_dir}"
        )

    if (
        not allow_any_count
        and len(files) != required_files
    ):
        raise RuntimeError(
            f"Expected exactly {required_files} UO .txt files, "
            f"found {len(files)}:\n"
            + "\n".join(str(path) for path in files)
        )

    print(
        f"Discovered {len(files)} UO story files:",
        flush=True,
    )

    for path in files:
        print(f"  {path}", flush=True)

    return files


def split_uo_train_validation(
    files: Sequence[Path],
    val_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str], Dict[str, object]]:
    train_stories: List[str] = []
    val_stories: List[str] = []

    file_stats: Dict[str, object] = {}

    for file_index, path in enumerate(files):
        stories = split_story_file(path)

        if not stories:
            raise RuntimeError(
                f"No stories could be parsed from {path}"
            )

        indices = list(range(len(stories)))

        rng = random.Random(
            seed
            + 10007 * (file_index + 1)
        )

        rng.shuffle(indices)

        if val_fraction <= 0.0:
            val_count = 0
        else:
            val_count = max(
                1,
                int(round(
                    len(stories) * val_fraction
                )),
            )

            if val_count >= len(stories):
                val_count = max(
                    0,
                    len(stories) - 1,
                )

        val_indices = set(
            indices[:val_count]
        )

        file_train = [
            story
            for index, story in enumerate(stories)
            if index not in val_indices
        ]

        file_val = [
            story
            for index, story in enumerate(stories)
            if index in val_indices
        ]

        train_stories.extend(file_train)
        val_stories.extend(file_val)

        file_stats[path.name] = {
            "stories": len(stories),
            "train": len(file_train),
            "validation": len(file_val),
            "sha256": sha256_file(path),
        }

        print(
            f"  parsed {path.name}: "
            f"{len(stories):,} stories "
            f"({len(file_train):,} train / "
            f"{len(file_val):,} val)",
            flush=True,
        )

    return train_stories, val_stories, file_stats


# ---------------------------------------------------------------------------
# Token packing
# ---------------------------------------------------------------------------


def pack_story_list(
    stories: Sequence[str],
    tokenizer: Tokenizer,
    output_path: Path,
    batch_size: int,
    label: str,
) -> int:
    eos_id = tokenizer.token_to_id("<EOS>")

    if eos_id is None:
        raise RuntimeError(
            "the pretrained tokenizer has no <EOS> token"
        )

    temp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    if temp_path.exists():
        temp_path.unlink()

    token_count = 0
    started = time.perf_counter()

    with temp_path.open(
        "wb",
        buffering=8 * 1024 * 1024,
    ) as handle:
        for start in range(
            0,
            len(stories),
            batch_size,
        ):
            batch = stories[
                start:start + batch_size
            ]

            encodings = tokenizer.encode_batch(
                list(batch)
            )

            flattened: List[int] = []

            for encoding in encodings:
                flattened.extend(
                    encoding.ids
                )
                flattened.append(eos_id)

            array = np.asarray(
                flattened,
                dtype=np.uint16,
            )

            array.tofile(handle)
            token_count += int(array.size)

            if (
                start == 0
                or start // batch_size % 100 == 0
            ):
                elapsed = max(
                    1e-9,
                    time.perf_counter() - started,
                )

                print(
                    f"  pack {label}: "
                    f"stories={min(start + len(batch), len(stories)):,}/"
                    f"{len(stories):,} "
                    f"tokens={token_count:,} "
                    f"rate={token_count / elapsed:,.0f} tok/s",
                    flush=True,
                )

    temp_path.replace(output_path)

    print(
        f"Packed {label}: "
        f"{len(stories):,} stories, "
        f"{token_count:,} tokens, "
        f"{output_path.stat().st_size / 2**20:.1f} MiB",
        flush=True,
    )

    return token_count


def uo_source_fingerprint(
    files: Sequence[Path],
    tokenizer_path: Path,
    val_fraction: float,
) -> str:
    digest = hashlib.sha256()

    digest.update(
        f"val_fraction={val_fraction:.12f}".encode()
    )

    digest.update(
        sha256_file(tokenizer_path).encode()
    )

    for path in files:
        digest.update(
            path.name.encode(
                "utf-8",
                errors="replace",
            )
        )
        digest.update(
            sha256_file(path).encode()
        )

    return digest.hexdigest()


def prepare_uo_corpus(
    output_dir: Path,
    files: Sequence[Path],
    tokenizer: Tokenizer,
    tokenizer_path: Path,
    val_fraction: float,
    seed: int,
    pack_batch_size: int,
) -> Dict[str, object]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_bin = output_dir / "uo_train.bin"
    val_bin = output_dir / "uo_validation.bin"
    meta_path = output_dir / "uo_corpus_meta.json"

    fingerprint = uo_source_fingerprint(
        files,
        tokenizer_path,
        val_fraction,
    )

    if meta_path.exists():
        try:
            meta = json.loads(
                meta_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                meta.get("fingerprint") == fingerprint
                and train_bin.exists()
                and int(meta.get("train_tokens", 0)) * 2
                == train_bin.stat().st_size
            ):
                validation_tokens = int(
                    meta.get(
                        "validation_tokens",
                        0,
                    )
                )

                val_ok = (
                    validation_tokens == 0
                    or (
                        val_bin.exists()
                        and validation_tokens * 2
                        == val_bin.stat().st_size
                    )
                )

                if val_ok:
                    print(
                        "UO packed corpus cache is valid; "
                        "skipping UO tokenization.",
                        flush=True,
                    )
                    return meta

        except Exception:
            pass

    print(
        "Parsing and packing UO story files...",
        flush=True,
    )

    (
        train_stories,
        val_stories,
        file_stats,
    ) = split_uo_train_validation(
        files=files,
        val_fraction=val_fraction,
        seed=seed,
    )

    train_tokens = pack_story_list(
        stories=train_stories,
        tokenizer=tokenizer,
        output_path=train_bin,
        batch_size=pack_batch_size,
        label="UO train",
    )

    validation_tokens = 0

    if val_stories:
        validation_tokens = pack_story_list(
            stories=val_stories,
            tokenizer=tokenizer,
            output_path=val_bin,
            batch_size=pack_batch_size,
            label="UO validation",
        )
    else:
        if val_bin.exists():
            val_bin.unlink()

    meta = {
        "fingerprint": fingerprint,
        "files": file_stats,
        "total_train_stories": len(train_stories),
        "total_validation_stories": len(val_stories),
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "train_bin": str(train_bin),
        "validation_bin": (
            str(val_bin)
            if validation_tokens > 0
            else None
        ),
        "uo_val_fraction": val_fraction,
        "tokenizer_sha256": sha256_file(
            tokenizer_path
        ),
    }

    meta_path.write_text(
        json.dumps(
            meta,
            indent=2,
        ),
        encoding="utf-8",
    )

    return meta


# ---------------------------------------------------------------------------
# TinyStories fallback preparation
# ---------------------------------------------------------------------------


def parquet_text_batches(
    files: Sequence[Path],
    batch_rows: int = 4096,
) -> Iterator[List[str]]:
    for path in files:
        parquet_file = pq.ParquetFile(path)

        for batch in parquet_file.iter_batches(
            batch_size=batch_rows,
            columns=["text"],
            use_threads=True,
        ):
            texts = [
                text
                for text in batch.column(0).to_pylist()
                if text
            ]

            if texts:
                yield texts


def pack_parquet_with_existing_tokenizer(
    files: Sequence[Path],
    tokenizer: Tokenizer,
    output_path: Path,
    batch_rows: int,
    label: str,
) -> int:
    eos_id = tokenizer.token_to_id("<EOS>")

    if eos_id is None:
        raise RuntimeError("<EOS> missing")

    temp = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    if temp.exists():
        temp.unlink()

    token_count = 0
    story_count = 0
    started = time.perf_counter()

    with temp.open(
        "wb",
        buffering=8 * 1024 * 1024,
    ) as handle:
        for batch_index, texts in enumerate(
            parquet_text_batches(
                files,
                batch_rows=batch_rows,
            ),
            start=1,
        ):
            encodings = tokenizer.encode_batch(
                texts
            )

            flattened: List[int] = []

            for encoding in encodings:
                flattened.extend(
                    encoding.ids
                )
                flattened.append(eos_id)

            array = np.asarray(
                flattened,
                dtype=np.uint16,
            )

            array.tofile(handle)

            token_count += int(
                array.size
            )

            story_count += len(
                encodings
            )

            if (
                batch_index == 1
                or batch_index % 100 == 0
            ):
                elapsed = max(
                    1e-9,
                    time.perf_counter() - started,
                )

                print(
                    f"  pack {label}: "
                    f"stories={story_count:,} "
                    f"tokens={token_count:,} "
                    f"rate={token_count / elapsed:,.0f} tok/s",
                    flush=True,
                )

    temp.replace(output_path)

    return token_count


def prepare_tinystories_if_needed(
    workspace: Path,
    tokenizer: Tokenizer,
) -> Dict[str, object]:
    train_bin = workspace / "train.bin"
    val_bin = workspace / "validation.bin"
    meta_path = workspace / "corpus_meta.json"

    if (
        meta_path.exists()
        and train_bin.exists()
        and val_bin.exists()
    ):
        meta = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )

        train_tokens = int(
            meta.get("train_tokens", 0)
        )

        validation_tokens = int(
            meta.get(
                "validation_tokens",
                0,
            )
        )

        if (
            train_tokens > 0
            and validation_tokens > 0
            and train_bin.stat().st_size
            == train_tokens * 2
            and val_bin.stat().st_size
            == validation_tokens * 2
        ):
            print(
                "Using existing packed TinyStories train/validation binaries.",
                flush=True,
            )
            return meta

    print(
        "Packed TinyStories binaries are missing; "
        "downloading official Parquet shards and rebuilding them "
        "with the EXISTING pretrained tokenizer.",
        flush=True,
    )

    cache_dir = workspace / "hf_cache"
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot = Path(
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            allow_patterns=[
                "data/*.parquet"
            ],
            cache_dir=str(
                cache_dir
            ),
        )
    )

    train_files = sorted(
        snapshot.glob(
            "data/train-*.parquet"
        )
    )

    val_files = sorted(
        snapshot.glob(
            "data/validation-*.parquet"
        )
    )

    if not train_files or not val_files:
        raise RuntimeError(
            "official TinyStories Parquet shards not found"
        )

    workspace.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_tokens = pack_parquet_with_existing_tokenizer(
        files=train_files,
        tokenizer=tokenizer,
        output_path=train_bin,
        batch_rows=2048,
        label="TinyStories train",
    )

    validation_tokens = pack_parquet_with_existing_tokenizer(
        files=val_files,
        tokenizer=tokenizer,
        output_path=val_bin,
        batch_rows=2048,
        label="TinyStories validation",
    )

    meta = {
        "dataset_repo": DATASET_REPO,
        "dtype": "uint16",
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "train_bin": str(
            train_bin
        ),
        "validation_bin": str(
            val_bin
        ),
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
    }

    meta_path.write_text(
        json.dumps(
            meta,
            indent=2,
        ),
        encoding="utf-8",
    )

    return meta


# ---------------------------------------------------------------------------
# Binary fixed-window access
# ---------------------------------------------------------------------------


class BinaryBlockDataset:
    """
    View a flat token stream as non-overlapping T-token inputs with a one-token
    target shift.

    Block i:
        x = tokens[i*T : i*T+T]
        y = tokens[i*T+1 : i*T+T+1]

    The view itself allocates no corpus-sized matrix.
    """

    def __init__(
        self,
        path: Path,
        token_count: int,
        seq_len: int,
        keep_in_ram: bool = False,
    ) -> None:
        self.path = Path(path)
        self.token_count = int(
            token_count
        )
        self.seq_len = int(
            seq_len
        )

        if self.token_count <= self.seq_len:
            raise ValueError(
                f"{self.path} is too small for one {self.seq_len}-token block"
            )

        mapped = np.memmap(
            self.path,
            mode="r",
            dtype=np.uint16,
            shape=(
                self.token_count,
            ),
        )

        if keep_in_ram:
            self.data = np.asarray(
                mapped
            ).copy()

            del mapped
        else:
            self.data = mapped

        self.num_blocks = (
            self.token_count - 1
        ) // self.seq_len

        itemsize = np.dtype(
            np.uint16
        ).itemsize

        self.windows = np.lib.stride_tricks.as_strided(
            self.data,
            shape=(
                self.num_blocks,
                self.seq_len + 1,
            ),
            strides=(
                self.seq_len * itemsize,
                itemsize,
            ),
            writeable=False,
        )

    def gather_into(
        self,
        block_indices: np.ndarray,
        x_dest: np.ndarray,
        y_dest: np.ndarray,
        rows: np.ndarray,
    ) -> None:
        if block_indices.size == 0:
            return

        selected = self.windows[
            block_indices
        ]

        x_dest[
            rows,
            :,
        ] = selected[
            :,
            :self.seq_len,
        ]

        y_dest[
            rows,
            :,
        ] = selected[
            :,
            1:,
        ]


# ---------------------------------------------------------------------------
# Curriculum schedules
# ---------------------------------------------------------------------------


@dataclass
class StagePlan:
    stage_index: int
    uo_ratio: float
    uo_sequences: int
    tiny_sequences: int
    total_sequences: int
    batches: int


@dataclass
class StageSchedule:
    domains: np.ndarray
    indices: np.ndarray
    plan: StagePlan
    batch_size: int

    def batch_slice(
        self,
        batch_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        start = (
            batch_index
            * self.batch_size
        )

        end = min(
            start + self.batch_size,
            self.plan.total_sequences,
        )

        return (
            self.domains[
                start:end
            ],
            self.indices[
                start:end
            ],
        )


def make_chunk_ordered_tiny_indices(
    total_blocks: int,
    count: int,
    chunk_size: int,
    rng: random.Random,
) -> np.ndarray:
    """
    Sample TinyStories blocks without replacement until a full corpus cycle
    is exhausted. The block order stays mostly sequential inside shuffled
    chunks, which is friendlier to memmap/page-cache access than fully random
    individual seeks.
    """

    if count <= 0:
        return np.empty(
            (0,),
            dtype=np.int64,
        )

    result: List[int] = []

    chunk_count = math.ceil(
        total_blocks / chunk_size
    )

    chunks = list(
        range(chunk_count)
    )

    while len(result) < count:
        rng.shuffle(chunks)

        for chunk_id in chunks:
            start = (
                chunk_id
                * chunk_size
            )

            end = min(
                start + chunk_size,
                total_blocks,
            )

            result.extend(
                range(start, end)
            )

            if len(result) >= count:
                break

    return np.asarray(
        result[:count],
        dtype=np.int64,
    )


def build_stage_schedule(
    stage_index: int,
    uo_ratio: float,
    uo_blocks: int,
    tiny_blocks: int,
    batch_size: int,
    tiny_chunk_blocks: int,
    seed: int,
) -> StageSchedule:
    if uo_ratio >= 1.0:
        tiny_count = 0
    else:
        tiny_count = int(
            round(
                uo_blocks
                * (1.0 - uo_ratio)
                / uo_ratio
            )
        )

    uo_count = uo_blocks
    total_count = (
        uo_count
        + tiny_count
    )

    rng = random.Random(
        seed
        + 100_003 * (
            stage_index + 1
        )
    )

    # UO appears exactly once per stage.
    uo_indices = list(
        range(
            uo_blocks
        )
    )

    rng.shuffle(
        uo_indices
    )

    uo_indices_np = np.asarray(
        uo_indices,
        dtype=np.int64,
    )

    tiny_indices = make_chunk_ordered_tiny_indices(
        total_blocks=tiny_blocks,
        count=tiny_count,
        chunk_size=tiny_chunk_blocks,
        rng=rng,
    )

    # Shuffle only the DOMAIN mask. The relative TinyStories block order is
    # retained so its disk access stays mostly sequential through shuffled
    # chunks; UO indices are independently shuffled.
    domains = np.concatenate(
        [
            np.ones(
                uo_count,
                dtype=np.uint8,
            ),
            np.zeros(
                tiny_count,
                dtype=np.uint8,
            ),
        ]
    )

    np_rng = np.random.default_rng(
        seed
        + 1_000_003
        * (
            stage_index + 1
        )
    )

    np_rng.shuffle(
        domains
    )

    indices = np.empty(
        total_count,
        dtype=np.int64,
    )

    uo_positions = np.flatnonzero(
        domains == 1
    )

    tiny_positions = np.flatnonzero(
        domains == 0
    )

    indices[
        uo_positions
    ] = uo_indices_np

    if tiny_count > 0:
        indices[
            tiny_positions
        ] = tiny_indices

    plan = StagePlan(
        stage_index=stage_index,
        uo_ratio=uo_ratio,
        uo_sequences=uo_count,
        tiny_sequences=tiny_count,
        total_sequences=total_count,
        batches=math.ceil(
            total_count
            / batch_size
        ),
    )

    return StageSchedule(
        domains=domains,
        indices=indices,
        plan=plan,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Background mixed-batch prefetcher
# ---------------------------------------------------------------------------


@dataclass
class PreparedBatch:
    buffer_id: int
    x_cpu: torch.Tensor
    y_cpu: torch.Tensor
    real_sequences: int
    uo_sequences: int
    tiny_sequences: int
    batch_index: int


class MixedBatchPrefetcher:
    """
    Background CPU assembler.

    It prepares mixed rows from two memory-mapped corpora while CUDA computes
    the previous batch. H2D copies use pinned host memory and static GPU tensors.
    """

    def __init__(
        self,
        schedule: StageSchedule,
        uo_dataset: BinaryBlockDataset,
        tiny_dataset: BinaryBlockDataset,
        seq_len: int,
        batch_size: int,
        start_batch: int,
        prefetch_buffers: int = 3,
    ) -> None:
        self.schedule = schedule
        self.uo_dataset = uo_dataset
        self.tiny_dataset = tiny_dataset
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.start_batch = start_batch
        self.prefetch_buffers = max(
            2,
            int(prefetch_buffers),
        )

        self.buffers: List[
            Tuple[
                torch.Tensor,
                torch.Tensor,
                np.ndarray,
                np.ndarray,
            ]
        ] = []

        for _ in range(
            self.prefetch_buffers
        ):
            x = torch.empty(
                (
                    batch_size,
                    seq_len,
                ),
                dtype=torch.long,
                pin_memory=True,
            )

            y = torch.empty(
                (
                    batch_size,
                    seq_len,
                ),
                dtype=torch.long,
                pin_memory=True,
            )

            self.buffers.append(
                (
                    x,
                    y,
                    x.numpy(),
                    y.numpy(),
                )
            )

        self.free_queue: queue.Queue = queue.Queue()
        self.ready_queue: queue.Queue = queue.Queue(
            maxsize=self.prefetch_buffers
        )

        for buffer_id in range(
            self.prefetch_buffers
        ):
            self.free_queue.put(
                buffer_id
            )

        self.error: Optional[
            BaseException
        ] = None

        self.thread = threading.Thread(
            target=self._worker,
            name="uo-mixed-batch-prefetch",
            daemon=True,
        )

        self.thread.start()

    def _worker(self) -> None:
        try:
            for batch_index in range(
                self.start_batch,
                self.schedule.plan.batches,
            ):
                buffer_id = self.free_queue.get()

                (
                    x_cpu,
                    y_cpu,
                    x_np,
                    y_np,
                ) = self.buffers[
                    buffer_id
                ]

                # Fill unused static rows safely.
                x_np.fill(0)
                y_np.fill(-100)

                (
                    domains,
                    indices,
                ) = self.schedule.batch_slice(
                    batch_index
                )

                real_count = len(
                    domains
                )

                rows = np.arange(
                    real_count,
                    dtype=np.int64,
                )

                uo_rows = rows[
                    domains == 1
                ]

                tiny_rows = rows[
                    domains == 0
                ]

                if uo_rows.size:
                    self.uo_dataset.gather_into(
                        block_indices=indices[
                            domains == 1
                        ],
                        x_dest=x_np,
                        y_dest=y_np,
                        rows=uo_rows,
                    )

                if tiny_rows.size:
                    self.tiny_dataset.gather_into(
                        block_indices=indices[
                            domains == 0
                        ],
                        x_dest=x_np,
                        y_dest=y_np,
                        rows=tiny_rows,
                    )

                prepared = PreparedBatch(
                    buffer_id=buffer_id,
                    x_cpu=x_cpu,
                    y_cpu=y_cpu,
                    real_sequences=real_count,
                    uo_sequences=int(
                        uo_rows.size
                    ),
                    tiny_sequences=int(
                        tiny_rows.size
                    ),
                    batch_index=batch_index,
                )

                self.ready_queue.put(
                    prepared
                )

            self.ready_queue.put(
                None
            )

        except BaseException as exc:
            self.error = exc
            self.ready_queue.put(
                None
            )

    def __iter__(
        self,
    ) -> Iterator[PreparedBatch]:
        while True:
            item = self.ready_queue.get()

            if item is None:
                if self.error is not None:
                    raise self.error
                break

            yield item

    def release(
        self,
        batch: PreparedBatch,
    ) -> None:
        self.free_queue.put(
            batch.buffer_id
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate_domain(
    model: UOMindLM,
    dataset: BinaryBlockDataset,
    batch_size: int,
    max_batches: int,
    device: torch.device,
    label: str,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    x_cpu = torch.empty(
        (
            batch_size,
            model.config.max_seq_len,
        ),
        dtype=torch.long,
        pin_memory=True,
    )

    y_cpu = torch.empty_like(
        x_cpu
    )

    x_np = x_cpu.numpy()
    y_np = y_cpu.numpy()

    x_gpu = torch.empty(
        (
            batch_size,
            model.config.max_seq_len,
        ),
        dtype=torch.long,
        device=device,
    )

    y_gpu = torch.empty_like(
        x_gpu
    )

    blocks = dataset.num_blocks

    total_batches = min(
        max_batches,
        math.ceil(
            blocks / batch_size
        ),
    )

    weighted_losses: List[
        torch.Tensor
    ] = []

    real_tokens = 0

    started = time.perf_counter()

    for batch_index in range(
        total_batches
    ):
        start = (
            batch_index
            * batch_size
        )

        end = min(
            start + batch_size,
            blocks,
        )

        real = end - start

        x_np.fill(0)
        y_np.fill(-100)

        indices = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        rows = np.arange(
            real,
            dtype=np.int64,
        )

        dataset.gather_into(
            block_indices=indices,
            x_dest=x_np,
            y_dest=y_np,
            rows=rows,
        )

        x_gpu.copy_(
            x_cpu,
            non_blocking=False,
        )

        y_gpu.copy_(
            y_cpu,
            non_blocking=False,
        )

        with amp_context():
            _, loss = model(
                x_gpu,
                y_gpu,
            )

        batch_real_tokens = (
            real
            * model.config.max_seq_len
        )

        weighted_losses.append(
            loss.detach() * batch_real_tokens
        )

        real_tokens += batch_real_tokens

    torch.cuda.synchronize()

    elapsed = max(
        1e-9,
        time.perf_counter() - started,
    )

    mean_loss = float(
        (
            torch.stack(weighted_losses).sum()
            / max(1, real_tokens)
        ).item()
    )

    if was_training:
        model.train()

    metrics = {
        "loss": mean_loss,
        "ppl": math.exp(
            min(
                mean_loss,
                20.0,
            )
        ),
        "tokens": real_tokens,
        "tok_per_sec": (
            real_tokens
            / elapsed
        ),
        "batches": total_batches,
    }

    print(
        f"  {label}: "
        f"loss={metrics['loss']:.4f} "
        f"ppl={metrics['ppl']:.3f} "
        f"tokens={real_tokens:,}",
        flush=True,
    )

    return metrics


# ---------------------------------------------------------------------------
# Optimizer / LR / checkpoints
# ---------------------------------------------------------------------------


def build_optimizer(
    model: UOMindLM,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    groups = model.parameter_groups(
        weight_decay
    )

    try:
        optimizer = torch.optim.AdamW(
            groups,
            lr=lr,
            betas=(
                0.9,
                0.95,
            ),
            eps=1e-8,
            fused=True,
        )

        print(
            "Optimizer: fused AdamW (fresh domain-adaptation optimizer)",
            flush=True,
        )

        return optimizer

    except Exception as exc:
        print(
            f"Fused AdamW unavailable ({type(exc).__name__}); "
            "using foreach AdamW.",
            flush=True,
        )

        return torch.optim.AdamW(
            groups,
            lr=lr,
            betas=(
                0.9,
                0.95,
            ),
            eps=1e-8,
            foreach=True,
        )


def learning_rate_at_step(
    step: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
    warmup_fraction: float,
) -> float:
    warmup_steps = max(
        10,
        int(
            total_steps
            * warmup_fraction
        ),
    )

    warmup_steps = min(
        warmup_steps,
        max(
            1,
            total_steps // 4,
        ),
    )

    if step < warmup_steps:
        return max_lr * (
            step + 1
        ) / warmup_steps

    if total_steps <= warmup_steps:
        return min_lr

    progress = (
        step - warmup_steps
    ) / max(
        1,
        total_steps - warmup_steps,
    )

    progress = min(
        1.0,
        max(
            0.0,
            progress,
        ),
    )

    cosine = 0.5 * (
        1.0
        + math.cos(
            math.pi
            * progress
        )
    )

    return (
        min_lr
        + (
            max_lr
            - min_lr
        )
        * cosine
    )


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    lr: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def recursive_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()

    if isinstance(value, dict):
        return {
            key: recursive_to_cpu(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            recursive_to_cpu(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            recursive_to_cpu(item)
            for item in value
        )

    return value


def optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(
            state.items()
        ):
            if torch.is_tensor(
                value
            ):
                state[
                    key
                ] = value.to(
                    device
                )


def save_checkpoint(
    path: Path,
    model: UOMindLM,
    optimizer: torch.optim.Optimizer,
    scaler,
    state: Dict[str, object],
    model_config: ModelConfig,
    curriculum_config: Dict[str, object],
    history: List[Dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    payload = {
        "model_state_dict": recursive_to_cpu(
            model.state_dict()
        ),
        "optimizer_state_dict": recursive_to_cpu(
            optimizer.state_dict()
        ),
        "scaler_state_dict": scaler.state_dict(),
        "state": state,
        "model_config": asdict(
            model_config
        ),
        "curriculum_config": curriculum_config,
        "history": history,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }

    torch.save(
        payload,
        temp,
    )

    temp.replace(
        path
    )


def restore_rng(
    payload: Dict[str, object],
) -> None:
    rng = payload.get(
        "rng"
    )

    if not rng:
        return

    try:
        random.setstate(
            rng["python"]
        )

        np.random.set_state(
            rng["numpy"]
        )

        torch.set_rng_state(
            rng["torch"]
        )

        torch.cuda.set_rng_state_all(
            rng["cuda"]
        )

    except Exception:
        pass


# ---------------------------------------------------------------------------
# Optional compilation
# ---------------------------------------------------------------------------


def maybe_compile(
    model: UOMindLM,
    enabled: bool,
    mode: str,
):
    if not enabled:
        print(
            "torch.compile disabled.",
            flush=True,
        )
        return model

    print(
        f"Compiling training graph with torch.compile(mode={mode!r})...",
        flush=True,
    )

    try:
        compiled = torch.compile(
            model,
            mode=mode,
            dynamic=False,
        )

        return compiled

    except Exception as exc:
        print(
            f"torch.compile unavailable; using eager: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        return model


# ---------------------------------------------------------------------------
# Curriculum training
# ---------------------------------------------------------------------------


def build_all_stage_plans(
    ratios: Sequence[float],
    uo_blocks: int,
    tiny_blocks: int,
    batch_size: int,
) -> List[StagePlan]:
    plans: List[
        StagePlan
    ] = []

    for stage_index, ratio in enumerate(
        ratios
    ):
        if ratio >= 1.0:
            tiny_count = 0
        else:
            tiny_count = int(
                round(
                    uo_blocks
                    * (
                        1.0
                        - ratio
                    )
                    / ratio
                )
            )

        total = (
            uo_blocks
            + tiny_count
        )

        plans.append(
            StagePlan(
                stage_index=stage_index,
                uo_ratio=ratio,
                uo_sequences=uo_blocks,
                tiny_sequences=tiny_count,
                total_sequences=total,
                batches=math.ceil(
                    total
                    / batch_size
                ),
            )
        )

    return plans


def copy_backup(
    source: Path,
    backup_dir: str,
) -> None:
    if not backup_dir:
        return

    destination_dir = Path(
        backup_dir
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination_dir
        / source.name,
    )


def run_curriculum(
    args: argparse.Namespace,
    model: UOMindLM,
    model_config: ModelConfig,
    tokenizer: Tokenizer,
    tokenizer_path: Path,
    pretrained_checkpoint: Path,
    pretrained_training_config: Dict[str, object],
    tiny_meta: Dict[str, object],
    uo_meta: Dict[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    device = torch.device(
        "cuda"
    )

    ratios = parse_ratios(
        args.uo_ratios
    )

    if len(ratios) != 5:
        print(
            f"Note: curriculum has {len(ratios)} stages rather than 5.",
            flush=True,
        )

    pretrained_batch_size = int(
        pretrained_training_config.get(
            "batch_size",
            args.fallback_batch_size,
        )
    )

    if args.batch_size != "auto":
        batch_size = int(args.batch_size)
        batch_reason = "explicit"
    else:
        # Continuation data can be dramatically smaller than TinyStories.
        # Preserve enough optimizer updates in the final 100%-UO pass while
        # never exceeding the already-proven pretraining batch size.
        uo_blocks_estimate = max(
            1,
            (int(uo_meta["train_tokens"]) - 1)
            // model_config.max_seq_len,
        )

        if args.min_uo_steps_per_stage > 0:
            max_batch_for_updates = max(
                args.min_batch_size,
                uo_blocks_estimate
                // args.min_uo_steps_per_stage,
            )

            # Tensor-core-friendly / allocator-friendly multiple of 8.
            if max_batch_for_updates >= 8:
                max_batch_for_updates = max(
                    args.min_batch_size,
                    (max_batch_for_updates // 8) * 8,
                )

            batch_size = min(
                pretrained_batch_size,
                max_batch_for_updates,
            )
            batch_reason = (
                f"auto: pretrained={pretrained_batch_size}, "
                f"target >= {args.min_uo_steps_per_stage} pure-UO updates"
            )
        else:
            batch_size = pretrained_batch_size
            batch_reason = "auto: reused pretrained batch"

    if batch_size <= 0:
        raise ValueError("batch size must be positive")

    print(
        f"Continuation batch size: {batch_size} ({batch_reason})",
        flush=True,
    )

    tiny_train = BinaryBlockDataset(
        path=Path(
            tiny_meta["train_bin"]
        ),
        token_count=int(
            tiny_meta["train_tokens"]
        ),
        seq_len=model_config.max_seq_len,
        keep_in_ram=False,
    )

    tiny_val = BinaryBlockDataset(
        path=Path(
            tiny_meta["validation_bin"]
        ),
        token_count=int(
            tiny_meta["validation_tokens"]
        ),
        seq_len=model_config.max_seq_len,
        keep_in_ram=False,
    )

    uo_train = BinaryBlockDataset(
        path=Path(
            uo_meta["train_bin"]
        ),
        token_count=int(
            uo_meta["train_tokens"]
        ),
        seq_len=model_config.max_seq_len,
        keep_in_ram=args.uo_in_ram,
    )

    uo_val: Optional[
        BinaryBlockDataset
    ] = None

    if int(
        uo_meta.get(
            "validation_tokens",
            0,
        )
    ) > model_config.max_seq_len:
        uo_val = BinaryBlockDataset(
            path=Path(
                uo_meta[
                    "validation_bin"
                ]
            ),
            token_count=int(
                uo_meta[
                    "validation_tokens"
                ]
            ),
            seq_len=model_config.max_seq_len,
            keep_in_ram=True,
        )

    print()
    print("Block counts", flush=True)
    print("------------", flush=True)
    print(
        f"TinyStories train: {tiny_train.num_blocks:,} x 128",
        flush=True,
    )
    print(
        f"TinyStories val:   {tiny_val.num_blocks:,} x 128",
        flush=True,
    )
    print(
        f"UO train:          {uo_train.num_blocks:,} x 128",
        flush=True,
    )

    if uo_val is not None:
        print(
            f"UO val:            {uo_val.num_blocks:,} x 128",
            flush=True,
        )
    else:
        print(
            "UO val:            disabled",
            flush=True,
        )

    plans = build_all_stage_plans(
        ratios=ratios,
        uo_blocks=uo_train.num_blocks,
        tiny_blocks=tiny_train.num_blocks,
        batch_size=batch_size,
    )

    print()
    print("Curriculum", flush=True)
    print("----------", flush=True)

    total_steps = sum(
        plan.batches
        for plan in plans
    )

    for plan in plans:
        actual_ratio = (
            plan.uo_sequences
            / plan.total_sequences
        )

        print(
            f"Stage {plan.stage_index + 1}: "
            f"target UO={plan.uo_ratio * 100:5.1f}% | "
            f"actual={actual_ratio * 100:5.2f}% | "
            f"UO={plan.uo_sequences:,} seq | "
            f"Tiny={plan.tiny_sequences:,} seq | "
            f"batches={plan.batches:,}",
            flush=True,
        )

    print(
        f"Total continuation steps: {total_steps:,}",
        flush=True,
    )

    # New optimizer on pretrained weights.
    optimizer = build_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = make_grad_scaler()

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_path = (
        checkpoint_dir
        / "latest.pt"
    )

    best_uo_path = (
        checkpoint_dir
        / "best_uo.pt"
    )

    final_path = (
        checkpoint_dir
        / "final.pt"
    )

    history_path = (
        output_dir
        / "curriculum_history.json"
    )

    history: List[
        Dict[str, object]
    ] = []

    state: Dict[str, object] = {
        "stage_index": 0,
        "batch_in_stage": 0,
        "global_step": 0,
        "real_tokens_seen": 0,
        "uo_tokens_seen": 0,
        "tiny_tokens_seen": 0,
        "best_uo_val_loss": float(
            "inf"
        ),
    }

    # Resume only from a continuation checkpoint, never from the original
    # TinyStories optimizer state.
    resume_payload = None

    if args.resume == "auto":
        if latest_path.exists():
            resume_payload = torch.load(
                latest_path,
                map_location="cpu",
                weights_only=False,
            )

    elif args.resume != "none":
        resume_path = Path(
            args.resume
        )

        if not resume_path.exists():
            raise FileNotFoundError(
                resume_path
            )

        resume_payload = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )

    if resume_payload is not None:
        print(
            f"Resuming UO curriculum from "
            f"{args.resume if args.resume != 'auto' else latest_path}",
            flush=True,
        )

        saved_cfg = resume_payload.get(
            "curriculum_config",
            {},
        )

        saved_ratios = tuple(
            saved_cfg.get(
                "uo_ratios",
                [],
            )
        )

        if (
            saved_ratios
            and saved_ratios
            != tuple(ratios)
        ):
            raise RuntimeError(
                "Resume curriculum ratios do not match current --uo-ratios"
            )

        saved_fingerprint = saved_cfg.get("uo_fingerprint")
        if (
            saved_fingerprint
            and saved_fingerprint != uo_meta["fingerprint"]
        ):
            raise RuntimeError(
                "The UO story corpus changed since this continuation "
                "checkpoint was created. Refusing to resume an old schedule."
            )

        saved_batch_size = saved_cfg.get("batch_size")
        if (
            saved_batch_size is not None
            and int(saved_batch_size) != batch_size
        ):
            raise RuntimeError(
                f"Resume batch size mismatch: checkpoint={saved_batch_size}, "
                f"current={batch_size}"
            )

        model.load_state_dict(
            resume_payload[
                "model_state_dict"
            ],
            strict=True,
        )

        optimizer.load_state_dict(
            resume_payload[
                "optimizer_state_dict"
            ]
        )

        optimizer_to_device(
            optimizer,
            device,
        )

        scaler.load_state_dict(
            resume_payload.get(
                "scaler_state_dict",
                {},
            )
        )

        state.update(
            resume_payload[
                "state"
            ]
        )

        history = list(
            resume_payload.get(
                "history",
                [],
            )
        )

        restore_rng(
            resume_payload
        )

    compile_enabled = (
        not args.no_compile
        and total_steps >= args.compile_min_steps
    )

    curriculum_config = {
        "source_checkpoint": str(
            pretrained_checkpoint
        ),
        "tokenizer": str(
            tokenizer_path
        ),
        "uo_ratios": list(
            ratios
        ),
        "batch_size": batch_size,
        "pretrained_batch_size": pretrained_batch_size,
        "min_uo_steps_per_stage": args.min_uo_steps_per_stage,
        "seq_len": model_config.max_seq_len,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_fraction": args.warmup_fraction,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "tiny_chunk_blocks": args.tiny_chunk_blocks,
        "uo_val_fraction": args.uo_val_fraction,
        "uo_fingerprint": uo_meta[
            "fingerprint"
        ],
        "total_steps": total_steps,
        "compile_enabled": compile_enabled,
        "compile_min_steps": args.compile_min_steps,
    }

    # Static GPU buffers.
    x_gpu = torch.empty(
        (
            batch_size,
            model_config.max_seq_len,
        ),
        dtype=torch.long,
        device=device,
    )

    y_gpu = torch.empty_like(
        x_gpu
    )

    if (
        not args.no_compile
        and not compile_enabled
    ):
        print(
            f"Skipping torch.compile: curriculum has only {total_steps:,} "
            f"steps (< --compile-min-steps {args.compile_min_steps:,}); "
            "eager avoids compile overhead.",
            flush=True,
        )

    train_model = maybe_compile(
        model=model,
        enabled=compile_enabled,
        mode=args.compile_mode,
    )

    # First-use compile can be slow. Compile it now with a real-sized static
    # tensor before timing curriculum steps.
    if compile_enabled:
        print(
            "Warming compiled graph...",
            flush=True,
        )

        x_gpu.zero_()
        y_gpu.zero_()

        optimizer.zero_grad(
            set_to_none=True
        )

        with amp_context():
            _, warm_loss = train_model(
                x_gpu,
                y_gpu,
            )

        scaler.scale(
            warm_loss
        ).backward()

        optimizer.zero_grad(
            set_to_none=True
        )

        del warm_loss

        torch.cuda.synchronize()

        print(
            "Compiled graph warmup complete.",
            flush=True,
        )

    # Baseline eval before first UO update unless resuming after step 0.
    if int(
        state[
            "global_step"
        ]
    ) == 0:
        print()
        print(
            "Pre-curriculum validation baseline:",
            flush=True,
        )

        baseline_tiny = evaluate_domain(
            model=model,
            dataset=tiny_val,
            batch_size=min(
                args.val_batch_size,
                batch_size,
            ),
            max_batches=args.tiny_val_batches,
            device=device,
            label="TinyStories val",
        )

        baseline_uo = None

        if uo_val is not None:
            baseline_uo = evaluate_domain(
                model=model,
                dataset=uo_val,
                batch_size=min(
                    args.val_batch_size,
                    batch_size,
                ),
                max_batches=args.uo_val_batches,
                device=device,
                label="UO val",
            )

        history.append(
            {
                "event": "baseline",
                "global_step": 0,
                "tiny_validation": baseline_tiny,
                "uo_validation": baseline_uo,
                "geometry": model.geometry_gain_summary(),
            }
        )

        history_path.write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding="utf-8",
        )

    start_stage = int(
        state[
            "stage_index"
        ]
    )

    start_batch = int(
        state[
            "batch_in_stage"
        ]
    )

    global_step = int(
        state[
            "global_step"
        ]
    )

    real_tokens_seen = int(
        state[
            "real_tokens_seen"
        ]
    )

    uo_tokens_seen = int(
        state[
            "uo_tokens_seen"
        ]
    )

    tiny_tokens_seen = int(
        state[
            "tiny_tokens_seen"
        ]
    )

    best_uo_val_loss = float(
        state[
            "best_uo_val_loss"
        ]
    )

    loss_samples: List[
        torch.Tensor
    ] = []

    log_processed_start = 0
    processed_since_log = 0
    real_since_log = 0
    log_started = time.perf_counter()

    model.train()

    for stage_index in range(
        start_stage,
        len(
            ratios
        ),
    ):
        ratio = ratios[
            stage_index
        ]

        schedule = build_stage_schedule(
            stage_index=stage_index,
            uo_ratio=ratio,
            uo_blocks=uo_train.num_blocks,
            tiny_blocks=tiny_train.num_blocks,
            batch_size=batch_size,
            tiny_chunk_blocks=args.tiny_chunk_blocks,
            seed=args.seed,
        )

        stage_start_batch = (
            start_batch
            if stage_index
            == start_stage
            else 0
        )

        print()
        print(
            "=" * 78,
            flush=True,
        )

        print(
            f"STAGE {stage_index + 1}/{len(ratios)} | "
            f"{ratio * 100:.0f}% UO / "
            f"{(1.0 - ratio) * 100:.0f}% TinyStories | "
            f"one complete UO pass",
            flush=True,
        )

        print(
            f"Starting batch "
            f"{stage_start_batch:,}/{schedule.plan.batches:,}",
            flush=True,
        )

        print(
            "=" * 78,
            flush=True,
        )

        # Stage-local log timers exclude previous validation/checkpoint time.
        loss_samples.clear()
        processed_since_log = 0
        real_since_log = 0
        log_started = time.perf_counter()

        prefetcher = MixedBatchPrefetcher(
            schedule=schedule,
            uo_dataset=uo_train,
            tiny_dataset=tiny_train,
            seq_len=model_config.max_seq_len,
            batch_size=batch_size,
            start_batch=stage_start_batch,
            prefetch_buffers=args.prefetch_buffers,
        )

        stage_uo_seen = 0
        stage_tiny_seen = 0
        stage_real_seen = 0
        stage_started = time.perf_counter()

        for batch in prefetcher:
            lr = learning_rate_at_step(
                step=global_step,
                total_steps=total_steps,
                max_lr=args.lr,
                min_lr=args.min_lr,
                warmup_fraction=args.warmup_fraction,
            )

            set_optimizer_lr(
                optimizer,
                lr,
            )

            # Blocking copies are tiny and make it immediately safe for the
            # background worker to reuse this pinned buffer while the GPU does
            # forward/backward on static CUDA tensors.
            x_gpu.copy_(
                batch.x_cpu,
                non_blocking=False,
            )

            y_gpu.copy_(
                batch.y_cpu,
                non_blocking=False,
            )

            prefetcher.release(
                batch
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with amp_context():
                _, loss = train_model(
                    x_gpu,
                    y_gpu,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip,
                foreach=True,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            loss_samples.append(
                loss.detach()
            )

            global_step += 1

            real_tokens = (
                batch.real_sequences
                * model_config.max_seq_len
            )

            uo_tokens = (
                batch.uo_sequences
                * model_config.max_seq_len
            )

            tiny_tokens = (
                batch.tiny_sequences
                * model_config.max_seq_len
            )

            real_tokens_seen += real_tokens
            uo_tokens_seen += uo_tokens
            tiny_tokens_seen += tiny_tokens

            stage_real_seen += real_tokens
            stage_uo_seen += uo_tokens
            stage_tiny_seen += tiny_tokens

            # Hardware processed a full static B*T even on the final padded
            # batch. Use it for tok/s, and separately report real domain tokens.
            processed_since_log += (
                batch_size
                * model_config.max_seq_len
            )

            real_since_log += real_tokens

            next_batch = (
                batch.batch_index
                + 1
            )

            if (
                global_step
                % args.log_every
                == 0
            ):
                torch.cuda.synchronize()

                now = time.perf_counter()

                interval = max(
                    1e-9,
                    now
                    - log_started,
                )

                mean_loss = float(
                    torch.stack(
                        loss_samples
                    ).mean().item()
                )

                if not math.isfinite(
                    mean_loss
                ):
                    raise FloatingPointError(
                        f"non-finite mean loss at global step {global_step}"
                    )

                progress = (
                    100.0
                    * next_batch
                    / schedule.plan.batches
                )

                print(
                    f"stage={stage_index + 1}/{len(ratios)} "
                    f"mix={ratio * 100:.0f}%UO "
                    f"batch={next_batch:,}/{schedule.plan.batches:,} "
                    f"({progress:5.1f}%) "
                    f"global={global_step:,}/{total_steps:,} "
                    f"loss={mean_loss:.4f} "
                    f"ppl={math.exp(min(mean_loss, 20.0)):.2f} "
                    f"lr={lr:.2e} "
                    f"gpu_tok/s={processed_since_log / interval:,.0f} "
                    f"real_tok/s={real_since_log / interval:,.0f}",
                    flush=True,
                )

                loss_samples.clear()
                processed_since_log = 0
                real_since_log = 0
                log_started = now

            if (
                args.checkpoint_every
                > 0
                and global_step
                % args.checkpoint_every
                == 0
            ):
                torch.cuda.synchronize()

                checkpoint_state = {
                    "stage_index": stage_index,
                    "batch_in_stage": next_batch,
                    "global_step": global_step,
                    "real_tokens_seen": real_tokens_seen,
                    "uo_tokens_seen": uo_tokens_seen,
                    "tiny_tokens_seen": tiny_tokens_seen,
                    "best_uo_val_loss": best_uo_val_loss,
                }

                print(
                    f"Saving resume checkpoint at global step {global_step:,}...",
                    flush=True,
                )

                save_checkpoint(
                    path=latest_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    state=checkpoint_state,
                    model_config=model_config,
                    curriculum_config=curriculum_config,
                    history=history,
                )

                copy_backup(
                    latest_path,
                    args.backup_dir,
                )

        torch.cuda.synchronize()

        tail_loss = None
        if loss_samples:
            tail_loss = float(
                torch.stack(loss_samples).mean().item()
            )
            loss_samples.clear()

            if not math.isfinite(tail_loss):
                raise FloatingPointError(
                    f"non-finite loss at end of stage {stage_index + 1}"
                )

        stage_elapsed = max(
            1e-9,
            time.perf_counter()
            - stage_started,
        )

        actual_stage_ratio = (
            stage_uo_seen
            / max(
                1,
                stage_real_seen,
            )
        )

        print()
        print(
            f"Stage {stage_index + 1} complete: "
            f"UO={stage_uo_seen:,} tok | "
            f"Tiny={stage_tiny_seen:,} tok | "
            f"actual UO={actual_stage_ratio * 100:.3f}% | "
            f"real tok/s={stage_real_seen / stage_elapsed:,.0f}"
            + (f" | tail_loss={tail_loss:.4f}" if tail_loss is not None else ""),
            flush=True,
        )

        print(
            "Validation:",
            flush=True,
        )

        tiny_metrics = evaluate_domain(
            model=model,
            dataset=tiny_val,
            batch_size=min(
                args.val_batch_size,
                batch_size,
            ),
            max_batches=args.tiny_val_batches,
            device=device,
            label="TinyStories val",
        )

        uo_metrics = None

        if uo_val is not None:
            uo_metrics = evaluate_domain(
                model=model,
                dataset=uo_val,
                batch_size=min(
                    args.val_batch_size,
                    batch_size,
                ),
                max_batches=args.uo_val_batches,
                device=device,
                label="UO val",
            )

        geometry = model.geometry_gain_summary()

        stage_record = {
            "event": "stage_complete",
            "stage_index": stage_index,
            "stage_number": stage_index + 1,
            "target_uo_ratio": ratio,
            "actual_uo_ratio": actual_stage_ratio,
            "global_step": global_step,
            "real_tokens_seen": real_tokens_seen,
            "uo_tokens_seen": uo_tokens_seen,
            "tiny_tokens_seen": tiny_tokens_seen,
            "stage_uo_tokens": stage_uo_seen,
            "stage_tiny_tokens": stage_tiny_seen,
            "stage_elapsed_seconds": stage_elapsed,
            "tiny_validation": tiny_metrics,
            "uo_validation": uo_metrics,
            "geometry": geometry,
        }

        history.append(
            stage_record
        )

        history_path.write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding="utf-8",
        )

        next_state = {
            "stage_index": stage_index + 1,
            "batch_in_stage": 0,
            "global_step": global_step,
            "real_tokens_seen": real_tokens_seen,
            "uo_tokens_seen": uo_tokens_seen,
            "tiny_tokens_seen": tiny_tokens_seen,
            "best_uo_val_loss": best_uo_val_loss,
        }

        stage_pct = int(
            round(
                ratio
                * 100
            )
        )

        stage_path = (
            checkpoint_dir
            / (
                f"stage_{stage_index + 1}_"
                f"{stage_pct}pct_uo.pt"
            )
        )

        if (
            uo_metrics is not None
            and uo_metrics[
                "loss"
            ]
            < best_uo_val_loss
        ):
            best_uo_val_loss = float(
                uo_metrics[
                    "loss"
                ]
            )

            next_state[
                "best_uo_val_loss"
            ] = best_uo_val_loss

            print(
                f"New best UO validation loss "
                f"{best_uo_val_loss:.4f}; saving best_uo.pt",
                flush=True,
            )

            save_checkpoint(
                path=best_uo_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                state=next_state,
                model_config=model_config,
                curriculum_config=curriculum_config,
                history=history,
            )

            copy_backup(
                best_uo_path,
                args.backup_dir,
            )

        save_checkpoint(
            path=stage_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            state=next_state,
            model_config=model_config,
            curriculum_config=curriculum_config,
            history=history,
        )

        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            state=next_state,
            model_config=model_config,
            curriculum_config=curriculum_config,
            history=history,
        )

        copy_backup(
            stage_path,
            args.backup_dir,
        )

        copy_backup(
            latest_path,
            args.backup_dir,
        )

        # Reset resume offset after first resumed stage.
        start_batch = 0

    final_state = {
        "stage_index": len(
            ratios
        ),
        "batch_in_stage": 0,
        "global_step": global_step,
        "real_tokens_seen": real_tokens_seen,
        "uo_tokens_seen": uo_tokens_seen,
        "tiny_tokens_seen": tiny_tokens_seen,
        "best_uo_val_loss": best_uo_val_loss,
    }

    save_checkpoint(
        path=final_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        state=final_state,
        model_config=model_config,
        curriculum_config=curriculum_config,
        history=history,
    )

    copy_backup(
        final_path,
        args.backup_dir,
    )

    summary = {
        "source_checkpoint": str(
            pretrained_checkpoint
        ),
        "model_config": asdict(
            model_config
        ),
        "curriculum_config": curriculum_config,
        "final_state": final_state,
        "geometry": model.geometry_gain_summary(),
        "history_file": str(
            history_path
        ),
        "final_checkpoint": str(
            final_path
        ),
        "best_uo_checkpoint": (
            str(
                best_uo_path
            )
            if best_uo_path.exists()
            else None
        ),
    }

    summary_path = (
        output_dir
        / "curriculum_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 78,
        flush=True,
    )

    print(
        "UO CURRICULUM COMPLETE",
        flush=True,
    )

    print(
        "=" * 78,
        flush=True,
    )

    print(
        f"global steps:       {global_step:,}",
        flush=True,
    )

    print(
        f"real tokens seen:   {real_tokens_seen:,}",
        flush=True,
    )

    print(
        f"UO tokens seen:     {uo_tokens_seen:,}",
        flush=True,
    )

    print(
        f"Tiny tokens seen:   {tiny_tokens_seen:,}",
        flush=True,
    )

    print(
        f"final checkpoint:   {final_path}",
        flush=True,
    )

    if best_uo_path.exists():
        print(
            f"best UO checkpoint: {best_uo_path}",
            flush=True,
        )

    print(
        f"history:            {history_path}",
        flush=True,
    )

    return summary


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue a TinyStories-pretrained UO-Mind model through a "
            "five-stage TinyStories -> Ultima Online domain curriculum."
        )
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=str(
            DEFAULT_WORKSPACE
        ),
    )

    parser.add_argument(
        "--uo-dir",
        type=str,
        default=str(
            DEFAULT_UO_DIR
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="auto",
        help=(
            "TinyStories pretrained checkpoint or 'auto'. "
            "Auto prefers checkpoints/best.pt."
        ),
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="auto",
        help=(
            "Exact tokenizer.json used for pretraining or 'auto'."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=(
            "Defaults to <workspace>/uo_curriculum."
        ),
    )

    parser.add_argument(
        "--required-uo-files",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--allow-any-file-count",
        action="store_true",
    )

    parser.add_argument(
        "--uo-val-fraction",
        type=float,
        default=0.05,
        help=(
            "Per-file UO story fraction reserved for real held-out UO "
            "validation. Use 0 to train every story."
        ),
    )

    parser.add_argument(
        "--uo-ratios",
        type=str,
        default="0.20,0.40,0.60,0.80,1.00",
        help=(
            "UO fractions for successive one-UO-pass stages."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help=(
            "Integer or auto. Auto reuses the TinyStories pretraining batch."
        ),
    )

    parser.add_argument(
        "--fallback-batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--min-uo-steps-per-stage",
        type=int,
        default=20,
        help=(
            "When --batch-size auto, cap batch size so the final 100%-UO "
            "pass gets about this many optimizer updates when possible."
        ),
    )

    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=16,
        help="Lower bound for the automatic continuation batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help=(
            "Fresh continuation max LR."
        ),
    )

    parser.add_argument(
        "--min-lr",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tiny-chunk-blocks",
        type=int,
        default=256,
        help=(
            "Sequential TinyStories blocks per shuffled sampling chunk."
        ),
    )

    parser.add_argument(
        "--prefetch-buffers",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--uo-in-ram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep the packed UO training stream in CPU RAM. Usually small "
            "and faster for mixed random row access."
        ),
    )

    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
    )

    parser.add_argument(
        "--no-compile",
        action="store_true",
    )

    parser.add_argument(
        "--compile-min-steps",
        type=int,
        default=500,
        help=(
            "Skip torch.compile automatically for shorter curricula, where "
            "compile latency is likely larger than the training time saved."
        ),
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5000,
        help=(
            "0 disables mid-stage resume checkpoints. Stage boundaries are "
            "always saved; 5000 keeps large checkpoint writes out of short stages."
        ),
    )

    parser.add_argument(
        "--tiny-val-batches",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--uo-val-batches",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--pack-batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help=(
            "'auto', 'none', or a UO curriculum checkpoint."
        ),
    )

    parser.add_argument(
        "--backup-dir",
        type=str,
        default="",
        help=(
            "Optional mounted persistent directory for checkpoint copies."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required. In Colab select a GPU runtime."
        )

    if not (
        0.0
        <= args.uo_val_fraction
        < 1.0
    ):
        raise ValueError(
            "--uo-val-fraction must be in [0,1)"
        )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision(
        "high"
    )

    set_seed(
        args.seed
    )

    workspace = Path(
        args.workspace
    )

    output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir
        else workspace
        / "uo_curriculum"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Runtime", flush=True)
    print("-------", flush=True)
    print(
        f"torch:  {torch.__version__}",
        flush=True,
    )
    print(
        f"cuda:   {torch.version.cuda}",
        flush=True,
    )
    print(
        f"gpu:    {torch.cuda.get_device_name(0)}",
        flush=True,
    )
    print(
        f"memory: {gpu_memory_line()}",
        flush=True,
    )
    print()

    checkpoint_path = find_checkpoint(
        requested=args.checkpoint,
        workspace=workspace,
    )

    tokenizer_path = find_tokenizer(
        requested=args.tokenizer,
        workspace=workspace,
        checkpoint_path=checkpoint_path,
    )

    print(
        f"Tokenizer: {tokenizer_path}",
        flush=True,
    )

    tokenizer = Tokenizer.from_file(
        str(
            tokenizer_path
        )
    )

    model, pretrained_payload, model_config = load_pretrained_model(
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer,
        device=torch.device(
            "cuda"
        ),
    )

    pretrained_training_config = dict(
        pretrained_payload.get("training_config", {})
    )
    del pretrained_payload
    gc.collect()

    tiny_meta = prepare_tinystories_if_needed(
        workspace=workspace,
        tokenizer=tokenizer,
    )

    uo_files = discover_uo_files(
        uo_dir=Path(
            args.uo_dir
        ),
        required_files=args.required_uo_files,
        allow_any_count=args.allow_any_file_count,
    )

    uo_meta = prepare_uo_corpus(
        output_dir=output_dir,
        files=uo_files,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        val_fraction=args.uo_val_fraction,
        seed=args.seed + 7000,
        pack_batch_size=args.pack_batch_size,
    )

    print()
    print("UO corpus", flush=True)
    print("---------", flush=True)
    print(
        f"train stories:      {uo_meta['total_train_stories']:,}",
        flush=True,
    )
    print(
        f"validation stories: {uo_meta['total_validation_stories']:,}",
        flush=True,
    )
    print(
        f"train tokens:       {int(uo_meta['train_tokens']):,}",
        flush=True,
    )
    print(
        f"validation tokens:  {int(uo_meta['validation_tokens']):,}",
        flush=True,
    )
    print()

    run_curriculum(
        args=args,
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        pretrained_checkpoint=checkpoint_path,
        pretrained_training_config=pretrained_training_config,
        tiny_meta=tiny_meta,
        uo_meta=uo_meta,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
