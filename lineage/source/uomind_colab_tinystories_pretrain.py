#!/usr/bin/env python3
"""
uomind_colab_tinystories_pretrain.py

One-file Google Colab pretraining launcher for the short-context UO-Mind model.

Default behavior
----------------
1. Checks for CUDA and prints the GPU/runtime configuration.
2. Installs only the missing lightweight data/tokenizer dependencies.
3. Downloads the official roneneldan/TinyStories Parquet shards once.
4. Trains a 4096-token byte-level BPE tokenizer once.
5. Packs ALL TinyStories train/validation text once into flat uint16 binaries.
6. Benchmarks the geometry attention implementation on the current GPU.
7. Benchmarks practical batch sizes and chooses the fastest safe batch.
8. Optionally tests torch.compile and keeps it only when it is beneficial.
9. Pretrains the ~13.64M parameter NoPE + fixed relational geometry model.
10. Makes THREE full token passes by default.
11. Saves resumable checkpoints and best/final weights.

Training hot path
-----------------
The network, Hugging Face, Parquet, and tokenizer are completely absent from
training after preprocessing. Training reads local packed uint16 tokens using
large sequential chunks and reusable pinned/static buffers.

Run in Colab
------------
    !python uomind_colab_tinystories_pretrain.py

Useful overrides
----------------
    !python uomind_colab_tinystories_pretrain.py --passes 3 --batch-size auto
    !python uomind_colab_tinystories_pretrain.py --no-compile
    !python uomind_colab_tinystories_pretrain.py --resume none

Notes
-----
- Default context is exactly 128 tokens.
- No additive positional embedding is used.
- Structural position enters only as fixed multi-scale relative geometry.
- Geometry scale trust is learned with a bounded per-layer/per-head gain.
- The default TinyStories source is the official Parquet conversion in
  roneneldan/TinyStories/data.
- Data are downloaded and tokenized once; every later pass is local.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard-library bootstrap. Set performance-related env vars BEFORE torch.
# ---------------------------------------------------------------------------

import argparse
import contextlib
import gc
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def ensure_packages() -> None:
    """Install only packages that are missing from the Colab runtime."""

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
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


# ---------------------------------------------------------------------------
# Constants / reserved vocabulary
# ---------------------------------------------------------------------------

DATASET_REPO = "roneneldan/TinyStories"
DEFAULT_WORKSPACE = "/content/uomind_tinystories"

BASE_SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<NPC>",
    "<PLAYER>",
    "<STATE>",
    "<MEMORY>",
    "<NOW>",
    "<ACTION>",
    "<TONE>",
    "<SAY>",
    "<REL>",
    "<MOOD>",
    "<WORLD>",
    "<FACT>",
    "<ITEM>",
    "<GOLD>",
    "<QUEST>",
    "<TARGET>",
    "<LOCATION>",
    "<FACTION>",
    "<CRIME>",
    "<FAME>",
    "<KARMA>",
    "<HEALTH>",
    "<END_STATE>",
]

# Keep spare one-token control symbols available for later UO fine-tuning.
RESERVED_TOKENS = [f"<R{i:02d}>" for i in range(32)]
SPECIAL_TOKENS = BASE_SPECIAL_TOKENS + RESERVED_TOKENS


# ---------------------------------------------------------------------------
# Configuration
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
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if len(self.geometry_scales) != self.n_heads:
            raise ValueError("one geometry scale is required per head")
        if self.max_seq_len != 128:
            raise ValueError("this pretraining build is intentionally fixed at T=128")
        if self.vocab_size > 65535:
            raise ValueError("vocab must fit uint16 packed storage")
        if self.attention_backend not in {"sdpa", "manual"}:
            raise ValueError("attention_backend must be 'sdpa' or 'manual'")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Small compile-friendly RMSNorm with FP32 variance accumulation."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        normed = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(dtype) * self.weight


class GeometryAttention(nn.Module):
    """
    Fused-QKV causal self-attention with NoPE and separate relational geometry.

        score[h,i,j] = Q_i K_j^T / sqrt(dh) + gain[h] * (-|i-j|/tau[h])

    There is no additive positional vector in token states.
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

        # This production pretraining build is explicitly FP16 on a T4.
        # Storing the fixed field in FP16 avoids recasting ~2M geometry values
        # across the 16 layers on every forward. The field's bounded numeric
        # range is comfortably representable in FP16.
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
        # Only the tiny [H] gain vector is cast each forward. The full fixed
        # geometry and causal field already live in FP16 for the T4 hot path.
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
                f"static training graph requires seq_len={self.max_seq_len}, got {seq_len}"
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

        # [B, H, T, Hd]
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
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tanh approximation is substantially easier to fuse than erf GELU.
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class UOMindBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = GeometryAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
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
            [UOMindBlock(config) for _ in range(config.n_layers)]
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

        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
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
            [block.attn.gains() for block in self.blocks],
            dim=0,
        )

        return {
            "mean": float(gains.mean().detach().cpu().item()),
            "min": float(gains.min().detach().cpu().item()),
            "max": float(gains.max().detach().cpu().item()),
            "matrix": gains.detach().cpu().tolist(),
        }

    def parameter_groups(self, weight_decay: float) -> List[Dict[str, object]]:
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
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_name() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    return torch.cuda.get_device_name(0)


def gpu_memory_line() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"

    free, total = torch.cuda.mem_get_info()
    return (
        f"free={free / 2**30:.2f} GiB / "
        f"total={total / 2**30:.2f} GiB"
    )


def amp_context(enabled: bool = True):
    if enabled:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
    return contextlib.nullcontext()


def make_grad_scaler(enabled: bool = True):
    # Compatible with both recent and older Colab PyTorch builds.
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def print_environment() -> None:
    print("Runtime", flush=True)
    print("-------", flush=True)
    print(f"python: {sys.version.split()[0]}", flush=True)
    print(f"torch:  {torch.__version__}", flush=True)
    print(f"cuda:   {torch.version.cuda}", flush=True)
    print(f"gpu:    {gpu_name()}", flush=True)
    print(f"memory: {gpu_memory_line()}", flush=True)
    print(flush=True)


# ---------------------------------------------------------------------------
# TinyStories download / tokenizer / binary packing
# ---------------------------------------------------------------------------


def download_tinystories(workspace: Path) -> Tuple[List[Path], List[Path]]:
    cache_dir = workspace / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Downloading/caching official TinyStories Parquet shards from {DATASET_REPO}...",
        flush=True,
    )

    snapshot = Path(
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            allow_patterns=["data/*.parquet"],
            cache_dir=str(cache_dir),
        )
    )

    train_files = sorted(
        snapshot.glob("data/train-*.parquet")
    )

    validation_files = sorted(
        snapshot.glob("data/validation-*.parquet")
    )

    if not train_files:
        raise RuntimeError("TinyStories training Parquet shards were not found")

    if not validation_files:
        raise RuntimeError("TinyStories validation Parquet shard was not found")

    print(
        f"TinyStories ready: {len(train_files)} train shard(s), "
        f"{len(validation_files)} validation shard(s)",
        flush=True,
    )

    return train_files, validation_files


def parquet_text_batches(
    files: Sequence[Path],
    batch_rows: int = 4096,
    max_rows: Optional[int] = None,
) -> Iterator[List[str]]:
    emitted = 0

    for path in files:
        parquet_file = pq.ParquetFile(path)

        for batch in parquet_file.iter_batches(
            batch_size=batch_rows,
            columns=["text"],
            use_threads=True,
        ):
            texts = batch.column(0).to_pylist()
            texts = [text for text in texts if text]

            if max_rows is not None:
                remaining = max_rows - emitted

                if remaining <= 0:
                    return

                texts = texts[:remaining]

            if texts:
                emitted += len(texts)
                yield texts

            if max_rows is not None and emitted >= max_rows:
                return


def count_parquet_rows(files: Sequence[Path]) -> int:
    return sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in files
    )


def train_or_load_tokenizer(
    workspace: Path,
    train_files: Sequence[Path],
    vocab_size: int,
    sample_stories: int,
) -> Tokenizer:
    tokenizer_path = workspace / "tokenizer.json"
    tokenizer_meta_path = workspace / "tokenizer_meta.json"

    if tokenizer_path.exists() and tokenizer_meta_path.exists():
        meta = json.loads(
            tokenizer_meta_path.read_text(encoding="utf-8")
        )

        if int(meta.get("vocab_size", -1)) == vocab_size:
            print(f"Loading cached tokenizer: {tokenizer_path}", flush=True)
            return Tokenizer.from_file(str(tokenizer_path))

    print(
        f"Training {vocab_size}-token byte-level BPE on up to "
        f"{sample_stories:,} TinyStories...",
        flush=True,
    )

    tokenizer = Tokenizer(
        models.BPE(
            unk_token="<UNK>",
        )
    )

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(
        parquet_text_batches(
            train_files,
            batch_rows=4096,
            max_rows=sample_stories,
        ),
        trainer=trainer,
        length=sample_stories,
    )

    if tokenizer.get_vocab_size() > 65535:
        raise RuntimeError("tokenizer no longer fits uint16 storage")

    # Confirm critical symbols are atomic.
    for token in ("<PAD>", "<UNK>", "<BOS>", "<EOS>"):
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise RuntimeError(f"special token missing from tokenizer: {token}")

    tokenizer.save(str(tokenizer_path))

    tokenizer_meta_path.write_text(
        json.dumps(
            {
                "vocab_size": tokenizer.get_vocab_size(),
                "requested_vocab_size": vocab_size,
                "sample_stories": sample_stories,
                "special_tokens": SPECIAL_TOKENS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Tokenizer saved: {tokenizer_path}", flush=True)
    return tokenizer


def packed_file_valid(path: Path, token_count: int) -> bool:
    return (
        path.exists()
        and path.stat().st_size == token_count * np.dtype(np.uint16).itemsize
    )


def encode_and_pack_split(
    tokenizer: Tokenizer,
    parquet_files: Sequence[Path],
    output_path: Path,
    batch_rows: int,
    split_name: str,
) -> int:
    eos_id = tokenizer.token_to_id("<EOS>")
    if eos_id is None:
        raise RuntimeError("<EOS> token missing")

    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if temp_path.exists():
        temp_path.unlink()

    story_count = 0
    token_count = 0
    started = time.perf_counter()

    with temp_path.open("wb", buffering=8 * 1024 * 1024) as handle:
        for batch_index, texts in enumerate(
            parquet_text_batches(
                parquet_files,
                batch_rows=batch_rows,
                max_rows=None,
            ),
            start=1,
        ):
            encodings = tokenizer.encode_batch(texts)

            # Flatten one batch only. This is the one-time preprocessing path,
            # never part of the GPU training loop.
            flattened: List[int] = []

            approx_tokens = sum(
                len(encoding.ids) + 1
                for encoding in encodings
            )

            # Preallocation is not exposed by list, but this keeps the object
            # bounded to one Parquet batch and avoids corpus-sized Python data.
            if approx_tokens <= 0:
                continue

            for encoding in encodings:
                flattened.extend(encoding.ids)
                flattened.append(eos_id)

            array = np.asarray(
                flattened,
                dtype=np.uint16,
            )

            array.tofile(handle)

            story_count += len(encodings)
            token_count += int(array.size)

            if batch_index == 1 or batch_index % 100 == 0:
                elapsed = max(1e-9, time.perf_counter() - started)
                print(
                    f"  pack {split_name}: stories={story_count:,} "
                    f"tokens={token_count:,} "
                    f"rate={token_count / elapsed:,.0f} tok/s",
                    flush=True,
                )

    temp_path.replace(output_path)

    elapsed = max(1e-9, time.perf_counter() - started)

    print(
        f"Packed {split_name}: {story_count:,} stories, "
        f"{token_count:,} tokens, "
        f"{output_path.stat().st_size / 2**30:.2f} GiB, "
        f"{token_count / elapsed:,.0f} tok/s",
        flush=True,
    )

    return token_count


def prepare_binary_corpus(
    workspace: Path,
    tokenizer: Tokenizer,
    train_files: Sequence[Path],
    validation_files: Sequence[Path],
    pack_batch_rows: int,
) -> Dict[str, object]:
    train_bin = workspace / "train.bin"
    validation_bin = workspace / "validation.bin"
    meta_path = workspace / "corpus_meta.json"

    cached_meta: Optional[Dict[str, object]] = None

    if meta_path.exists():
        try:
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            cached_meta = None

    if cached_meta is not None:
        train_tokens = int(cached_meta.get("train_tokens", -1))
        validation_tokens = int(cached_meta.get("validation_tokens", -1))

        if (
            train_tokens > 0
            and validation_tokens > 0
            and packed_file_valid(train_bin, train_tokens)
            and packed_file_valid(validation_bin, validation_tokens)
        ):
            print("Loading cached packed TinyStories binaries.", flush=True)
            return cached_meta

    print("Packing all TinyStories text into local uint16 token streams...", flush=True)

    train_tokens = encode_and_pack_split(
        tokenizer=tokenizer,
        parquet_files=train_files,
        output_path=train_bin,
        batch_rows=pack_batch_rows,
        split_name="train",
    )

    validation_tokens = encode_and_pack_split(
        tokenizer=tokenizer,
        parquet_files=validation_files,
        output_path=validation_bin,
        batch_rows=pack_batch_rows,
        split_name="validation",
    )

    meta = {
        "dataset_repo": DATASET_REPO,
        "dtype": "uint16",
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "train_bin": str(train_bin),
        "validation_bin": str(validation_bin),
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "train_parquet_rows": count_parquet_rows(train_files),
        "validation_parquet_rows": count_parquet_rows(validation_files),
    }

    meta_path.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    return meta


# ---------------------------------------------------------------------------
# Packed token iteration with static GPU input addresses
# ---------------------------------------------------------------------------


class PackedTokenStream:
    """
    Three design goals:
      1. Every pass consumes every complete B*T token block exactly once.
      2. Disk access is mostly sequential inside large shuffled chunks.
      3. GPU input tensor addresses remain static for compiler/CUDA-graph use.
    """

    def __init__(
        self,
        path: Path,
        token_count: int,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        chunk_batches: int = 64,
    ) -> None:
        self.path = Path(path)
        self.token_count = int(token_count)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.device = device
        self.chunk_batches = int(chunk_batches)

        self.span = self.batch_size * self.seq_len
        self.steps_per_pass = (self.token_count - 1) // self.span

        if self.steps_per_pass <= 0:
            raise ValueError("token stream is smaller than one training batch")

        self.memmap = np.memmap(
            self.path,
            mode="r",
            dtype=np.uint16,
            shape=(self.token_count,),
        )

        # Two reusable pinned host buffers. Each holds one flat B*T+1 slice.
        self.cpu_buffers = [
            torch.empty(
                self.span + 1,
                dtype=torch.long,
                pin_memory=True,
            )
            for _ in range(2)
        ]

        self.cpu_arrays = [buffer.numpy() for buffer in self.cpu_buffers]
        # Reuse two CUDA events rather than creating one Python/CUDA event
        # object per batch.
        self.copy_events = [
            torch.cuda.Event(blocking=False),
            torch.cuda.Event(blocking=False),
        ]
        self.copy_event_valid = [False, False]

        # Static GPU addresses, valuable for torch.compile/reduce-overhead.
        self.x_gpu = torch.empty(
            (self.batch_size, self.seq_len),
            dtype=torch.long,
            device=self.device,
        )

        self.y_gpu = torch.empty_like(self.x_gpu)

    def _step_order(self, pass_index: int, seed: int) -> Iterator[int]:
        num_chunks = math.ceil(
            self.steps_per_pass / self.chunk_batches
        )

        chunks = list(range(num_chunks))
        rng = random.Random(seed + 1009 * pass_index)
        rng.shuffle(chunks)

        for chunk_index in chunks:
            first = chunk_index * self.chunk_batches
            last = min(
                first + self.chunk_batches,
                self.steps_per_pass,
            )

            # Sequential access inside each shuffled multi-megabyte chunk.
            for step_index in range(first, last):
                yield step_index

    def iter_pass(
        self,
        pass_index: int,
        seed: int,
        start_step_in_pass: int = 0,
    ) -> Iterator[Tuple[int, torch.Tensor, torch.Tensor]]:
        yielded = 0
        buffer_index = 0

        for logical_step in self._step_order(pass_index, seed):
            if yielded < start_step_in_pass:
                yielded += 1
                continue

            event = self.copy_events[buffer_index]
            if self.copy_event_valid[buffer_index]:
                event.synchronize()

            token_start = logical_step * self.span
            token_end = token_start + self.span + 1

            source = self.memmap[token_start:token_end]

            # Cast uint16 -> int64 directly into persistent pinned memory.
            np.copyto(
                self.cpu_arrays[buffer_index],
                source,
                casting="unsafe",
            )

            cpu_flat = self.cpu_buffers[buffer_index]
            x_cpu = cpu_flat[:-1].view(
                self.batch_size,
                self.seq_len,
            )
            y_cpu = cpu_flat[1:].view(
                self.batch_size,
                self.seq_len,
            )

            # Copies use the default stream. Because x_gpu/y_gpu are static,
            # the next copies queue behind previous compute automatically.
            self.x_gpu.copy_(x_cpu, non_blocking=True)
            self.y_gpu.copy_(y_cpu, non_blocking=True)

            event.record(torch.cuda.current_stream())
            self.copy_event_valid[buffer_index] = True

            current_step_in_pass = yielded
            yielded += 1
            buffer_index = 1 - buffer_index

            yield current_step_in_pass, self.x_gpu, self.y_gpu


# ---------------------------------------------------------------------------
# Attention / batch / compile autotuning
# ---------------------------------------------------------------------------


def benchmark_model_steps(
    model: UOMindLM,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    warmup: int = 2,
    measured: int = 4,
) -> Tuple[float, float]:
    """Forward+backward throughput and peak allocated GiB, no optimizer update."""

    device = torch.device("cuda")

    x = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    y = torch.randint_like(x, low=0, high=vocab_size)

    scaler = make_grad_scaler(enabled=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total = warmup + measured
    started = None

    for index in range(total):
        model.zero_grad(set_to_none=True)

        with amp_context(True):
            _, loss = model(x, y)

        scaler.scale(loss).backward()

        if index == warmup - 1:
            torch.cuda.synchronize()
            started = time.perf_counter()

    torch.cuda.synchronize()

    if started is None:
        raise RuntimeError("benchmark timing was not initialized")

    elapsed = max(1e-9, time.perf_counter() - started)
    throughput = measured * batch_size * seq_len / elapsed
    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    model.zero_grad(set_to_none=True)
    del x, y, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return throughput, peak_gib


def choose_attention_backend(
    base_config: ModelConfig,
    seed: int,
    requested: str,
) -> str:
    if requested in {"sdpa", "manual"}:
        return requested

    print("Benchmarking attention backends on this GPU...", flush=True)

    results: Dict[str, float] = {}

    for backend in ("sdpa", "manual"):
        config = ModelConfig(
            **{
                **asdict(base_config),
                "attention_backend": backend,
            }
        )

        try:
            set_seed(seed)
            model = UOMindLM(config).cuda()
            throughput, peak = benchmark_model_steps(
                model=model,
                batch_size=64,
                seq_len=config.max_seq_len,
                vocab_size=config.vocab_size,
                warmup=2,
                measured=3,
            )

            results[backend] = throughput

            print(
                f"  {backend:6s}: {throughput:,.0f} tok/s, "
                f"peak={peak:.2f} GiB",
                flush=True,
            )

        except Exception as exc:
            print(
                f"  {backend:6s}: unavailable ({type(exc).__name__}: {exc})",
                flush=True,
            )

        finally:
            try:
                del model
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    if not results:
        raise RuntimeError("neither SDPA nor manual attention benchmark succeeded")

    winner = max(results, key=results.get)
    print(f"Selected attention backend: {winner}", flush=True)
    return winner


def parse_batch_candidates(text: str) -> List[int]:
    values = []

    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("batch candidates must be positive")
        values.append(value)

    if not values:
        raise ValueError("no batch candidates supplied")

    return sorted(set(values))


def choose_batch_size(
    config: ModelConfig,
    seed: int,
    requested: str,
    candidates: Sequence[int],
    max_memory_fraction: float,
) -> int:
    if requested != "auto":
        batch = int(requested)
        if batch <= 0:
            raise ValueError("batch size must be positive")
        return batch

    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    allowed_peak = total_gib * max_memory_fraction

    print(
        f"Autotuning batch size; allowed benchmark peak <= {allowed_peak:.2f} GiB...",
        flush=True,
    )

    set_seed(seed)
    model = UOMindLM(config).cuda()

    results: Dict[int, Tuple[float, float]] = {}

    for batch in candidates:
        try:
            throughput, peak = benchmark_model_steps(
                model=model,
                batch_size=batch,
                seq_len=config.max_seq_len,
                vocab_size=config.vocab_size,
                warmup=1,
                measured=3,
            )

            safe = peak <= allowed_peak

            print(
                f"  batch={batch:3d}: {throughput:,.0f} tok/s, "
                f"peak={peak:.2f} GiB, {'safe' if safe else 'too close to limit'}",
                flush=True,
            )

            if safe:
                results[batch] = (throughput, peak)

        except torch.OutOfMemoryError:
            print(f"  batch={batch:3d}: OOM", flush=True)
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"  batch={batch:3d}: OOM", flush=True)
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if not results:
        raise RuntimeError(
            "no safe batch size was found; provide a smaller --batch-candidates list"
        )

    winner = max(
        results,
        key=lambda batch: results[batch][0],
    )

    print(
        f"Selected batch size: {winner} "
        f"({results[winner][0]:,.0f} benchmark tok/s)",
        flush=True,
    )

    return winner


def benchmark_callable_training(
    callable_model,
    raw_model: UOMindLM,
    x: torch.Tensor,
    y: torch.Tensor,
    warmup: int,
    measured: int,
) -> float:
    scaler = make_grad_scaler(enabled=True)

    for _ in range(warmup):
        raw_model.zero_grad(set_to_none=True)
        with amp_context(True):
            _, loss = callable_model(x, y)
        scaler.scale(loss).backward()

    torch.cuda.synchronize()
    started = time.perf_counter()

    for _ in range(measured):
        raw_model.zero_grad(set_to_none=True)
        with amp_context(True):
            _, loss = callable_model(x, y)
        scaler.scale(loss).backward()

    torch.cuda.synchronize()
    elapsed = max(1e-9, time.perf_counter() - started)

    raw_model.zero_grad(set_to_none=True)
    del scaler

    return measured * x.numel() / elapsed


def maybe_compile_model(
    raw_model: UOMindLM,
    batch_size: int,
    config: ModelConfig,
    enable_compile: bool,
    compile_mode: str,
) -> Tuple[object, bool]:
    if not enable_compile:
        print("torch.compile disabled by argument.", flush=True)
        return raw_model, False

    x = torch.randint(
        0,
        config.vocab_size,
        (batch_size, config.max_seq_len),
        dtype=torch.long,
        device="cuda",
    )

    y = torch.randint_like(
        x,
        low=0,
        high=config.vocab_size,
    )

    print("Benchmarking eager training path before compile...", flush=True)

    eager_tps = benchmark_callable_training(
        raw_model,
        raw_model,
        x,
        y,
        warmup=2,
        measured=4,
    )

    print(f"  eager: {eager_tps:,.0f} tok/s", flush=True)
    print(
        f"Attempting torch.compile(mode={compile_mode!r}); first call may be slow...",
        flush=True,
    )

    try:
        compiled = torch.compile(
            raw_model,
            mode=compile_mode,
            dynamic=False,
        )

        compiled_tps = benchmark_callable_training(
            compiled,
            raw_model,
            x,
            y,
            warmup=3,
            measured=5,
        )

        print(f"  compiled: {compiled_tps:,.0f} tok/s", flush=True)

        if compiled_tps >= eager_tps * 1.02:
            print(
                f"Using compiled model ({compiled_tps / eager_tps:.2f}x eager).",
                flush=True,
            )
            del x, y
            raw_model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return compiled, True

        print(
            "Compile did not improve steady-state throughput enough; using eager.",
            flush=True,
        )

    except Exception as exc:
        print(
            f"torch.compile unavailable/failed; using eager: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

    del x, y
    raw_model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return raw_model, False


# ---------------------------------------------------------------------------
# Optimizer / LR / validation / checkpoints
# ---------------------------------------------------------------------------


def build_optimizer(
    model: UOMindLM,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    groups = model.parameter_groups(weight_decay)

    try:
        optimizer = torch.optim.AdamW(
            groups,
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=True,
        )
        print("Optimizer: fused AdamW", flush=True)
        return optimizer
    except Exception as exc:
        print(
            f"Fused AdamW unavailable ({type(exc).__name__}); using foreach AdamW.",
            flush=True,
        )

        return torch.optim.AdamW(
            groups,
            lr=lr,
            betas=(0.9, 0.95),
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
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if total_steps <= warmup_steps:
        return min_lr

    progress = (
        (step - warmup_steps)
        / max(1, total_steps - warmup_steps)
    )

    progress = min(1.0, max(0.0, progress))

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


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
        return {k: recursive_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [recursive_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(recursive_to_cpu(v) for v in value)
    return value


def optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(
    path: Path,
    raw_model: UOMindLM,
    optimizer: torch.optim.Optimizer,
    scaler,
    state: Dict[str, object],
    model_config: ModelConfig,
    training_config: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")

    payload = {
        "model_state_dict": recursive_to_cpu(raw_model.state_dict()),
        "optimizer_state_dict": recursive_to_cpu(optimizer.state_dict()),
        "scaler_state_dict": scaler.state_dict(),
        "state": state,
        "model_config": asdict(model_config),
        "training_config": training_config,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }

    torch.save(payload, temp)
    temp.replace(path)


def restore_rng(payload: Dict[str, object]) -> None:
    rng = payload.get("rng")
    if not rng:
        return

    try:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception:
        pass


@torch.inference_mode()
def evaluate_validation(
    raw_model: UOMindLM,
    validation_stream: PackedTokenStream,
    max_batches: int,
    amp_enabled: bool,
) -> Dict[str, float]:
    # No dropout exists, but use eval() for semantic correctness.
    was_training = raw_model.training
    raw_model.eval()

    loss_samples: List[torch.Tensor] = []
    token_sum = 0
    started = time.perf_counter()

    iterator = validation_stream.iter_pass(
        pass_index=0,
        seed=0,
        start_step_in_pass=0,
    )

    used = 0

    for _, x, y in iterator:
        if used >= max_batches:
            break

        with amp_context(amp_enabled):
            _, loss = raw_model(x, y)

        loss_samples.append(loss.detach())
        token_sum += x.numel()
        used += 1

    torch.cuda.synchronize()
    elapsed = max(1e-9, time.perf_counter() - started)

    if was_training:
        raw_model.train()

    mean_loss = (
        float(torch.stack(loss_samples).mean().item())
        if loss_samples
        else float("nan")
    )

    return {
        "loss": mean_loss,
        "ppl": math.exp(min(mean_loss, 20.0)),
        "tokens": float(token_sum),
        "tok_per_sec": token_sum / elapsed,
        "batches": float(used),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    args: argparse.Namespace,
    workspace: Path,
    tokenizer: Tokenizer,
    corpus_meta: Dict[str, object],
) -> Dict[str, object]:
    device = torch.device("cuda")

    checkpoint_dir = workspace / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    final_path = checkpoint_dir / "final.pt"

    resume_payload = None

    if args.resume == "auto" and latest_path.exists():
        print(f"Loading resumable checkpoint: {latest_path}", flush=True)
        resume_payload = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=False,
        )
    elif args.resume not in {"auto", "none"}:
        resume_path = Path(args.resume)
        print(f"Loading checkpoint: {resume_path}", flush=True)
        resume_payload = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )

    if resume_payload is not None:
        saved_training = resume_payload["training_config"]
        attention_backend = str(saved_training["attention_backend"])
        batch_size = int(saved_training["batch_size"])
        compile_requested = bool(saved_training.get("compile_requested", True))

        print(
            f"Resume locks backend={attention_backend}, batch={batch_size}",
            flush=True,
        )

    else:
        base_config = ModelConfig(
            vocab_size=tokenizer.get_vocab_size(),
            attention_backend="sdpa",
        )

        attention_backend = choose_attention_backend(
            base_config=base_config,
            seed=args.seed + 11,
            requested=args.attention,
        )

        config_for_batch = ModelConfig(
            vocab_size=tokenizer.get_vocab_size(),
            attention_backend=attention_backend,
        )

        batch_size = choose_batch_size(
            config=config_for_batch,
            seed=args.seed + 22,
            requested=args.batch_size,
            candidates=parse_batch_candidates(args.batch_candidates),
            max_memory_fraction=args.max_memory_fraction,
        )

        compile_requested = not args.no_compile

    model_config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        attention_backend=attention_backend,
    )

    set_seed(args.seed)

    raw_model = UOMindLM(model_config).to(device)

    print()
    print("Model", flush=True)
    print("-----", flush=True)
    print(f"parameters: {raw_model.num_parameters():,}", flush=True)
    print(f"context:    {model_config.max_seq_len}", flush=True)
    print(f"layers:     {model_config.n_layers}", flush=True)
    print(f"width:      {model_config.d_model}", flush=True)
    print(f"heads:      {model_config.n_heads}", flush=True)
    print(f"vocab:      {model_config.vocab_size}", flush=True)
    print(f"attention:  {model_config.attention_backend}", flush=True)
    print(f"batch:      {batch_size}", flush=True)
    print(flush=True)

    optimizer = build_optimizer(
        raw_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = make_grad_scaler(enabled=True)

    state: Dict[str, object] = {
        "pass_index": 0,
        "step_in_pass": 0,
        "global_step": 0,
        "tokens_seen": 0,
        "best_val_loss": float("inf"),
    }

    if resume_payload is not None:
        raw_model.load_state_dict(
            resume_payload["model_state_dict"],
            strict=True,
        )

        optimizer.load_state_dict(
            resume_payload["optimizer_state_dict"]
        )

        optimizer_to_device(optimizer, device)
        scaler.load_state_dict(
            resume_payload.get("scaler_state_dict", {})
        )

        state.update(resume_payload["state"])
        restore_rng(resume_payload)

        print(
            f"Resuming at pass={int(state['pass_index']) + 1}, "
            f"step_in_pass={state['step_in_pass']}, "
            f"global_step={state['global_step']:,}",
            flush=True,
        )

    train_tokens = int(corpus_meta["train_tokens"])
    validation_tokens = int(corpus_meta["validation_tokens"])

    train_stream = PackedTokenStream(
        path=workspace / "train.bin",
        token_count=train_tokens,
        batch_size=batch_size,
        seq_len=model_config.max_seq_len,
        device=device,
        chunk_batches=args.chunk_batches,
    )

    val_batch_size = min(
        batch_size,
        int(args.val_batch_size),
    )

    validation_stream = PackedTokenStream(
        path=workspace / "validation.bin",
        token_count=validation_tokens,
        batch_size=val_batch_size,
        seq_len=model_config.max_seq_len,
        device=device,
        chunk_batches=64,
    )

    steps_per_pass = train_stream.steps_per_pass
    total_steps = steps_per_pass * args.passes
    tokens_per_step = batch_size * model_config.max_seq_len

    training_config = {
        "passes": args.passes,
        "batch_size": batch_size,
        "seq_len": model_config.max_seq_len,
        "attention_backend": attention_backend,
        "compile_requested": compile_requested,
        "compile_mode": args.compile_mode,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_fraction": args.warmup_fraction,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "steps_per_pass": steps_per_pass,
        "total_steps": total_steps,
        "tokens_per_step": tokens_per_step,
        "train_tokens": train_tokens,
    }

    print()
    print("Training schedule", flush=True)
    print("-----------------", flush=True)
    print(f"train tokens:     {train_tokens:,}", flush=True)
    print(f"tokens/step:      {tokens_per_step:,}", flush=True)
    print(f"steps/pass:       {steps_per_pass:,}", flush=True)
    print(f"passes:           {args.passes}", flush=True)
    print(f"planned steps:    {total_steps:,}", flush=True)
    print(
        f"planned exposure: {steps_per_pass * tokens_per_step * args.passes:,} tokens",
        flush=True,
    )
    print(flush=True)

    # Compile only AFTER resume state is loaded so no graph is built around
    # stale initialization. Benchmark keeps compile only if it wins.
    train_model, compile_used = maybe_compile_model(
        raw_model=raw_model,
        batch_size=batch_size,
        config=model_config,
        enable_compile=compile_requested,
        compile_mode=args.compile_mode,
    )

    training_config["compile_used"] = compile_used

    global_step = int(state["global_step"])
    tokens_seen = int(state["tokens_seen"])
    best_val_loss = float(state["best_val_loss"])
    start_pass = int(state["pass_index"])
    start_step_in_pass = int(state["step_in_pass"])

    raw_model.train()

    # Keep detached scalar loss tensors and reduce only at logging intervals.
    # This avoids a separate GPU reduction/add kernel and, critically, avoids
    # any CUDA->CPU synchronization inside the per-step hot path.
    loss_samples: List[torch.Tensor] = []

    run_started = time.perf_counter()
    log_started = time.perf_counter()
    log_tokens_start = tokens_seen

    for pass_index in range(start_pass, args.passes):
        pass_start_step = (
            start_step_in_pass
            if pass_index == start_pass
            else 0
        )

        print()
        print(
            f"========== PASS {pass_index + 1}/{args.passes} "
            f"(starting step {pass_start_step:,}/{steps_per_pass:,}) ==========",
            flush=True,
        )

        iterator = train_stream.iter_pass(
            pass_index=pass_index,
            seed=args.seed + 5000,
            start_step_in_pass=pass_start_step,
        )

        for step_in_pass, x, y in iterator:
            lr = learning_rate_at_step(
                step=global_step,
                total_steps=total_steps,
                max_lr=args.lr,
                min_lr=args.min_lr,
                warmup_fraction=args.warmup_fraction,
            )

            set_optimizer_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)

            with amp_context(True):
                _, loss = train_model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(),
                max_norm=args.grad_clip,
                foreach=True,
            )

            scaler.step(optimizer)
            scaler.update()

            loss_samples.append(loss.detach())

            global_step += 1
            tokens_seen += tokens_per_step

            next_step_in_pass = step_in_pass + 1

            if global_step % args.log_every == 0:
                torch.cuda.synchronize()
                now = time.perf_counter()
                interval = max(1e-9, now - log_started)
                interval_tokens = tokens_seen - log_tokens_start
                if loss_samples:
                    mean_loss = float(torch.stack(loss_samples).mean().item())
                else:
                    mean_loss = float("nan")

                if not math.isfinite(mean_loss):
                    raise FloatingPointError(
                        f"non-finite mean loss detected at global step {global_step}"
                    )

                tps = interval_tokens / interval
                progress = 100.0 * next_step_in_pass / steps_per_pass

                print(
                    f"pass={pass_index + 1}/{args.passes} "
                    f"step={next_step_in_pass:,}/{steps_per_pass:,} "
                    f"({progress:5.1f}%) "
                    f"global={global_step:,} "
                    f"loss={mean_loss:.4f} "
                    f"ppl={math.exp(min(mean_loss, 20.0)):.2f} "
                    f"lr={lr:.2e} "
                    f"tok/s={tps:,.0f} "
                    f"seen={tokens_seen:,}",
                    flush=True,
                )

                loss_samples.clear()
                log_started = now
                log_tokens_start = tokens_seen

            if (
                args.checkpoint_every > 0
                and global_step % args.checkpoint_every == 0
            ):
                checkpoint_state = {
                    "pass_index": pass_index,
                    "step_in_pass": next_step_in_pass,
                    "global_step": global_step,
                    "tokens_seen": tokens_seen,
                    "best_val_loss": best_val_loss,
                }

                torch.cuda.synchronize()
                print(
                    f"Saving resumable checkpoint at global step {global_step:,}...",
                    flush=True,
                )

                save_checkpoint(
                    latest_path,
                    raw_model,
                    optimizer,
                    scaler,
                    checkpoint_state,
                    model_config,
                    training_config,
                )

                if args.backup_dir:
                    backup_dir = Path(args.backup_dir)
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(
                        latest_path,
                        backup_dir / latest_path.name,
                    )

        # Completed the pass.
        torch.cuda.synchronize()

        print(
            f"Validating after pass {pass_index + 1}...",
            flush=True,
        )

        metrics = evaluate_validation(
            raw_model=raw_model,
            validation_stream=validation_stream,
            max_batches=args.val_batches,
            amp_enabled=True,
        )

        geometry = raw_model.geometry_gain_summary()

        print(
            f"PASS {pass_index + 1} VALIDATION: "
            f"loss={metrics['loss']:.4f} "
            f"ppl={metrics['ppl']:.3f} "
            f"tokens={int(metrics['tokens']):,} "
            f"eval_tok/s={metrics['tok_per_sec']:,.0f} "
            f"geometry_gain_mean={geometry['mean']:.4f}",
            flush=True,
        )

        # Next checkpoint starts at the next pass, step zero.
        next_state = {
            "pass_index": pass_index + 1,
            "step_in_pass": 0,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "best_val_loss": min(best_val_loss, metrics["loss"]),
        }

        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            next_state["best_val_loss"] = best_val_loss

            print(
                f"New best validation loss {best_val_loss:.4f}; saving best.pt",
                flush=True,
            )

            save_checkpoint(
                best_path,
                raw_model,
                optimizer,
                scaler,
                next_state,
                model_config,
                training_config,
            )

        save_checkpoint(
            latest_path,
            raw_model,
            optimizer,
            scaler,
            next_state,
            model_config,
            training_config,
        )

        # Optional durable copy to a user-provided mounted location.
        if args.backup_dir:
            backup_dir = Path(args.backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)

            for artifact in (latest_path, best_path):
                if artifact.exists():
                    destination = backup_dir / artifact.name
                    shutil.copy2(artifact, destination)

            shutil.copy2(workspace / "tokenizer.json", backup_dir / "tokenizer.json")
            shutil.copy2(workspace / "corpus_meta.json", backup_dir / "corpus_meta.json")

            print(f"Backup refreshed: {backup_dir}", flush=True)

        start_step_in_pass = 0

    final_state = {
        "pass_index": args.passes,
        "step_in_pass": 0,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
    }

    save_checkpoint(
        final_path,
        raw_model,
        optimizer,
        scaler,
        final_state,
        model_config,
        training_config,
    )

    torch.cuda.synchronize()
    elapsed = max(1e-9, time.perf_counter() - run_started)

    summary = {
        "model_config": asdict(model_config),
        "training_config": training_config,
        "final_state": final_state,
        "geometry": raw_model.geometry_gain_summary(),
        "elapsed_seconds_this_run": elapsed,
        "average_training_token_rate_this_run": (
            max(0, tokens_seen - int(state["tokens_seen"])) / elapsed
        ),
        "gpu": gpu_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }

    summary_path = workspace / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("Training complete", flush=True)
    print("-----------------", flush=True)
    print(f"passes:          {args.passes}", flush=True)
    print(f"global steps:    {global_step:,}", flush=True)
    print(f"tokens seen:     {tokens_seen:,}", flush=True)
    print(f"best val loss:   {best_val_loss:.4f}", flush=True)
    print(f"final checkpoint:{final_path}", flush=True)
    print(f"best checkpoint: {best_path}", flush=True)
    print(f"summary:         {summary_path}", flush=True)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download, prepare, autotune, and pretrain the 128-context "
            "UO-Mind fixed-geometry model on TinyStories."
        )
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=DEFAULT_WORKSPACE,
    )

    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="Complete passes through the packed TinyStories token stream.",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--tokenizer-stories",
        type=int,
        default=250000,
        help="Number of stories used to learn the 4K BPE merges.",
    )

    parser.add_argument(
        "--pack-batch-rows",
        type=int,
        default=2048,
        help="Stories tokenized per one-time preprocessing batch.",
    )

    parser.add_argument(
        "--attention",
        choices=("auto", "sdpa", "manual"),
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help="Integer batch size or 'auto'.",
    )

    parser.add_argument(
        "--batch-candidates",
        type=str,
        default="96,128,160,192,224,256",
    )

    parser.add_argument(
        "--max-memory-fraction",
        type=float,
        default=0.82,
        help="Maximum eager benchmark peak accepted by batch autotuning.",
    )

    parser.add_argument(
        "--chunk-batches",
        type=int,
        default=64,
        help="Sequential training batches per shuffled disk chunk.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=6e-4,
    )

    parser.add_argument(
        "--min-lr",
        type=float,
        default=6e-5,
    )

    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.01,
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
        "--log-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5000,
        help="0 disables mid-pass checkpoints. Default is about hourly-scale on a T4.",
    )

    parser.add_argument(
        "--val-batches",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="'auto', 'none', or a checkpoint path.",
    )

    parser.add_argument(
        "--backup-dir",
        type=str,
        default="",
        help=(
            "Optional already-mounted persistent directory, e.g. "
            "/content/drive/MyDrive/uomind. Copied only at pass boundaries."
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

    if args.passes <= 0:
        raise ValueError("--passes must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. In Colab choose Runtime > Change runtime type > T4 GPU."
        )

    # This model is dominated by FP16 matmul; avoid deterministic slow paths.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    set_seed(args.seed)
    print_environment()

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    train_files, validation_files = download_tinystories(workspace)

    tokenizer = train_or_load_tokenizer(
        workspace=workspace,
        train_files=train_files,
        vocab_size=args.vocab_size,
        sample_stories=args.tokenizer_stories,
    )

    if tokenizer.get_vocab_size() != args.vocab_size:
        print(
            f"Tokenizer learned {tokenizer.get_vocab_size()} tokens "
            f"(requested {args.vocab_size}).",
            flush=True,
        )

    corpus_meta = prepare_binary_corpus(
        workspace=workspace,
        tokenizer=tokenizer,
        train_files=train_files,
        validation_files=validation_files,
        pack_batch_rows=args.pack_batch_rows,
    )

    print()
    print("Packed corpus", flush=True)
    print("-------------", flush=True)
    print(f"train tokens:      {int(corpus_meta['train_tokens']):,}", flush=True)
    print(f"validation tokens: {int(corpus_meta['validation_tokens']):,}", flush=True)
    print(flush=True)

    train(
        args=args,
        workspace=workspace,
        tokenizer=tokenizer,
        corpus_meta=corpus_meta,
    )


if __name__ == "__main__":
    main()
