#!/usr/bin/env python3
"""
uomind_colab_inference_test_v3.py

20-query inference/evaluation harness for the final 128-token UO-Mind model.

Default checkpoint search order:
    /content/uomind_tinystories/final_sft/checkpoints/best_balanced.pt
    /content/uomind_tinystories/final_sft/checkpoints/best_sft.pt
    /content/uomind_tinystories/final_sft/checkpoints/final.pt
    /content/uomind_tinystories/npc_behavior/checkpoints/best_balanced.pt

Default tokenizer:
    /content/uomind_tinystories/tokenizer.json

The default suite contains:
    12 REAL held-out conversations reconstructed from /content/SFT
     4 harder hand-authored natural-language generalization tests
     4 structured behavior tests using the earlier ACTION/TONE/SAY protocol

Each result prints:
    - test/category
    - evaluation goal
    - rendered prompt
    - generated response
    - action/tone parsing for structured tests
    - generated token count
    - stop reason
    - latency
    - tokens/sec
    - mean selected-token probability
    - any unexpected control-token leakage

Results are also written to:
    /content/uomind_tinystories/inference_test/
        inference_results.json
        inference_results.txt

Run:
    !python uomind_colab_inference_test_v3.py

Sample instead of greedy:
    !python uomind_colab_inference_test_v3.py --temperature 0.7 --top-p 0.9 --seed 123

Longer generations:
    !python uomind_colab_inference_test_v3.py --max-new-tokens 48

Run only natural or structured tests:
    !python uomind_colab_inference_test_v3.py --suite natural
    !python uomind_colab_inference_test_v3.py --suite structured

Interactive mode after the scripted suite:
    !python uomind_colab_inference_test_v3.py --interactive
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tokenizers import Tokenizer
except ImportError:
    print("Installing tokenizers...", flush=True)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tokenizers>=0.21"])
    from tokenizers import Tokenizer


# ---------------------------------------------------------------------------
# Paths / controls
# ---------------------------------------------------------------------------

DEFAULT_WORKSPACE = Path("/content/uomind_tinystories")
DEFAULT_OUTPUT = DEFAULT_WORKSPACE / "inference_test"
DEFAULT_SFT_DIR = Path("/content/SFT")

ACTION_TOKENS: Dict[str, str] = {
    "ACCEPT": "<R00>",
    "COUNTER": "<R01>",
    "REFUSE": "<R02>",
    "CALL_GUARDS": "<R03>",
    "ADMIT": "<R04>",
    "FLEE": "<R05>",
    "DEFY": "<R06>",
    "BELIEVE": "<R07>",
    "DOUBT": "<R08>",
    "ACCUSE": "<R09>",
    "DEFLECT": "<R10>",
    "RECALL": "<R11>",
    "HELP": "<R12>",
    "DEFER": "<R13>",
    "TREAT": "<R14>",
    "CHARGE": "<R15>",
    "SHARE": "<R16>",
    "HINT": "<R17>",
    "REPORT": "<R18>",
    "IGNORE": "<R19>",
    "OFFER_QUEST": "<R20>",
    "COMPLETE_QUEST": "<R21>",
    "SERVE": "<R22>",
    "ASK_LEAVE": "<R23>",
    "TAKE_BRIBE": "<R24>",
    "PAY_REWARD": "<R25>",
}

TONE_TOKENS: Dict[str, str] = {
    "GRUFF": "<R26>",
    "WARM": "<R27>",
    "WARY": "<R28>",
    "ANGRY": "<R29>",
    "FORMAL": "<R30>",
    "AFRAID": "<R31>",
}

TOKEN_TO_ACTION = {token: name for name, token in ACTION_TOKENS.items()}
TOKEN_TO_TONE = {token: name for name, token in TONE_TOKENS.items()}


# ---------------------------------------------------------------------------
# Exact model architecture
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
            raise ValueError("one geometry scale is required per head")
        if self.max_seq_len != 128:
            raise ValueError("this inference harness expects context 128")

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
        x_norm = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return x_norm.to(dtype) * self.weight


class GeometryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.backend = config.attention_backend
        self.gain_max = float(config.geometry_gain_max)

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        scales = torch.tensor(config.geometry_scales, dtype=torch.float32)
        positions = torch.arange(config.max_seq_len, dtype=torch.float32)
        distance = torch.abs(positions[:, None] - positions[None, :])
        base_geometry = (-distance.unsqueeze(0) / scales[:, None, None]).to(torch.float16)
        self.register_buffer("base_geometry", base_geometry.unsqueeze(0), persistent=True)

        causal = torch.zeros(
            config.max_seq_len,
            config.max_seq_len,
            dtype=torch.float16,
        )
        causal = causal.masked_fill(
            torch.triu(torch.ones_like(causal, dtype=torch.bool), diagonal=1),
            float("-inf"),
        )
        self.register_buffer(
            "causal_bias",
            causal.unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        init_ratio = config.geometry_gain_init / config.geometry_gain_max
        raw_init = math.log(init_ratio / (1.0 - init_ratio))
        self.raw_gain = nn.Parameter(
            torch.full((config.n_heads,), raw_init, dtype=torch.float32)
        )

    def gains(self) -> torch.Tensor:
        return self.gain_max * torch.sigmoid(self.raw_gain)

    def _geometry_bias(self, dtype: torch.dtype) -> torch.Tensor:
        gains = self.gains().to(dtype=dtype).view(1, self.n_heads, 1, 1)
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
            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores * (1.0 / math.sqrt(self.head_dim))
            scores = scores + bias
            weights = torch.softmax(scores, dim=-1)
            y = torch.matmul(weights, v)

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [UOMindBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        seen = set()
        total = 0
        for parameter in self.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            total += parameter.numel()
        return total

    def geometry_gain_summary(self) -> Dict[str, float]:
        gains = torch.stack([block.attn.gains() for block in self.blocks], dim=0)
        return {
            "mean": float(gains.mean().detach().cpu().item()),
            "min": float(gains.min().detach().cpu().item()),
            "max": float(gains.max().detach().cpu().item()),
        }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@dataclass
class NaturalCase:
    number: int
    category: str
    goal: str
    messages: List[Dict[str, str]]
    reference: Optional[str] = None
    source: str = "challenge"
    cached_row: Optional[int] = None
    exact_prefix_ids: Optional[List[int]] = None


@dataclass
class RebuiltValidationExample:
    row: int
    category: str
    category_id: int
    source: str
    messages: List[Dict[str, str]]
    reference: str
    input_ids: np.ndarray
    labels: np.ndarray
    context_truncated: bool
    response_truncated: bool


@dataclass
class StructuredCase:
    number: int
    category: str
    goal: str
    prompt: str
    expected_actions: Tuple[str, ...]


def _normalize_text(text: str) -> str:
    return " ".join(str(text).replace("\\r\\n", "\\n").replace("\\r", "\\n").split()).strip()


def _conversation_fingerprint(messages: Sequence[Dict[str, str]]) -> str:
    normalized = [
        {"role": m["role"], "content": _normalize_text(m["content"])}
        for m in messages
    ]
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _category_name(path: Path, root: Path) -> str:
    return str(path.relative_to(root).with_suffix("")).replace(os.sep, "/")


def _parse_sft_conversations(sft_dir: Path) -> List[Dict[str, object]]:
    """Parse/deduplicate the same OpenAI-style JSONL records used by SFT."""
    if not sft_dir.exists():
        return []

    files = sorted(path for path in sft_dir.rglob("*.jsonl") if path.is_file())
    seen = set()
    conversations: List[Dict[str, object]] = []

    for path in files:
        category = _category_name(path, sft_dir)
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue

                messages = record.get("messages")
                if not isinstance(messages, list):
                    continue

                clean: List[Dict[str, str]] = []
                valid = True
                assistant_count = 0
                for message in messages:
                    if not isinstance(message, dict):
                        valid = False
                        break
                    role = message.get("role")
                    content = message.get("content")
                    if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                        valid = False
                        break
                    content = _normalize_text(content)
                    if not content:
                        valid = False
                        break
                    clean.append({"role": role, "content": content})
                    assistant_count += int(role == "assistant")

                if not valid or assistant_count == 0:
                    continue

                fingerprint = _conversation_fingerprint(clean)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                conversations.append(
                    {
                        "category": category,
                        "file": str(path),
                        "line": line_no,
                        "messages": clean,
                        "fingerprint": fingerprint,
                    }
                )

    return conversations


def _heldout_sft_conversations(
    sft_dir: Path,
    val_fraction: float,
    split_seed: int,
) -> List[Dict[str, object]]:
    """Reconstruct the exact category-stratified split used by the SFT trainer."""
    conversations = _parse_sft_conversations(sft_dir)
    by_category: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for conversation in conversations:
        by_category[str(conversation["category"])].append(conversation)

    heldout: List[Dict[str, object]] = []
    for category in sorted(by_category):
        items = list(by_category[category])
        items.sort(
            key=lambda item: _stable_hash_int(
                f"{split_seed}|{item['fingerprint']}"
            )
        )
        if val_fraction <= 0.0:
            val_count = 0
        elif len(items) >= 5:
            val_count = max(1, int(round(len(items) * val_fraction)))
            val_count = min(val_count, len(items) - 1)
        else:
            val_count = 0
        heldout.extend(items[:val_count])

    return heldout


def _target_turn(conversation: Dict[str, object]) -> Optional[Tuple[List[Dict[str, str]], str]]:
    """Use the last assistant turn so multi-turn history is genuinely exercised."""
    messages = list(conversation["messages"])
    assistant_positions = [
        i for i, message in enumerate(messages)
        if message["role"] == "assistant"
    ]
    if not assistant_positions:
        return None
    index = assistant_positions[-1]
    return messages[:index], messages[index]["content"]


def load_heldout_cases(
    sft_dir: Path,
    count: int,
    val_fraction: float,
    split_seed: int,
) -> List[NaturalCase]:
    heldout = _heldout_sft_conversations(
        sft_dir=sft_dir,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )
    if not heldout:
        return []

    by_category: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in heldout:
        by_category[str(item["category"])].append(item)

    categories = sorted(by_category)
    # Spread selections across the full category list instead of only taking
    # early-numbered domains.
    if count >= len(categories):
        selected_categories = categories
    else:
        positions = np.linspace(0, len(categories) - 1, count)
        selected_categories = []
        used = set()
        for position in positions:
            idx = int(round(float(position)))
            while idx in used and idx + 1 < len(categories):
                idx += 1
            if idx in used:
                idx = next(i for i in range(len(categories)) if i not in used)
            used.add(idx)
            selected_categories.append(categories[idx])

    cases: List[NaturalCase] = []
    for number, category in enumerate(selected_categories[:count], start=1):
        items = sorted(
            by_category[category],
            key=lambda item: _stable_hash_int(str(item["fingerprint"])),
        )
        conversation = items[0]
        target = _target_turn(conversation)
        if target is None:
            continue
        preceding, reference = target
        cases.append(
            NaturalCase(
                number=number,
                category=f"heldout/{category}",
                goal="Autoregressively answer a real held-out SFT conversation; compare against the curated reference.",
                messages=preceding,
                reference=reference,
                source=f"{Path(str(conversation['file'])).name}:{conversation['line']}",
            )
        )

    return cases


def challenge_cases(start_number: int = 13) -> List[NaturalCase]:
    """Four deliberately harder prompts that are not copied from the SFT files."""
    return [
        NaturalCase(
            start_number,
            "challenge/identity correction",
            "Keep the supplied identity and explicitly reject a false player assumption.",
            [
                {"role": "system", "content": "Your name is Marta. You are a baker from Britain. You are warm but proud of your craft."},
                {"role": "user", "content": "Are you Jorah the fisherman from Vesper?"},
            ],
        ),
        NaturalCase(
            start_number + 1,
            "challenge/practical weather",
            "Use the observed weather to give one practical piece of farm advice.",
            [
                {"role": "system", "content": "Your name is Edric. You are an experienced farmer near Minoc."},
                {"role": "user", "content": "Black clouds are rolling in and the wind has turned cold. Should I leave the cut hay outside?"},
            ],
        ),
        NaturalCase(
            start_number + 2,
            "challenge/gossip uncertainty",
            "Do not turn an unverified accusation into a fact.",
            [
                {"role": "system", "content": "Your name is Jorah. You are a discreet scribe in Britain and dislike spreading dangerous rumors without evidence."},
                {"role": "user", "content": "Someone says the mayor steals from the treasury. Is that true?"},
            ],
        ),
        NaturalCase(
            start_number + 3,
            "challenge/multi-turn memory",
            "Recall Yew from the earlier player turn while preserving the brewer persona.",
            [
                {"role": "system", "content": "Your name is Elara. You are a friendly brewer from Cove."},
                {"role": "user", "content": "I walked here all the way from Yew."},
                {"role": "assistant", "content": "Yew is a long road from Cove. Thou must be thirsty after such a journey."},
                {"role": "user", "content": "Where did I say I came from?"},
            ],
            reference="Thou saidst thou came from Yew.",
        ),
    ]


def structured_cases() -> List[StructuredCase]:
    """Short, rule-exact prompts drawn directly from the behavior generator logic."""
    return [
        StructuredCase(
            17,
            "structured/commerce",
            "offer=60 is far below the effective floor, so the learned rule should REFUSE.",
            '<BOS> <NPC> Brann <LOCATION> Britain <MOOD> gruff <REL> 0 '
            '<STATE> task=commerce;role=blacksmith;voice=plain;item=sword;ask=105;floor=85;offer=60;greed=1 '
            '<MEMORY> insult=0;last_offer=55 <PLAYER> "Will 60 gold buy the sword?" <ACTION>',
            ("REFUSE",),
        ),
        StructuredCase(
            18,
            "structured/gatekeeping",
            "wanted=1 deterministically maps to CALL_GUARDS in behavior training.",
            '<BOS> <NPC> Gareth <LOCATION> Trinsic <MOOD> stern <REL> -1 '
            '<STATE> task=gatekeeping;role=gate guard;voice=formal;wanted=1;pass=0;curfew=1;ally=0;corrupt=0;bribe=0 '
            '<MEMORY> warning=none <PLAYER> "Open the gate. I have business within." <ACTION>',
            ("CALL_GUARDS",),
        ),
        StructuredCase(
            19,
            "structured/threat",
            "threat=2 with guards nearby deterministically maps to CALL_GUARDS.",
            '<BOS> <NPC> Marta <LOCATION> Britain <MOOD> wary <REL> -1 '
            '<STATE> task=threat;role=merchant;voice=plain;threat=2;courage=1;guards=1;health=2 '
            '<MEMORY> prior_attack=0 <PLAYER> "Do as I say, or taste my steel." <ACTION>',
            ("CALL_GUARDS",),
        ),
        StructuredCase(
            20,
            "structured/favor",
            "debt=1, low cost/danger, betrayed=0 deterministically maps to HELP.",
            '<BOS> <NPC> Elric <LOCATION> Yew <MOOD> warm <REL> 1 '
            '<STATE> task=favor;role=carpenter;voice=plain;debt=1;trust=1;cost=0;danger=0;urgent=0 '
            '<MEMORY> betrayed=0 <PLAYER> "I ask a favor of thee." <ACTION>',
            ("HELP",),
        ),
    ]

# ---------------------------------------------------------------------------
# Loading / prompt construction
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_checkpoint(requested: str, workspace: Path) -> Path:
    if requested != "auto":
        path = Path(requested)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = [
        workspace / "final_sft" / "checkpoints" / "best_balanced.pt",
        workspace / "final_sft" / "checkpoints" / "best_sft.pt",
        workspace / "final_sft" / "checkpoints" / "final.pt",
        workspace / "final_sft" / "checkpoints" / "latest.pt",
        workspace / "npc_behavior" / "checkpoints" / "best_balanced.pt",
        workspace / "npc_behavior" / "checkpoints" / "best_npc.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find final UO-Mind checkpoint automatically. "
        "Pass --checkpoint /path/to/checkpoint.pt"
    )


def find_tokenizer(requested: str, workspace: Path) -> Path:
    if requested != "auto":
        path = Path(requested)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    path = workspace / "tokenizer.json"
    if path.exists():
        return path

    raise FileNotFoundError(
        "Could not find tokenizer.json. Pass --tokenizer /path/to/tokenizer.json"
    )


def load_model(
    checkpoint_path: Path,
    tokenizer: Tokenizer,
    device: torch.device,
) -> Tuple[UOMindLM, ModelConfig]:
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_config" not in payload or "model_state_dict" not in payload:
        raise RuntimeError("checkpoint does not contain model_config/model_state_dict")

    config = ModelConfig(**dict(payload["model_config"]))

    if tokenizer.get_vocab_size() != config.vocab_size:
        raise RuntimeError(
            f"vocab mismatch tokenizer={tokenizer.get_vocab_size()} "
            f"model={config.vocab_size}"
        )

    model = UOMindLM(config)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)

    missing = [
        key
        for key in incompatible.missing_keys
        if "causal_bias" not in key
    ]
    unexpected = list(incompatible.unexpected_keys)

    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch: missing={missing} unexpected={unexpected}"
        )

    model = model.to(device)
    model.eval()
    return model, config


def encode_text(tokenizer: Tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text).ids


def encode_context_message(
    tokenizer: Tokenizer,
    message: Dict[str, str],
) -> List[int]:
    role = message["role"]
    content = " ".join(message["content"].split())

    if role == "system":
        marker = "<STATE>"
    elif role == "user":
        marker = "<PLAYER>"
    elif role == "assistant":
        marker = "<SAY>"
    else:
        raise ValueError(f"unsupported role {role}")

    ids = [tokenizer.token_to_id(marker)]
    ids.extend(encode_text(tokenizer, " " + content))

    if role == "assistant":
        ids.append(tokenizer.token_to_id("<EOS>"))

    return ids


def build_natural_prompt(
    tokenizer: Tokenizer,
    messages: Sequence[Dict[str, str]],
    max_prompt_tokens: int,
    max_system_tokens: int = 40,
) -> Tuple[List[int], bool]:
    """Exact SFT-style context construction ending in <SAY>."""

    bos_id = tokenizer.token_to_id("<BOS>")
    say_id = tokenizer.token_to_id("<SAY>")

    system_tokens: List[int] = []
    dynamic_tokens: List[int] = []

    for message in messages:
        tokens = encode_context_message(tokenizer, message)
        if message["role"] == "system" and not system_tokens:
            system_tokens = tokens[:max_system_tokens]
        else:
            dynamic_tokens.extend(tokens)

    remaining = max_prompt_tokens - 2
    if remaining < 0:
        raise ValueError("max_prompt_tokens too small")

    system_keep = system_tokens[:remaining]
    remaining -= len(system_keep)
    dynamic_keep = dynamic_tokens[-remaining:] if remaining > 0 else []

    prompt = [bos_id] + system_keep + dynamic_keep + [say_id]

    original_len = 1 + len(system_tokens) + len(dynamic_tokens) + 1
    return prompt, len(prompt) < original_len



def encode_assistant_example_exact_sft(
    tokenizer: Tokenizer,
    preceding_messages: Sequence[Dict[str, str]],
    assistant_text: str,
    seq_len: int,
    max_target_tokens: int,
    max_system_tokens: int,
    pad_id: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, bool]:
    """Bit-for-bit copy of the final-SFT response encoder."""
    eos_id = tokenizer.token_to_id("<EOS>")
    target = encode_text(tokenizer, " " + assistant_text)
    response_truncated = False

    max_content_tokens = max(1, max_target_tokens - 1)
    if len(target) > max_content_tokens:
        target = target[:max_content_tokens]
        response_truncated = True

    target_with_eos = target + [eos_id]
    stream_budget = seq_len + 1
    prefix_budget = stream_budget - len(target_with_eos)
    if prefix_budget < 2:
        return None, None, response_truncated, True

    prompt, context_truncated = build_natural_prompt(
        tokenizer=tokenizer,
        messages=preceding_messages,
        max_prompt_tokens=prefix_budget,
        max_system_tokens=max_system_tokens,
    )

    full = prompt + target_with_eos
    if len(full) < 2 or len(full) > stream_budget:
        return None, None, response_truncated, context_truncated

    x = np.full((seq_len,), pad_id, dtype=np.uint16)
    y = np.full((seq_len,), -100, dtype=np.int16)
    input_ids = full[:-1]
    next_ids = full[1:]
    x[:len(input_ids)] = np.asarray(input_ids, dtype=np.uint16)

    # <SAY> itself predicts assistant token 0.
    supervised_start = len(prompt) - 1
    y[supervised_start:len(next_ids)] = np.asarray(
        next_ids[supervised_start:], dtype=np.int16
    )
    if not np.any(y != -100):
        return None, None, response_truncated, context_truncated
    return x, y, response_truncated, context_truncated


def rebuild_validation_examples(
    sft_dir: Path,
    tokenizer: Tokenizer,
    category_map: Dict[str, int],
    val_fraction: float,
    split_seed: int,
    seq_len: int,
    max_target_tokens: int,
    max_system_tokens: int,
) -> List[RebuiltValidationExample]:
    heldout = _heldout_sft_conversations(
        sft_dir=sft_dir,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )
    pad_id = tokenizer.token_to_id("<PAD>")
    rebuilt: List[RebuiltValidationExample] = []
    row = 0
    for conversation in heldout:
        messages = list(conversation["messages"])
        category = str(conversation["category"])
        if category not in category_map:
            raise RuntimeError(f"category {category!r} is absent from cached category_map")
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            x, y, response_truncated, context_truncated = encode_assistant_example_exact_sft(
                tokenizer=tokenizer,
                preceding_messages=messages[:index],
                assistant_text=message["content"],
                seq_len=seq_len,
                max_target_tokens=max_target_tokens,
                max_system_tokens=max_system_tokens,
                pad_id=pad_id,
            )
            if x is None or y is None:
                continue
            rebuilt.append(
                RebuiltValidationExample(
                    row=row,
                    category=category,
                    category_id=int(category_map[category]),
                    source=f"{Path(str(conversation['file'])).name}:{conversation['line']}",
                    messages=messages[:index],
                    reference=message["content"],
                    input_ids=x,
                    labels=y,
                    context_truncated=context_truncated,
                    response_truncated=response_truncated,
                )
            )
            row += 1
    return rebuilt


def prefix_from_cached_row(
    x_row: np.ndarray,
    label_row: np.ndarray,
    tokenizer: Tokenizer,
) -> Tuple[List[int], int]:
    supervised = np.flatnonzero(np.asarray(label_row) != -100)
    if supervised.size == 0:
        raise RuntimeError("cached validation row has no supervised tokens")
    say_position = int(supervised[0])
    prefix = np.asarray(x_row[:say_position + 1], dtype=np.int64).tolist()
    say_id = tokenizer.token_to_id("<SAY>")
    if not prefix or prefix[-1] != say_id:
        raise RuntimeError(
            f"cached prefix does not end in <SAY>: row SAY position={say_position}, "
            f"last_id={prefix[-1] if prefix else None}, expected={say_id}"
        )
    return [int(v) for v in prefix], say_position


def reference_from_cached_labels(
    label_row: np.ndarray,
    tokenizer: Tokenizer,
) -> str:
    eos_id = tokenizer.token_to_id("<EOS>")
    ids = [int(v) for v in np.asarray(label_row) if int(v) != -100]
    if ids and ids[-1] == eos_id:
        ids = ids[:-1]
    return tokenizer.decode(ids, skip_special_tokens=False).strip()


def audit_sft_cache(
    workspace: Path,
    sft_dir: Path,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
) -> Dict[str, object]:
    cache_dir = Path(args.sft_cache_dir) if args.sft_cache_dir else workspace / "final_sft"
    meta_path = cache_dir / "sft_dataset_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"SFT audit requires {meta_path}. Run it in the same Colab workspace as final SFT."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    required_keys = (
        "val_inputs",
        "val_labels",
        "val_categories",
        "category_map",
    )
    missing = [key for key in required_keys if key not in meta]
    if missing:
        raise RuntimeError(f"SFT metadata missing keys: {missing}")

    x_path = Path(meta["val_inputs"])
    y_path = Path(meta["val_labels"])
    c_path = Path(meta["val_categories"])
    for path in (x_path, y_path, c_path):
        if not path.exists():
            raise FileNotFoundError(f"cached SFT validation array missing: {path}")

    cached_x = np.load(x_path, mmap_mode="r")
    cached_y = np.load(y_path, mmap_mode="r")
    cached_c = np.load(c_path, mmap_mode="r")

    # The final-SFT v3 cache metadata records the encoded arrays/category map
    # but not these three CLI values. Defaults here intentionally match the
    # trainer that created the user's run; all are overrideable on the CLI.
    val_fraction = float(args.sft_val_fraction)
    max_target_tokens = int(args.sft_max_target_tokens)
    max_system_tokens = int(args.max_system_tokens)
    category_map = {str(k): int(v) for k, v in meta["category_map"].items()}

    rebuilt = rebuild_validation_examples(
        sft_dir=sft_dir,
        tokenizer=tokenizer,
        category_map=category_map,
        val_fraction=val_fraction,
        split_seed=args.sft_split_seed,
        seq_len=int(cached_x.shape[1]),
        max_target_tokens=max_target_tokens,
        max_system_tokens=max_system_tokens,
    )

    count_match = len(rebuilt) == int(cached_x.shape[0]) == int(cached_y.shape[0]) == int(cached_c.shape[0])
    compare_count = min(len(rebuilt), int(cached_x.shape[0]))
    input_matches = 0
    label_matches = 0
    category_matches = 0
    say_position_matches = 0
    first_mismatch: Optional[Dict[str, object]] = None

    say_id = tokenizer.token_to_id("<SAY>")
    for row in range(compare_count):
        item = rebuilt[row]
        cx = np.asarray(cached_x[row])
        cy = np.asarray(cached_y[row])
        cc = int(cached_c[row])
        x_ok = np.array_equal(cx, item.input_ids)
        y_ok = np.array_equal(cy, item.labels)
        c_ok = cc == item.category_id
        input_matches += int(x_ok)
        label_matches += int(y_ok)
        category_matches += int(c_ok)

        cached_supervised = np.flatnonzero(cy != -100)
        rebuilt_supervised = np.flatnonzero(item.labels != -100)
        say_ok = (
            cached_supervised.size > 0
            and rebuilt_supervised.size > 0
            and int(cached_supervised[0]) == int(rebuilt_supervised[0])
            and int(cx[int(cached_supervised[0])]) == say_id
            and int(item.input_ids[int(rebuilt_supervised[0])]) == say_id
        )
        say_position_matches += int(say_ok)

        if first_mismatch is None and not (x_ok and y_ok and c_ok and say_ok):
            detail: Dict[str, object] = {
                "row": row,
                "category": item.category,
                "source": item.source,
                "input_match": x_ok,
                "label_match": y_ok,
                "category_match": c_ok,
                "say_position_match": say_ok,
            }
            if not x_ok:
                positions = np.flatnonzero(cx != item.input_ids)
                detail["first_input_diff"] = int(positions[0]) if positions.size else None
            if not y_ok:
                positions = np.flatnonzero(cy != item.labels)
                detail["first_label_diff"] = int(positions[0]) if positions.size else None
            first_mismatch = detail

    all_match = (
        count_match
        and input_matches == compare_count
        and label_matches == compare_count
        and category_matches == compare_count
        and say_position_matches == compare_count
    )

    print("SFT FORMAT AUDIT")
    print("----------------")
    print(f"cache metadata:            {meta_path}")
    print(f"cached validation rows:    {cached_x.shape[0]:,}")
    print(f"rebuilt validation rows:   {len(rebuilt):,}")
    print(f"input tensors exact:       {input_matches:,}/{compare_count:,}")
    print(f"label tensors exact:       {label_matches:,}/{compare_count:,}")
    print(f"category IDs exact:        {category_matches:,}/{compare_count:,}")
    print(f"<SAY>/mask positions exact:{say_position_matches:,}/{compare_count:,}")
    print(f"split seed:                {args.sft_split_seed}")
    print(f"validation fraction:       {val_fraction}")
    print(f"max system tokens:         {max_system_tokens}")
    print(f"max target tokens:         {max_target_tokens}")
    print(f"AUDIT RESULT:              {'100% MATCH' if all_match else 'MISMATCH'}")
    if first_mismatch is not None:
        print(f"first mismatch:            {first_mismatch}")
    print()

    report = {
        "all_match": all_match,
        "cached_rows": int(cached_x.shape[0]),
        "rebuilt_rows": len(rebuilt),
        "compared_rows": compare_count,
        "input_matches": input_matches,
        "label_matches": label_matches,
        "category_matches": category_matches,
        "say_position_matches": say_position_matches,
        "first_mismatch": first_mismatch,
        "meta_path": str(meta_path),
        "validation_inputs": str(x_path),
        "validation_labels": str(y_path),
        "validation_categories": str(c_path),
        "split_seed": args.sft_split_seed,
        "val_fraction": val_fraction,
        "max_system_tokens": max_system_tokens,
        "max_target_tokens": max_target_tokens,
    }

    if not all_match and not args.allow_audit_mismatch:
        raise RuntimeError(
            "SFT format audit FAILED. Inference has been stopped before generation. "
            "Use --allow-audit-mismatch only for debugging, not for model evaluation."
        )

    return {
        "report": report,
        "meta": meta,
        "cached_x": cached_x,
        "cached_y": cached_y,
        "cached_c": cached_c,
        "rebuilt": rebuilt,
    }


def load_heldout_cases_from_audit(
    audit: Dict[str, object],
    tokenizer: Tokenizer,
    count: int,
) -> List[NaturalCase]:
    rebuilt: List[RebuiltValidationExample] = audit["rebuilt"]  # type: ignore[assignment]
    cached_x = audit["cached_x"]
    cached_y = audit["cached_y"]

    by_category: Dict[str, List[RebuiltValidationExample]] = defaultdict(list)
    for item in rebuilt:
        by_category[item.category].append(item)

    categories = sorted(by_category)
    if count >= len(categories):
        selected_categories = categories
    else:
        positions = np.linspace(0, len(categories) - 1, count)
        selected_categories = []
        used = set()
        for position in positions:
            idx = int(round(float(position)))
            while idx in used and idx + 1 < len(categories):
                idx += 1
            if idx in used:
                idx = next(i for i in range(len(categories)) if i not in used)
            used.add(idx)
            selected_categories.append(categories[idx])

    cases: List[NaturalCase] = []
    for number, category in enumerate(selected_categories[:count], start=1):
        items = by_category[category]
        # Keep selection deterministic while deriving the actual prefix only
        # from the cached SFT tensor, never from reconstructed text.
        item = sorted(items, key=lambda x: _stable_hash_int(x.source))[0]
        prefix_ids, _ = prefix_from_cached_row(
            np.asarray(cached_x[item.row]),
            np.asarray(cached_y[item.row]),
            tokenizer,
        )
        cached_reference = reference_from_cached_labels(
            np.asarray(cached_y[item.row]),
            tokenizer,
        )
        cases.append(
            NaturalCase(
                number=number,
                category=f"heldout/{category}",
                goal="Generate from the exact cached SFT validation prefix; compare against its exact cached supervised target.",
                messages=item.messages,
                reference=cached_reference,
                source=item.source,
                cached_row=item.row,
                exact_prefix_ids=prefix_ids,
            )
        )
    return cases


@torch.inference_mode()
def score_cached_validation_row(
    model: UOMindLM,
    x_row: np.ndarray,
    y_row: np.ndarray,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, object]:
    x = torch.as_tensor(np.asarray(x_row, dtype=np.int64), dtype=torch.long, device=device).unsqueeze(0)
    labels = torch.as_tensor(np.asarray(y_row, dtype=np.int64), dtype=torch.long, device=device).unsqueeze(0)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, model.config.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )
    mask = labels.ne(-100)
    predictions = logits.argmax(dim=-1)
    correct = (predictions.eq(labels) & mask).sum()
    total = mask.sum()
    loss_value = float(loss.item())
    return {
        "loss": loss_value,
        "ppl": math.exp(min(loss_value, 20.0)),
        "token_accuracy": float((correct.float() / total.clamp_min(1)).item()),
        "tokens": int(total.item()),
        "clipped": False,
        "source": "cached_validation_tensor",
    }


def render_natural_messages(messages: Sequence[Dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = message["role"].upper()
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


@torch.inference_mode()
def score_reference(
    model: UOMindLM,
    tokenizer: Tokenizer,
    prefix_ids: Sequence[int],
    reference: str,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, object]:
    """Teacher-force the curated reference under the same SFT response mask."""
    pad_id = tokenizer.token_to_id("<PAD>")
    eos_id = tokenizer.token_to_id("<EOS>")
    target_ids = tokenizer.encode(" " + reference).ids

    # A length-T causal model consumes a T+1 token stream for shifted labels.
    available_target = model.config.max_seq_len + 1 - len(prefix_ids)
    clipped = False
    if available_target <= 1:
        return {
            "loss": float("nan"),
            "ppl": float("nan"),
            "token_accuracy": 0.0,
            "tokens": 0,
            "clipped": True,
        }

    max_content = available_target - 1
    if len(target_ids) > max_content:
        target_ids = target_ids[:max_content]
        clipped = True

    full = list(prefix_ids) + target_ids + [eos_id]
    input_ids = full[:-1]
    next_ids = full[1:]

    x = torch.full(
        (1, model.config.max_seq_len),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    labels = torch.full_like(x, -100)

    x[0, :len(input_ids)] = torch.as_tensor(
        input_ids,
        dtype=torch.long,
        device=device,
    )

    supervised_start = len(prefix_ids) - 1
    labels[0, supervised_start:len(next_ids)] = torch.as_tensor(
        next_ids[supervised_start:],
        dtype=torch.long,
        device=device,
    )

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, model.config.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )

    mask = labels.ne(-100)
    predictions = logits.argmax(dim=-1)
    correct = (predictions.eq(labels) & mask).sum()
    total = mask.sum()

    loss_value = float(loss.item())
    return {
        "loss": loss_value,
        "ppl": math.exp(min(loss_value, 20.0)),
        "token_accuracy": float((correct.float() / total.clamp_min(1)).item()),
        "tokens": int(total.item()),
        "clipped": clipped,
    }


def repetition_fraction(token_ids: Sequence[int], n: int = 3) -> float:
    if len(token_ids) < n * 2:
        return 0.0
    grams = [tuple(token_ids[i:i+n]) for i in range(len(token_ids) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def choose_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator,
    generated_ids: Sequence[int],
    repetition_penalty: float,
) -> Tuple[int, float]:
    logits = logits.float()

    if repetition_penalty > 1.0 and generated_ids:
        seen = torch.as_tensor(
            sorted(set(int(x) for x in generated_ids)),
            dtype=torch.long,
            device=logits.device,
        )
        selected = logits.index_select(0, seen)
        adjusted = torch.where(
            selected < 0,
            selected * repetition_penalty,
            selected / repetition_penalty,
        )
        logits = logits.clone()
        logits.index_copy_(0, seen, adjusted)

    if temperature <= 0.0:
        probs = torch.softmax(logits, dim=-1)
        token = int(torch.argmax(logits).item())
        probability = float(probs[token].item())
        return token, probability

    logits = logits / temperature

    if top_k > 0 and top_k < logits.numel():
        cutoff = torch.topk(logits, top_k).values[-1]
        logits = torch.where(
            logits < cutoff,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(0, sorted_indices, sorted_logits)
        logits = filtered

    probs = torch.softmax(logits, dim=-1)
    token_tensor = torch.multinomial(probs, num_samples=1, generator=generator)
    token = int(token_tensor.item())
    probability = float(probs[token].item())
    return token, probability


@torch.inference_mode()
def generate(
    model: UOMindLM,
    tokenizer: Tokenizer,
    prefix_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    amp_dtype: torch.dtype,
    repetition_penalty: float,
    device: torch.device,
) -> Dict[str, object]:
    if not prefix_ids:
        raise ValueError("empty prefix")
    if len(prefix_ids) > model.config.max_seq_len:
        raise ValueError(
            f"prefix has {len(prefix_ids)} tokens but context is {model.config.max_seq_len}"
        )

    pad_id = tokenizer.token_to_id("<PAD>")
    eos_id = tokenizer.token_to_id("<EOS>")
    marker_stop_ids = {
        tokenizer.token_to_id(name)
        for name in ("<STATE>", "<PLAYER>", "<NPC>")
        if tokenizer.token_to_id(name) is not None
    }

    context_ids = list(prefix_ids)
    generated: List[int] = []
    probabilities: List[float] = []
    stop_reason = "max_new_tokens"
    x = torch.full(
        (1, model.config.max_seq_len),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(seed)

    torch.cuda.synchronize(device)
    started = time.perf_counter()

    for _ in range(max_new_tokens):
        # Training at x length T still has a valid y[T-1] next-token target.
        # Therefore len(context)==T is allowed for ONE final prediction.
        if len(context_ids) > model.config.max_seq_len:
            stop_reason = "context_limit"
            break

        x.fill_(pad_id)
        prefix_tensor = torch.as_tensor(context_ids, dtype=torch.long, device=device)
        x[0, :len(context_ids)].copy_(prefix_tensor)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(x)

        next_logits = logits[0, len(context_ids) - 1]
        token, probability = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=torch_generator,
            generated_ids=generated,
            repetition_penalty=repetition_penalty,
        )

        if token == eos_id:
            stop_reason = "EOS"
            break
        if token in marker_stop_ids:
            stop_reason = "turn_marker"
            break

        generated.append(token)
        probabilities.append(probability)

        if len(context_ids) == model.config.max_seq_len:
            # The sampled token is a valid prediction of the last model
            # position, but it cannot be fed back without exceeding context.
            stop_reason = "context_limit_after_token"
            break

        context_ids.append(token)

    torch.cuda.synchronize(device)
    elapsed = max(1e-9, time.perf_counter() - started)
    decoded = tokenizer.decode(generated, skip_special_tokens=False).strip()
    return {
        "generated_ids": generated,
        "text": decoded,
        "tokens": len(generated),
        "stop_reason": stop_reason,
        "latency_seconds": elapsed,
        "tokens_per_second": len(generated) / elapsed if generated else 0.0,
        "mean_selected_probability": (
            float(sum(probabilities) / len(probabilities)) if probabilities else 0.0
        ),
    }


def detect_control_leakage(
    tokenizer: Tokenizer,
    generated_ids: Sequence[int],
) -> List[str]:
    leakage = []
    watched = [
        "<ACTION>",
        "<TONE>",
        "<STATE>",
        "<PLAYER>",
        "<NPC>",
        *ACTION_TOKENS.values(),
        *TONE_TOKENS.values(),
    ]

    generated_set = set(generated_ids)
    for token in watched:
        token_id = tokenizer.token_to_id(token)
        if token_id is not None and token_id in generated_set:
            leakage.append(token)
    return leakage


def parse_structured_output(
    tokenizer: Tokenizer,
    generated_ids: Sequence[int],
) -> Dict[str, Optional[str]]:
    tokens = [tokenizer.id_to_token(int(token_id)) for token_id in generated_ids]

    action = None
    tone = None
    speech = None

    for token in tokens:
        if token in TOKEN_TO_ACTION:
            action = TOKEN_TO_ACTION[token]
            break

    tone_marker_id = tokenizer.token_to_id("<TONE>")
    say_marker_id = tokenizer.token_to_id("<SAY>")

    if tone_marker_id in generated_ids:
        ids = list(generated_ids)
        index = ids.index(tone_marker_id)
        for candidate_id in ids[index + 1:]:
            if candidate_id == say_marker_id:
                break
            token = tokenizer.id_to_token(int(candidate_id))
            if token in TOKEN_TO_TONE:
                tone = TOKEN_TO_TONE[token]
                break
            # ByteLevel can emit a standalone whitespace token around special
            # markers. Ignore it rather than reporting it as the tone.
            if tokenizer.decode([int(candidate_id)], skip_special_tokens=False).strip() == "":
                continue
            if tone is None:
                tone = token

    if say_marker_id in generated_ids:
        index = list(generated_ids).index(say_marker_id)
        speech_ids = list(generated_ids[index + 1:])
        speech = tokenizer.decode(speech_ids, skip_special_tokens=False).strip()

    return {
        "action": action,
        "tone": tone,
        "speech": speech,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_header(model: UOMindLM, checkpoint: Path, tokenizer_path: Path, dtype: torch.dtype) -> None:
    gains = model.geometry_gain_summary()
    print("UO-Mind inference test")
    print("----------------------")
    print(f"checkpoint:          {checkpoint}")
    print(f"tokenizer:           {tokenizer_path}")
    print(f"parameters:          {model.num_parameters():,}")
    print(f"context:             {model.config.max_seq_len}")
    print(f"precision:           {'bfloat16' if dtype == torch.bfloat16 else 'float16'}")
    print(f"geometry gain mean:  {gains['mean']:.4f}")
    print(f"geometry gain range: {gains['min']:.4f} .. {gains['max']:.4f}")
    print()


def result_text_block(result: Dict[str, object]) -> str:
    lines = [
        "=" * 80,
        f"TEST {result['number']:02d} | {result['category']}",
        "=" * 80,
        f"GOAL: {result['goal']}",
        "",
        "PROMPT (human-readable):",
        str(result["prompt_rendered"]),
        "",
    ]

    if result.get("model_input"):
        lines.extend(
            [
                "MODEL INPUT (actual control-token serialization):",
                str(result["model_input"]),
                "",
            ]
        )

    if result["mode"] == "structured":
        lines.extend(
            [
                f"ACTION: {result.get('action')}",
                f"TONE:   {result.get('tone')}",
                f"SPEECH: {result.get('speech')}",
                f"EXPECTED ACTION(S): {', '.join(result.get('expected_actions', []))}",
                f"ACTION CHECK: {'PASS' if result.get('action_check') else 'MISS'}",
                "",
                f"RAW OUTPUT: {result['text']}",
            ]
        )
    else:
        lines.extend(
            [
                "MODEL:",
                str(result["text"]),
            ]
        )

        if result.get("reference") is not None:
            lines.extend(
                [
                    "",
                    "CURATED REFERENCE:",
                    str(result["reference"]),
                ]
            )

            ref = result.get("reference_score") or {}
            if ref:
                lines.append(
                    "REFERENCE TEACHER-FORCED: "
                    f"loss={ref.get('loss', float('nan')):.3f} | "
                    f"ppl={ref.get('ppl', float('nan')):.2f} | "
                    f"token_acc={100.0 * ref.get('token_accuracy', 0.0):.1f}%"
                    + (" | clipped" if ref.get("clipped") else "")
                )

        lines.extend(
            [
                "",
                f"CONTROL LEAKAGE: {result.get('control_leakage') or 'none'}",
                f"REPEATED 3-GRAM FRACTION: {100.0 * float(result.get('repetition_fraction', 0.0)):.1f}%",
            ]
        )

    lines.extend(
        [
            "",
            f"prompt_tokens={result.get('prompt_tokens')} | "
            f"generated={result['tokens']} | stop={result['stop_reason']} | "
            f"latency={result['latency_seconds'] * 1000:.1f} ms | "
            f"tok/s={result['tokens_per_second']:.1f} | "
            f"mean_selected_p={result['mean_selected_probability']:.3f}",
            "",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


def warmup_model(
    model: UOMindLM,
    tokenizer: Tokenizer,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    pad_id = tokenizer.token_to_id("<PAD>")
    bos_id = tokenizer.token_to_id("<BOS>")
    x = torch.full(
        (1, model.config.max_seq_len),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    x[0, 0] = bos_id

    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=dtype,
    ):
        for _ in range(3):
            _ = model(x)

    torch.cuda.synchronize(device)


def run_suite(args: argparse.Namespace) -> List[Dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    device = torch.device("cuda")
    set_seed(args.seed)

    workspace = Path(args.workspace)
    checkpoint = find_checkpoint(args.checkpoint, workspace)
    tokenizer_path = find_tokenizer(args.tokenizer, workspace)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model, config = load_model(checkpoint, tokenizer, device)
    dtype = choose_dtype()

    print_header(model, checkpoint, tokenizer_path, dtype)
    print("Warming inference kernels...", flush=True)
    warmup_model(model, tokenizer, dtype, device)
    print("Warmup complete.\n", flush=True)

    results: List[Dict[str, object]] = []

    run_natural = args.suite in {"all", "natural"}
    run_structured = args.suite in {"all", "structured"}

    # Audit is deliberately performed before ANY generation. If the JSONL
    # reconstruction differs from the arrays that final SFT actually trained
    # and validated on, this script aborts by default.
    audit = audit_sft_cache(
        workspace=workspace,
        sft_dir=Path(args.sft_dir),
        tokenizer=tokenizer,
        args=args,
    )

    natural_test_cases: List[NaturalCase] = []
    if run_natural:
        heldout = load_heldout_cases_from_audit(
            audit=audit,
            tokenizer=tokenizer,
            count=args.heldout_count,
        )

        if heldout:
            print(
                f"Loaded {len(heldout)} held-out queries FROM THE EXACT CACHED SFT VALIDATION TENSORS.\n",
                flush=True,
            )
            natural_test_cases.extend(heldout)
        else:
            print(
                "WARNING: no cached held-out SFT queries were selected; "
                "running challenge tests only.\n",
                flush=True,
            )

        # Keep the suite at 16 natural queries when the expected 12 heldout
        # examples are available. If fewer are available, challenge numbering
        # follows the actual heldout count.
        challenge_start = len(natural_test_cases) + 1
        challenges = challenge_cases(start_number=challenge_start)
        natural_test_cases.extend(challenges)

        # Exactly 16 natural tests for the normal 20-query suite.
        if len(natural_test_cases) > 16:
            natural_test_cases = natural_test_cases[:16]

        # Renumber deterministically after any fallback/truncation.
        for index, case in enumerate(natural_test_cases, start=1):
            case.number = index

        cached_x = audit["cached_x"]
        cached_y = audit["cached_y"]

        for case in natural_test_cases:
            if case.exact_prefix_ids is not None:
                # Held-out cases generate from the exact prefix extracted from
                # the saved final-SFT validation input tensor. There is no
                # re-tokenization path left that could disagree with training.
                prompt_ids = list(case.exact_prefix_ids)
                trimmed = False
            else:
                # Challenge prompts have no training target, so preserve the
                # full prompt up to the actual 128-token model limit. Do not
                # pre-reserve generation space; generation stops naturally at
                # the context boundary.
                prompt_ids, trimmed = build_natural_prompt(
                    tokenizer,
                    case.messages,
                    max_prompt_tokens=config.max_seq_len,
                    max_system_tokens=args.max_system_tokens,
                )

            generated = generate(
                model=model,
                tokenizer=tokenizer,
                prefix_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed + case.number * 1009,
                amp_dtype=dtype,
                device=device,
                repetition_penalty=args.repetition_penalty,
            )

            reference_score = None
            if case.cached_row is not None:
                reference_score = score_cached_validation_row(
                    model=model,
                    x_row=np.asarray(cached_x[case.cached_row]),
                    y_row=np.asarray(cached_y[case.cached_row]),
                    amp_dtype=dtype,
                    device=device,
                )
            elif case.reference is not None:
                reference_score = score_reference(
                    model=model,
                    tokenizer=tokenizer,
                    prefix_ids=prompt_ids,
                    reference=case.reference,
                    amp_dtype=dtype,
                    device=device,
                )

            result = {
                "number": case.number,
                "mode": "natural",
                "category": case.category,
                "goal": case.goal,
                "source": case.source,
                "cached_validation_row": case.cached_row,
                "prompt_rendered": render_natural_messages(case.messages),
                "model_input": tokenizer.decode(
                    prompt_ids,
                    skip_special_tokens=False,
                ),
                "prompt_tokens": len(prompt_ids),
                "prompt_trimmed": trimmed,
                "reference": case.reference,
                "reference_score": reference_score,
                **generated,
                "control_leakage": detect_control_leakage(
                    tokenizer,
                    generated["generated_ids"],
                ),
                "repetition_fraction": repetition_fraction(
                    generated["generated_ids"],
                    n=3,
                ),
            }
            results.append(result)
            print(result_text_block(result), flush=True)

    if run_structured:
        struct_cases = structured_cases()
        # If natural tests were omitted, retain their canonical numbering.
        # If fewer than 16 natural tests were available, structured numbering
        # still prints 17-20 so comparisons across runs remain easy.
        for case in struct_cases:
            prompt_ids = tokenizer.encode(case.prompt).ids

            available = config.max_seq_len - len(prompt_ids) + 1
            if len(prompt_ids) > config.max_seq_len:
                raise RuntimeError(
                    f"structured test {case.number} prompt itself is too long: "
                    f"{len(prompt_ids)} > {config.max_seq_len}"
                )

            actual_max_new = min(
                args.structured_max_new_tokens,
                available,
            )

            if actual_max_new < args.structured_max_new_tokens:
                print(
                    f"[test {case.number}] structured output budget reduced "
                    f"from {args.structured_max_new_tokens} to {actual_max_new} "
                    f"tokens to respect context 128.",
                    flush=True,
                )

            generated = generate(
                model=model,
                tokenizer=tokenizer,
                prefix_ids=prompt_ids,
                max_new_tokens=actual_max_new,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed + case.number * 1009,
                amp_dtype=dtype,
                device=device,
                repetition_penalty=args.repetition_penalty,
            )

            parsed = parse_structured_output(
                tokenizer,
                generated["generated_ids"],
            )

            action_check = parsed["action"] in case.expected_actions

            result = {
                "number": case.number,
                "mode": "structured",
                "category": case.category,
                "goal": case.goal,
                "prompt_rendered": case.prompt,
                "model_input": case.prompt,
                "prompt_tokens": len(prompt_ids),
                "prompt_trimmed": False,
                **generated,
                **parsed,
                "expected_actions": list(case.expected_actions),
                "action_check": action_check,
            }
            results.append(result)
            print(result_text_block(result), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "inference_results_v3.json"
    txt_path = output_dir / "inference_results_v3.txt"

    serializable_results = []
    for result in results:
        clean = dict(result)
        clean.pop("generated_ids", None)
        serializable_results.append(clean)

    natural_results = [r for r in results if r["mode"] == "natural"]
    heldout_results = [
        r for r in natural_results
        if str(r["category"]).startswith("heldout/")
    ]
    challenge_results = [
        r for r in natural_results
        if str(r["category"]).startswith("challenge/")
    ]
    structured_results = [r for r in results if r["mode"] == "structured"]

    summary_metrics: Dict[str, object] = {}

    if natural_results:
        leakage_count = sum(bool(r["control_leakage"]) for r in natural_results)
        eos_count = sum(r["stop_reason"] == "EOS" for r in natural_results)
        mean_tokens = sum(r["tokens"] for r in natural_results) / len(natural_results)
        mean_latency = sum(r["latency_seconds"] for r in natural_results) / len(natural_results)
        repetitive = sum(float(r.get("repetition_fraction", 0.0)) >= 0.20 for r in natural_results)
        summary_metrics.update(
            {
                "natural_eos_stops": f"{eos_count}/{len(natural_results)}",
                "natural_control_leakage": f"{leakage_count}/{len(natural_results)}",
                "natural_mean_tokens": mean_tokens,
                "natural_mean_latency_ms": mean_latency * 1000.0,
                "natural_repetitive_3gram_cases": f"{repetitive}/{len(natural_results)}",
            }
        )

    if heldout_results:
        scores = [r["reference_score"] for r in heldout_results if r.get("reference_score")]
        if scores:
            mean_ref_loss = sum(float(x["loss"]) for x in scores) / len(scores)
            mean_ref_acc = sum(float(x["token_accuracy"]) for x in scores) / len(scores)
            summary_metrics["heldout_reference_teacher_forced_loss"] = mean_ref_loss
            summary_metrics["heldout_reference_teacher_forced_ppl"] = math.exp(min(mean_ref_loss, 20.0))
            summary_metrics["heldout_reference_token_accuracy"] = mean_ref_acc

    if structured_results:
        passed = sum(bool(r["action_check"]) for r in structured_results)
        summary_metrics["structured_action_checks"] = f"{passed}/{len(structured_results)}"

    summary = {
        "checkpoint": str(checkpoint),
        "tokenizer": str(tokenizer_path),
        "model_config": asdict(config),
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
        "sft_dir": args.sft_dir,
        "sft_val_fraction": args.sft_val_fraction,
        "sft_split_seed": args.sft_split_seed,
        "sft_format_audit": audit["report"],
        "summary_metrics": summary_metrics,
        "results": serializable_results,
    }

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    txt_path.write_text(
        "\n".join(result_text_block(result) for result in results),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("SUITE SUMMARY")
    print("=" * 80)
    print(f"SFT format audit:          {'100% MATCH' if audit['report']['all_match'] else 'MISMATCH'}")
    print(f"queries run: {len(results)}")
    print(f"held-out real SFT queries: {len(heldout_results)}")
    print(f"hard challenge queries:   {len(challenge_results)}")

    if natural_results:
        leakage_count = sum(bool(r["control_leakage"]) for r in natural_results)
        eos_count = sum(r["stop_reason"] == "EOS" for r in natural_results)
        repetitive = sum(float(r.get("repetition_fraction", 0.0)) >= 0.20 for r in natural_results)
        mean_tokens = sum(r["tokens"] for r in natural_results) / len(natural_results)
        mean_latency = sum(r["latency_seconds"] for r in natural_results) / len(natural_results)
        print(f"natural EOS stops:        {eos_count}/{len(natural_results)}")
        print(f"natural control leakage:  {leakage_count}/{len(natural_results)}")
        print(f"natural repetitive cases: {repetitive}/{len(natural_results)}")
        print(f"natural mean tokens:      {mean_tokens:.1f}")
        print(f"natural mean latency:     {mean_latency * 1000:.1f} ms")

    if heldout_results:
        scores = [r["reference_score"] for r in heldout_results if r.get("reference_score")]
        if scores:
            mean_ref_loss = sum(float(x["loss"]) for x in scores) / len(scores)
            mean_ref_acc = sum(float(x["token_accuracy"]) for x in scores) / len(scores)
            print(f"heldout reference TF loss:{mean_ref_loss:.3f} (ppl={math.exp(min(mean_ref_loss, 20.0)):.2f})")
            print(f"heldout reference tok acc:{mean_ref_acc * 100:.1f}%")

    if structured_results:
        passed = sum(bool(r["action_check"]) for r in structured_results)
        print(f"structured action checks: {passed}/{len(structured_results)}")

    print(f"results JSON: {json_path}")
    print(f"results text: {txt_path}")

    if args.interactive:
        interactive_loop(
            model=model,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            args=args,
        )

    return results


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------


def interactive_loop(
    model: UOMindLM,
    tokenizer: Tokenizer,
    dtype: torch.dtype,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    print("\nInteractive mode")
    print("----------------")
    print("Enter a system persona once, then chat. Type /quit to exit or /reset to clear history.")

    system = input("SYSTEM> ").strip()
    if not system:
        system = "Your name is Aldric. You are a friendly townsman from Britain."

    history: List[Dict[str, str]] = [
        {"role": "system", "content": system}
    ]

    turn = 0

    while True:
        user = input("YOU> ").strip()
        if user.lower() == "/quit":
            break
        if user.lower() == "/reset":
            history = [{"role": "system", "content": system}]
            print("History cleared.")
            continue
        if not user:
            continue

        history.append({"role": "user", "content": user})
        prompt_ids, trimmed = build_natural_prompt(
            tokenizer,
            history,
            max_prompt_tokens=model.config.max_seq_len,
            max_system_tokens=args.max_system_tokens,
        )

        generated = generate(
            model=model,
            tokenizer=tokenizer,
            prefix_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed + 100000 + turn,
            amp_dtype=dtype,
            device=device,
            repetition_penalty=args.repetition_penalty,
        )

        text = str(generated["text"])
        print(f"NPC> {text}")
        print(
            f"     [{generated['tokens']} tok, {generated['latency_seconds'] * 1000:.1f} ms, "
            f"stop={generated['stop_reason']}{', context trimmed' if trimmed else ''}]"
        )

        history.append({"role": "assistant", "content": text})
        turn += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a self-auditing 20-query inference diagnostic against final UO-Mind."
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=str(DEFAULT_WORKSPACE),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT),
    )
    parser.add_argument(
        "--sft-dir",
        type=str,
        default=str(DEFAULT_SFT_DIR),
        help="Folder containing the same curated JSONL files used for final SFT.",
    )
    parser.add_argument(
        "--heldout-count",
        type=int,
        default=12,
        help="Number of real category-stratified held-out SFT queries to sample.",
    )
    parser.add_argument(
        "--sft-val-fraction",
        type=float,
        default=0.08,
        help="Must match final-SFT --val-fraction to reconstruct heldout examples.",
    )
    parser.add_argument(
        "--sft-split-seed",
        type=int,
        default=5042,
        help="Default final-SFT split seed: training seed 42 + 5000.",
    )
    parser.add_argument(
        "--sft-cache-dir",
        type=str,
        default="",
        help=(
            "Directory containing sft_dataset_meta.json and cached final-SFT arrays. "
            "Defaults to <workspace>/final_sft."
        ),
    )
    parser.add_argument(
        "--allow-audit-mismatch",
        action="store_true",
        help=(
            "Debug only: continue generation even if rebuilt JSONL examples do not "
            "match the cached SFT validation tensors. Default is to abort."
        ),
    )
    parser.add_argument(
        "--suite",
        choices=("all", "natural", "structured"),
        default="all",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--structured-max-new-tokens",
        type=int,
        default=36,
    )
    parser.add_argument(
        "--max-system-tokens",
        type=int,
        default=40,
        help="Must match final-SFT --max-system-tokens (default was 40).",
    )
    parser.add_argument(
        "--sft-max-target-tokens",
        type=int,
        default=56,
        help="Must match final-SFT --max-target-tokens (default was 56).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy/deterministic. Try 0.6-0.8 for sampled dialogue.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="1.0 = disabled. Try 1.08-1.15 only after the raw greedy diagnostic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_new_tokens <= 0 or args.max_new_tokens >= 96:
        raise ValueError("--max-new-tokens should be between 1 and 95")
    if args.structured_max_new_tokens <= 0:
        raise ValueError("--structured-max-new-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if args.sft_max_target_tokens < 2 or args.sft_max_target_tokens > 128:
        raise ValueError("--sft-max-target-tokens must be between 2 and 128")
    if args.heldout_count < 0 or args.heldout_count > 16:
        raise ValueError("--heldout-count must be between 0 and 16")
    if not (0.0 < args.sft_val_fraction < 0.5):
        raise ValueError("--sft-val-fraction must be in (0, 0.5)")
    if not (0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0,1]")
    if args.repetition_penalty < 1.0:
        raise ValueError("--repetition-penalty must be >= 1.0")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    run_suite(args)


if __name__ == "__main__":
    main()
