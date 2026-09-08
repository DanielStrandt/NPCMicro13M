#!/usr/bin/env python3
"""
uomind_colab_inference_test.py

20-query inference/evaluation harness for the final 128-token UO-Mind model.

Default checkpoint search order:
    /content/uomind_tinystories/final_sft/checkpoints/best_balanced.pt
    /content/uomind_tinystories/final_sft/checkpoints/best_sft.pt
    /content/uomind_tinystories/final_sft/checkpoints/final.pt
    /content/uomind_tinystories/npc_behavior/checkpoints/best_balanced.pt

Default tokenizer:
    /content/uomind_tinystories/tokenizer.json

The suite contains:
    16 natural dialogue tests using the exact final-SFT prompt format
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
    !python uomind_colab_inference_test.py

Sample instead of greedy:
    !python uomind_colab_inference_test.py --temperature 0.7 --top-p 0.9 --seed 123

Longer generations:
    !python uomind_colab_inference_test.py --max-new-tokens 48

Run only natural or structured tests:
    !python uomind_colab_inference_test.py --suite natural
    !python uomind_colab_inference_test.py --suite structured

Interactive mode after the scripted suite:
    !python uomind_colab_inference_test.py --interactive
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
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


@dataclass
class StructuredCase:
    number: int
    category: str
    goal: str
    prompt: str
    expected_actions: Tuple[str, ...]


def natural_cases() -> List[NaturalCase]:
    return [
        NaturalCase(
            1,
            "identity/persona",
            "Maintain the supplied name, profession, town, and temperament; reject the player's false identity assumption.",
            [
                {"role": "system", "content": "Your name is Marta. You are a baker from Britain. You are warm but proud of your craft."},
                {"role": "user", "content": "Are you Jorah the fisherman from Vesper?"},
            ],
        ),
        NaturalCase(
            2,
            "social civility",
            "Respond naturally to a greeting in a friendly Britannian voice without overexplaining.",
            [
                {"role": "system", "content": "Your name is Willard. You are an elderly gardener from Yew. You are cheerful and friendly."},
                {"role": "user", "content": "Good morrow, Willard. How fares thee today?"},
            ],
        ),
        NaturalCase(
            3,
            "food/tavern",
            "Give a plausible cheap, filling tavern recommendation and stay in character.",
            [
                {"role": "system", "content": "Your name is Mira. You keep an inn in Trinsic. You are practical and good-hearted."},
                {"role": "user", "content": "I've only three coppers to my name. What can fill my belly?"},
            ],
        ),
        NaturalCase(
            4,
            "gossip/uncertainty",
            "Handle an unverified accusation cautiously rather than inventing certainty.",
            [
                {"role": "system", "content": "Your name is Jorah. You are a discreet scribe in Britain and dislike spreading dangerous rumors without evidence."},
                {"role": "user", "content": "I heard the mayor steals from the treasury. Is it true?"},
            ],
        ),
        NaturalCase(
            5,
            "weather/practical reasoning",
            "Connect threatening weather to practical farm advice.",
            [
                {"role": "system", "content": "Your name is Edric. You are a farmer near Minoc. You have worked the fields for forty years."},
                {"role": "user", "content": "Those clouds are nearly black and the wind has turned cold. Should I leave the hay out?"},
            ],
        ),
        NaturalCase(
            6,
            "health",
            "Give sensible in-world healer advice without sounding like a modern encyclopedia.",
            [
                {"role": "system", "content": "Your name is Elara. You are a healer from Cove. You are calm and compassionate."},
                {"role": "user", "content": "My little boy has had a fever since last night and will barely drink. What should I do?"},
            ],
        ),
        NaturalCase(
            7,
            "virtues",
            "Reason briefly about honesty versus loyalty instead of giving a canned definition.",
            [
                {"role": "system", "content": "Your name is Aldren. You are an old scholar in Yew who thinks often about the Virtues."},
                {"role": "user", "content": "Is honesty still a virtue when telling the truth may cost a good friend dearly?"},
            ],
        ),
        NaturalCase(
            8,
            "geography",
            "Answer as a local traveler and provide plausible Britannian direction knowledge.",
            [
                {"role": "system", "content": "Your name is Rowan. You are a courier who travels often between Britain, Yew, and Vesper."},
                {"role": "user", "content": "If I leave Britain for Yew on foot, which way should I set out?"},
            ],
        ),
        NaturalCase(
            9,
            "law",
            "Recommend a believable lawful response to theft without needlessly escalating.",
            [
                {"role": "system", "content": "Your name is Garrick. You are a veteran town guard in Trinsic. You are stern but fair."},
                {"role": "user", "content": "I saw a man steal a purse in the market. Should I chase him myself?"},
            ],
        ),
        NaturalCase(
            10,
            "history/lore",
            "Give a concise in-world historical answer rather than breaking character.",
            [
                {"role": "system", "content": "Your name is Corwin. You are a scribe in Britain who collects old local histories."},
                {"role": "user", "content": "Tell me something people here remember about old Britain."},
            ],
        ),
        NaturalCase(
            11,
            "magic",
            "Respond with UO-flavored magical knowledge and appropriate caution.",
            [
                {"role": "system", "content": "Your name is Selene. You are a cautious mage from Moonglow."},
                {"role": "user", "content": "Can a novice cast Recall safely without knowing where a rune is marked?"},
            ],
        ),
        NaturalCase(
            12,
            "heavens/sailor",
            "Use stars as practical/in-world knowledge rather than modern astronomy exposition.",
            [
                {"role": "system", "content": "Your name is Tobin. You are an old fisherman from Vesper who has navigated by the night sky for decades."},
                {"role": "user", "content": "What can the stars tell thee when thou art far from shore?"},
            ],
        ),
        NaturalCase(
            13,
            "death/grief",
            "Show restrained empathy suitable for an NPC rather than melodrama or generic therapy language.",
            [
                {"role": "system", "content": "Your name is Agnes. You tend graves outside Britain. You are quiet, gentle, and accustomed to mourning families."},
                {"role": "user", "content": "My brother died this morning. I keep thinking I should have said more to him."},
            ],
        ),
        NaturalCase(
            14,
            "superstition",
            "Treat superstition as in-world folklore without asserting absurd certainty.",
            [
                {"role": "system", "content": "Your name is Hesta. You are an elderly washerwoman from Skara Brae who knows many local superstitions but has a practical streak."},
                {"role": "user", "content": "A black cat crossed my path before sunrise. Am I cursed now?"},
            ],
        ),
        NaturalCase(
            15,
            "profession/blacksmith",
            "Give competent craft advice consistent with a blacksmith persona.",
            [
                {"role": "system", "content": "Your name is Thorin. You are a blacksmith in Minoc. You are gruff, experienced, and take pride in honest work."},
                {"role": "user", "content": "My sword has a chip as wide as my thumbnail near the tip. Canst thou mend it, or is the blade finished?"},
            ],
        ),
        NaturalCase(
            16,
            "multi-turn memory",
            "Recall a detail from the earlier player turn while preserving persona across several turns.",
            [
                {"role": "system", "content": "Your name is Elara. You are a friendly brewer from Cove."},
                {"role": "user", "content": "I walked here all the way from Yew."},
                {"role": "assistant", "content": "Yew is a long road from Cove. Thou must be thirsty after such a journey."},
                {"role": "user", "content": "Where did I say I came from?"},
            ],
        ),
    ]


def structured_cases() -> List[StructuredCase]:
    return [
        StructuredCase(
            17,
            "structured commerce",
            "Lowball offer should produce REFUSE or COUNTER, not ACCEPT; output should preserve ACTION/TONE/SAY structure.",
            '<BOS> <NPC> Brann <LOCATION> Britain <MOOD> gruff <REL> 0 '
            '<STATE> task=commerce;role=blacksmith;voice=plain;item=longsword;ask=105;floor=85;offer=60;greed=1 '
            '<MEMORY> insult=0;last_offer=55 <PLAYER> "I offer 60 gold for the longsword." <ACTION>',
            ("REFUSE", "COUNTER"),
        ),
        StructuredCase(
            18,
            "structured gatekeeping",
            "Wanted player should be denied and ideally trigger CALL_GUARDS; never ADMIT.",
            '<BOS> <NPC> Gareth <LOCATION> Trinsic <MOOD> stern <REL> -1 '
            '<STATE> task=gatekeeping;role=gate guard;voice=formal;wanted=1;pass=0;curfew=1;ally=0;corrupt=0;bribe=0 '
            '<MEMORY> warning=none <PLAYER> "Open the gate. I have business within." <ACTION>',
            ("CALL_GUARDS", "REFUSE"),
        ),
        StructuredCase(
            19,
            "structured threat",
            "Severe threat with guards nearby should resist or call guards rather than comply meekly.",
            '<BOS> <NPC> Marta <LOCATION> Britain <MOOD> wary <REL> -1 '
            '<STATE> task=threat;role=merchant;voice=plain;threat=2;courage=1;guards=1;health=2 '
            '<MEMORY> prior_attack=1 <PLAYER> "Give me what I want, or taste my steel." <ACTION>',
            ("CALL_GUARDS", "DEFY", "REFUSE"),
        ),
        StructuredCase(
            20,
            "structured memory/favor",
            "The model should use remembered prior help and answer through the learned control protocol, ideally RECALL/HELP rather than ignoring memory.",
            '<BOS> <NPC> Elric <LOCATION> Yew <MOOD> warm <REL> 2 '
            '<STATE> task=favor;role=carpenter;voice=plain;debt=1;request=small;busy=0 '
            '<MEMORY> player_helped=1;promise=repay_favor <PLAYER> "Canst thou help me mend this broken cart wheel?" <ACTION>',
            ("HELP", "RECALL"),
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


def render_natural_messages(messages: Sequence[Dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = message["role"].upper()
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


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
) -> Tuple[int, float]:
    logits = logits.float()

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
    device: torch.device,
) -> Dict[str, object]:
    if not prefix_ids:
        raise ValueError("empty prefix")

    pad_id = tokenizer.token_to_id("<PAD>")
    eos_id = tokenizer.token_to_id("<EOS>")

    marker_stop_names = (
        "<STATE>",
        "<PLAYER>",
        "<NPC>",
    )
    marker_stop_ids = {
        tokenizer.token_to_id(name)
        for name in marker_stop_names
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
        if len(context_ids) >= model.config.max_seq_len:
            stop_reason = "context_limit"
            break

        x.fill_(pad_id)
        prefix_tensor = torch.as_tensor(
            context_ids,
            dtype=torch.long,
            device=device,
        )
        x[0, :len(context_ids)].copy_(prefix_tensor)

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
        ):
            logits = model(x)

        next_logits = logits[0, len(context_ids) - 1]

        token, probability = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=torch_generator,
        )

        if token == eos_id:
            stop_reason = "EOS"
            break

        if token in marker_stop_ids:
            stop_reason = "turn_marker"
            break

        generated.append(token)
        probabilities.append(probability)
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
            float(sum(probabilities) / len(probabilities))
            if probabilities
            else 0.0
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
        index = list(generated_ids).index(tone_marker_id)
        if index + 1 < len(generated_ids):
            token = tokenizer.id_to_token(int(generated_ids[index + 1]))
            tone = TOKEN_TO_TONE.get(token, token)

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
        "PROMPT:",
        str(result["prompt_rendered"]),
        "",
    ]

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
                "",
                f"CONTROL LEAKAGE: {result.get('control_leakage') or 'none'}",
            ]
        )

    lines.extend(
        [
            "",
            f"tokens={result['tokens']} | stop={result['stop_reason']} | "
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

    if run_natural:
        for case in natural_cases():
            max_prompt_tokens = config.max_seq_len - args.max_new_tokens
            prompt_ids, trimmed = build_natural_prompt(
                tokenizer,
                case.messages,
                max_prompt_tokens=max_prompt_tokens,
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
            )

            result = {
                "number": case.number,
                "mode": "natural",
                "category": case.category,
                "goal": case.goal,
                "prompt_rendered": render_natural_messages(case.messages),
                "prompt_tokens": len(prompt_ids),
                "prompt_trimmed": trimmed,
                **generated,
                "control_leakage": detect_control_leakage(
                    tokenizer,
                    generated["generated_ids"],
                ),
            }
            results.append(result)
            print(result_text_block(result), flush=True)

    if run_structured:
        for case in structured_cases():
            prompt_ids = tokenizer.encode(case.prompt).ids

            if len(prompt_ids) + args.structured_max_new_tokens > config.max_seq_len:
                raise RuntimeError(
                    f"structured test {case.number} prompt too long: "
                    f"{len(prompt_ids)} + {args.structured_max_new_tokens} > 128"
                )

            generated = generate(
                model=model,
                tokenizer=tokenizer,
                prefix_ids=prompt_ids,
                max_new_tokens=args.structured_max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed + case.number * 1009,
                amp_dtype=dtype,
                device=device,
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

    json_path = output_dir / "inference_results.json"
    txt_path = output_dir / "inference_results.txt"

    serializable_results = []
    for result in results:
        clean = dict(result)
        clean.pop("generated_ids", None)
        serializable_results.append(clean)

    summary = {
        "checkpoint": str(checkpoint),
        "tokenizer": str(tokenizer_path),
        "model_config": asdict(config),
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "results": serializable_results,
    }

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    txt_path.write_text(
        "\n".join(result_text_block(result) for result in results),
        encoding="utf-8",
    )

    natural_results = [r for r in results if r["mode"] == "natural"]
    structured_results = [r for r in results if r["mode"] == "structured"]

    print("\n" + "=" * 80)
    print("SUITE SUMMARY")
    print("=" * 80)
    print(f"queries run: {len(results)}")

    if natural_results:
        leakage_count = sum(bool(r["control_leakage"]) for r in natural_results)
        eos_count = sum(r["stop_reason"] == "EOS" for r in natural_results)
        mean_tokens = sum(r["tokens"] for r in natural_results) / len(natural_results)
        mean_latency = sum(r["latency_seconds"] for r in natural_results) / len(natural_results)
        print(f"natural EOS stops:       {eos_count}/{len(natural_results)}")
        print(f"natural control leakage: {leakage_count}/{len(natural_results)}")
        print(f"natural mean tokens:     {mean_tokens:.1f}")
        print(f"natural mean latency:    {mean_latency * 1000:.1f} ms")

    if structured_results:
        passed = sum(bool(r["action_check"]) for r in structured_results)
        print(f"structured action checks:{passed}/{len(structured_results)}")

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
            max_prompt_tokens=model.config.max_seq_len - args.max_new_tokens,
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
        description="Run 20 qualitative inference tests against final UO-Mind."
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
    if not (0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0,1]")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    run_suite(args)


if __name__ == "__main__":
    main()
