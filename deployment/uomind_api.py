"""Small programmatic API for the UO-Mind v9 grounded inference bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from tokenizers import Tokenizer

import uomind_infer as runtime
from uomind_grounded_response import grounded_response


class UOMindRuntime:
    """Load once, then call respond() for independent single-pass turns.

    Conversation history is intentionally not accepted or sent. The caller
    may keep a transcript for display, but each turn sees only STATE and the
    current PLAYER text.
    """

    def __init__(
        self,
        bundle: Optional[str | Path] = None,
        checkpoint: Optional[str | Path] = None,
        device: str = "auto",
        precision: str = "auto",
        max_new_tokens: int = 32,
        max_system_tokens: int = 40,
    ):
        default_bundle = Path(__file__).resolve().parents[1]
        self.bundle = Path(bundle or default_bundle).resolve()
        checkpoint_path = Path(checkpoint) if checkpoint else Path("model/v9_base_finetune.pt")
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.bundle / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        tokenizer_path = self.bundle / "tokenizer" / "tokenizer.json"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if not tokenizer_path.is_file():
            raise FileNotFoundError(tokenizer_path)

        self.device = runtime.choose_device(device)
        self.precision = runtime.choose_precision(self.device, precision)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        runtime.validate_tokenizer(self.tokenizer)
        self.model, _, self.config = runtime.load_model(
            checkpoint_path,
            self.tokenizer,
            self.device,
            self.precision,
            attention_backend=None,
        )
        self.max_new_tokens = max_new_tokens
        self.max_system_tokens = max_system_tokens

    def respond(
        self,
        state: str,
        player: str,
        raw_model: bool = False,
        temperature: float = 0.0,
        top_p: float = 0.90,
    ) -> dict:
        generated = runtime.generate(
            self.model,
            self.tokenizer,
            state=state,
            player=player,
            device=self.device,
            precision=self.precision,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_system_tokens=self.max_system_tokens,
        )
        raw_text = generated["text"]
        text = raw_text if raw_model else grounded_response(
            state, player, lambda _state, _player: raw_text
        )
        return {
            **generated,
            "text": text,
            "raw_text": raw_text,
            "grounded_applied": text != raw_text,
        }


__all__ = ["UOMindRuntime"]
