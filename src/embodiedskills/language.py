"""Frozen MiniLM instruction embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

DEFAULT_LANGUAGE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def masked_mean_pool(last_hidden_state, attention_mask):
    """Mean-pool token states while excluding padding tokens exactly.

    ``attention_mask`` has shape ``[batch, sequence]``.  It is expanded only
    across the hidden dimension, used in both numerator and denominator, and
    clamped to make the helper safe for a malformed all-padding input.
    """
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def l2_normalize(embeddings):
    """Apply the sentence-transformer L2-normalization contract."""
    denominator = embeddings.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return embeddings / denominator


class LanguageEncoder:
    """Load a frozen Hugging Face text encoder and produce numpy embeddings.

    Loading is intentionally offline-only.  Downloading a model is an explicit
    setup action; training and deployment must fail clearly rather than making
    an implicit network request.
    """

    def __init__(
        self,
        model_id: str | Path = DEFAULT_LANGUAGE_MODEL,
        *,
        device: str = "cpu",
        max_length: int = 64,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length_must_be_positive")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "LanguageEncoder requires the local extras: pip install -e '.[encoders]'"
            ) from error
        self.torch = torch
        self.model_id = str(model_id)
        self.device = device
        self.max_length = int(max_length)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
            self.model = AutoModel.from_pretrained(self.model_id, local_files_only=True)
        except OSError as error:
            raise FileNotFoundError(
                f"offline_language_model_not_found:{self.model_id}; place it in the Hugging Face cache before running extraction or deployment"
            ) from error
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.feature_dim = int(self.model.config.hidden_size)

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        """Embed text batches with tokenizer padding and attention-mask pooling."""
        if batch_size <= 0:
            raise ValueError("batch_size_must_be_positive")
        values = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in values):
            raise ValueError("instructions_must_be_nonempty_strings")
        if not values:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        batches: list[np.ndarray] = []
        for start in range(0, len(values), batch_size):
            tokens = self.tokenizer(
                values[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with self.torch.inference_mode():
                output = self.model(**tokens)
                pooled = masked_mean_pool(output.last_hidden_state, tokens["attention_mask"])
                pooled = l2_normalize(pooled)
            batches.append(pooled.float().cpu().numpy())
        return np.concatenate(batches, axis=0).astype(np.float32, copy=False)
