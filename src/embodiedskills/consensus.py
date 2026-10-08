"""Candidate-relative consensus over three predicted consequences."""

from __future__ import annotations

import math

MEMBERS = 3
CANDIDATES = 5
LATENT_DIM = 32
FEATURES = 69
RESIDUAL_SCALE = 0.25


def fit_effect_rms(torch, pooled_delta, fit_index):
    """Fit-only per-dimension scale; candidates 1..4, never outcomes."""
    if pooled_delta.ndim != 4 or tuple(pooled_delta.shape[1:]) != (MEMBERS, CANDIDATES, LATENT_DIM):
        raise ValueError("future_consensus_pooled_delta_shape_invalid")
    selected = pooled_delta[fit_index]
    effect = selected[:, :, 1:] - selected[:, :, :1]
    value = effect.square().mean(dim=(0, 1, 2)).sqrt().clamp_min(1e-05)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("future_consensus_effect_rms_nonfinite")
    return value


def _check_shape(member_scores, pooled_delta) -> None:
    if member_scores.ndim != 3 or tuple(member_scores.shape[1:]) != (MEMBERS, CANDIDATES):
        raise ValueError("future_consensus_member_scores_shape_invalid")
    if tuple(pooled_delta.shape) != (member_scores.shape[0], MEMBERS, CANDIDATES, LATENT_DIM):
        raise ValueError("future_consensus_pooled_delta_shape_invalid")
    if not bool(member_scores.isfinite().all()) or not bool(pooled_delta.isfinite().all()):
        raise ValueError("future_consensus_input_nonfinite")


def build_future_consensus_head(effect_rms):
    import torch
    from torch import nn

    rms = torch.as_tensor(effect_rms, dtype=torch.float32).clone().detach()
    if (
        tuple(rms.shape) != (LATENT_DIM,)
        or not bool(torch.isfinite(rms).all())
        or bool((rms < 1e-05).any())
    ):
        raise ValueError("future_consensus_effect_rms_invalid")

    class FutureConsensusHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("effect_rms", rms)
            self.network = nn.Sequential(
                nn.LayerNorm(FEATURES),
                nn.Linear(FEATURES, 32),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.GELU(),
                nn.Linear(16, 4),
            )

        def forward(self, member_scores, pooled_delta):
            _check_shape(member_scores, pooled_delta)
            relative = member_scores - member_scores[:, :, :1]
            score_rms = (
                relative[:, :, 1:].square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-06)
            )
            normalized_scores = (relative / score_rms).transpose(1, 2)
            effect = (pooled_delta - pooled_delta[:, :, :1]) / self.effect_rms[None, None, None]
            mean_effect = effect.mean(dim=1)
            spread = effect.std(dim=1, unbiased=False)
            unit = torch.nn.functional.normalize(effect, dim=-1, eps=1e-06)
            agreement = (
                (unit[:, 0] * unit[:, 1]).sum(dim=-1)
                + (unit[:, 0] * unit[:, 2]).sum(dim=-1)
                + (unit[:, 1] * unit[:, 2]).sum(dim=-1)
            ) / 3
            effect_magnitude = effect.norm(dim=-1).mean(dim=1) / math.sqrt(LATENT_DIM)
            features = torch.cat(
                (
                    normalized_scores,
                    mean_effect,
                    spread,
                    agreement[..., None],
                    effect_magnitude[..., None],
                ),
                dim=-1,
            )
            logits = self.network(features)
            probabilities = logits.softmax(dim=-1)
            utility = (probabilities[..., 1] - probabilities[..., 2]).clone()
            utility[:, 0] = 0
            conservative = member_scores.mean(dim=1) - member_scores.std(dim=1, unbiased=False)
            advantage = conservative - conservative[:, :1]
            base_rms = advantage[:, 1:].square().mean(dim=1, keepdim=True).sqrt()
            base = advantage / base_rms.clamp_min(1e-06)
            final = base + RESIDUAL_SCALE * utility
            final = final.clone()
            final[:, 0] = 0
            if not bool(final.isfinite().all()):
                raise ValueError("future_consensus_output_nonfinite")
            return {
                "features": features,
                "logits": logits,
                "probabilities": probabilities,
                "utilities": utility,
                "base": base,
                "scores": final,
            }

    return FutureConsensusHead()


def joint_targets(torch, success):
    if success.ndim != 2 or success.shape[1] != CANDIDATES:
        raise ValueError("future_consensus_success_shape_invalid")
    direct = success[:, :1].to(torch.long)
    return 2 * direct + success.to(torch.long)
