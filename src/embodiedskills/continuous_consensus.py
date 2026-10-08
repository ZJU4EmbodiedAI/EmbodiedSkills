"""Candidate-relative consensus over three predicted consequences."""

from __future__ import annotations

import math

MEMBERS = 3
CANDIDATES = 5
SOURCE_LATENT_DIM = 1024
PROJECTED_DIM = 32
JOINT_CLASSES = 4
FEATURES = 69
RESIDUAL_SCALE = 0.25


def _check_source_shapes(member_scores, predicted_future) -> None:
    if member_scores.ndim != 3 or tuple(member_scores.shape[1:]) != (MEMBERS, CANDIDATES):
        raise ValueError("mt50_future_critic_member_scores_shape_invalid")
    if tuple(predicted_future.shape) != (
        member_scores.shape[0],
        MEMBERS,
        CANDIDATES,
        SOURCE_LATENT_DIM,
    ):
        raise ValueError("mt50_future_critic_predicted_future_shape_invalid")
    if not bool(member_scores.isfinite().all()) or not bool(predicted_future.isfinite().all()):
        raise ValueError("mt50_future_critic_source_nonfinite")


def fit_effect_projection(
    torch, predicted_future, fit_indices, *, batch_rows: int = 64, device: str = "cpu"
):
    """Fit RMS normalization and a deterministic 32-D PCA on fit rows only."""
    if predicted_future.ndim != 4 or tuple(predicted_future.shape[1:]) != (
        MEMBERS,
        CANDIDATES,
        SOURCE_LATENT_DIM,
    ):
        raise ValueError("mt50_future_critic_projection_source_shape_invalid")
    indices = torch.as_tensor(fit_indices, dtype=torch.long, device="cpu")
    if indices.ndim != 1 or not len(indices):
        raise ValueError("mt50_future_critic_projection_fit_indices_invalid")
    if int(indices.min()) < 0 or int(indices.max()) >= len(predicted_future):
        raise ValueError("mt50_future_critic_projection_fit_index_out_of_range")
    if isinstance(batch_rows, bool) or batch_rows <= 0:
        raise ValueError("mt50_future_critic_projection_batch_invalid")
    compute_device = torch.device(device)
    square_sum = torch.zeros(SOURCE_LATENT_DIM, dtype=torch.float64, device=compute_device)
    count = 0
    for start in range(0, len(indices), batch_rows):
        rows = indices[start : start + batch_rows]
        future = predicted_future[rows].to(dtype=torch.float64, device=compute_device)
        effect = future[:, :, 1:] - future[:, :, :1]
        square_sum += effect.square().sum(dim=(0, 1, 2))
        count += int(effect.shape[0] * effect.shape[1] * effect.shape[2])
    if count <= PROJECTED_DIM:
        raise ValueError("mt50_future_critic_projection_too_few_effects")
    effect_rms = (square_sum / count).sqrt().clamp_min(1e-05)
    effect_sum = torch.zeros(SOURCE_LATENT_DIM, dtype=torch.float64, device=compute_device)
    for start in range(0, len(indices), batch_rows):
        rows = indices[start : start + batch_rows]
        future = predicted_future[rows].to(dtype=torch.float64, device=compute_device)
        effect = ((future[:, :, 1:] - future[:, :, :1]) / effect_rms).reshape(-1, SOURCE_LATENT_DIM)
        effect_sum += effect.sum(dim=0)
    effect_mean = effect_sum / count
    covariance_sum = torch.zeros(
        (SOURCE_LATENT_DIM, SOURCE_LATENT_DIM), dtype=torch.float64, device=compute_device
    )
    for start in range(0, len(indices), batch_rows):
        rows = indices[start : start + batch_rows]
        future = predicted_future[rows].to(dtype=torch.float64, device=compute_device)
        effect = ((future[:, :, 1:] - future[:, :, :1]) / effect_rms).reshape(-1, SOURCE_LATENT_DIM)
        centered = effect - effect_mean
        covariance_sum.addmm_(centered.T, centered)
    covariance = covariance_sum / max(count - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)[:PROJECTED_DIM]
    basis = eigenvectors[:, order].contiguous()
    selected_values = eigenvalues[order]
    if not bool(torch.isfinite(selected_values).all()) or bool((selected_values < -1e-08).any()):
        raise ValueError("mt50_future_critic_projection_eigenvalues_invalid")
    pivots = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[pivots, torch.arange(PROJECTED_DIM)])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    basis = basis * signs[None]
    values = (effect_rms, effect_mean, basis, selected_values)
    if not all(bool(value.isfinite().all()) for value in values):
        raise ValueError("mt50_future_critic_projection_nonfinite")
    return tuple(value.float().cpu() for value in values)


def build_future_consequence_critic(*, effect_rms, effect_mean, pca_basis):
    """Build the fixed 69-feature single counterfactual critic."""
    import torch
    from torch import nn

    rms = torch.as_tensor(effect_rms, dtype=torch.float32).detach().clone()
    mean = torch.as_tensor(effect_mean, dtype=torch.float32).detach().clone()
    basis = torch.as_tensor(pca_basis, dtype=torch.float32).detach().clone()
    if (
        tuple(rms.shape) != (SOURCE_LATENT_DIM,)
        or tuple(mean.shape) != (SOURCE_LATENT_DIM,)
        or tuple(basis.shape) != (SOURCE_LATENT_DIM, PROJECTED_DIM)
        or (not bool(rms.isfinite().all()))
        or (not bool(mean.isfinite().all()))
        or (not bool(basis.isfinite().all()))
        or bool((rms < 1e-05).any())
    ):
        raise ValueError("mt50_future_critic_projection_buffers_invalid")

    class MT50FutureConsequenceCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("effect_rms", rms)
            self.register_buffer("effect_mean", mean)
            self.register_buffer("pca_basis", basis)
            self.network = nn.Sequential(
                nn.LayerNorm(FEATURES),
                nn.Linear(FEATURES, 32),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.GELU(),
                nn.Linear(16, JOINT_CLASSES),
            )

        def build_features(self, member_scores, predicted_future):
            _check_source_shapes(member_scores, predicted_future)
            relative = member_scores - member_scores[:, :, :1]
            score_rms = (
                relative[:, :, 1:].square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-06)
            )
            normalized_scores = (relative / score_rms).transpose(1, 2)
            raw_effect = predicted_future - predicted_future[:, :, :1]
            normalized_effect = raw_effect / self.effect_rms[None, None, None]
            projected = torch.matmul(
                normalized_effect - self.effect_mean[None, None, None], self.pca_basis
            )
            mean_effect = projected.mean(dim=1)
            spread = projected.std(dim=1, unbiased=False)
            unit = torch.nn.functional.normalize(projected, dim=-1, eps=1e-06)
            agreement = (
                (unit[:, 0] * unit[:, 1]).sum(dim=-1)
                + (unit[:, 0] * unit[:, 2]).sum(dim=-1)
                + (unit[:, 1] * unit[:, 2]).sum(dim=-1)
            ) / 3.0
            magnitude = projected.norm(dim=-1).mean(dim=1) / math.sqrt(PROJECTED_DIM)
            features = torch.cat(
                (
                    normalized_scores,
                    mean_effect,
                    spread,
                    agreement[..., None],
                    magnitude[..., None],
                ),
                dim=-1,
            )
            if tuple(features.shape) != (member_scores.shape[0], CANDIDATES, FEATURES) or not bool(
                features.isfinite().all()
            ):
                raise ValueError("mt50_future_critic_features_invalid")
            return features

        def forward(self, member_scores, predicted_future):
            features = self.build_features(member_scores, predicted_future)
            logits = self.network(features)
            probabilities = logits.softmax(dim=-1)
            utility = (probabilities[..., 1] - probabilities[..., 2]).clone()
            utility[:, 0] = 0.0
            conservative = member_scores.mean(dim=1) - member_scores.std(dim=1, unbiased=False)
            advantage = conservative - conservative[:, :1]
            base_rms = advantage[:, 1:].square().mean(dim=1, keepdim=True).sqrt()
            base = advantage / base_rms.clamp_min(1e-06)
            scores = base + RESIDUAL_SCALE * utility
            scores = scores.clone()
            scores[:, 0] = 0.0
            if not bool(scores.isfinite().all()):
                raise ValueError("mt50_future_critic_output_nonfinite")
            return {
                "features": features,
                "logits": logits,
                "probabilities": probabilities,
                "utilities": utility,
                "base": base,
                "scores": scores,
            }

    return MT50FutureConsequenceCritic()


def joint_targets(torch, success):
    """Class IDs ordered as 00, 01, 10, 11 relative to candidate zero."""
    if success.ndim != 2 or success.shape[1] != CANDIDATES:
        raise ValueError("mt50_future_critic_success_shape_invalid")
    direct = success[:, :1].to(torch.long)
    return 2 * direct + success.to(torch.long)
