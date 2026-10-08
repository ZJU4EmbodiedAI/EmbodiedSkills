"""Training objectives used by the spatial and continuous world models."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

CANDIDATES = 5


def _finite_tensor(value, *, name: str) -> None:
    import torch

    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"public_mt50_ranker_{name}_must_be_finite")


def candidate_ranking_loss(
    scores, targets, *, pairwise_weight: float = 0.25, tie_epsilon: float = 0.001
):
    """Listwise soft-label loss with an optional target-ordered pairwise term."""
    import torch
    from torch.nn import functional

    if scores.ndim != 2 or scores.shape[1] != CANDIDATES or targets.shape != scores.shape:
        raise ValueError("public_mt50_scores_and_targets_must_be_B_5")
    _finite_tensor(scores, name="scores")
    _finite_tensor(targets, name="targets")
    if not math.isfinite(pairwise_weight) or pairwise_weight < 0:
        raise ValueError("public_mt50_pairwise_weight_must_be_nonnegative")
    if not math.isfinite(tie_epsilon) or tie_epsilon < 0:
        raise ValueError("public_mt50_tie_epsilon_must_be_nonnegative")
    listwise = -(targets * torch.log_softmax(scores, dim=1)).sum(dim=1).mean()
    if pairwise_weight == 0:
        pairwise = scores.new_zeros(())
    else:
        upper = torch.triu_indices(CANDIDATES, CANDIDATES, offset=1, device=scores.device)
        score_delta = scores[:, upper[0]] - scores[:, upper[1]]
        target_delta = targets[:, upper[0]] - targets[:, upper[1]]
        mask = target_delta.abs() > float(tie_epsilon)
        if bool(mask.any()):
            labels = (target_delta[mask] > 0).to(scores.dtype)
            pairwise = functional.binary_cross_entropy_with_logits(score_delta[mask], labels)
        else:
            pairwise = scores.new_zeros(())
    total = listwise + float(pairwise_weight) * pairwise
    return {"loss": total, "listwise_loss": listwise, "pairwise_loss": pairwise}


def direct_relative_branch_loss(
    predicted_advantage, branch_return, paired_direct_reference_index, *, delta: float = 0.25
):
    """Supervise a candidate's actual return relative to paired direct pi0.5.

    ``branch_return`` is strictly a training label.  The reference index is a
    protocol-routing value: it identifies which *currently permuted* candidate
    came from the ordinary direct pi0.5 request.  It is never inferred from a
    candidate slot and is not an environment observation.  At deployment the
    public candidate protocol fixes this reference to physical candidate zero.
    """
    import torch
    from torch.nn import functional

    if predicted_advantage.ndim != 2 or predicted_advantage.shape[1] != CANDIDATES:
        raise ValueError("public_mt50_relative_advantage_must_be_B_5")
    returns = torch.as_tensor(
        branch_return, dtype=predicted_advantage.dtype, device=predicted_advantage.device
    )
    reference = torch.as_tensor(
        paired_direct_reference_index, dtype=torch.long, device=predicted_advantage.device
    )
    if returns.shape != predicted_advantage.shape:
        raise ValueError("public_mt50_relative_branch_return_must_be_B_5")
    if reference.shape != (len(predicted_advantage),):
        raise ValueError("public_mt50_relative_reference_index_must_be_B")
    if bool((reference < 0).any()) or bool((reference >= CANDIDATES).any()):
        raise ValueError("public_mt50_relative_reference_index_out_of_range")
    if not math.isfinite(float(delta)) or delta <= 0:
        raise ValueError("public_mt50_relative_huber_delta_must_be_positive")
    _finite_tensor(predicted_advantage, name="relative_advantage")
    _finite_tensor(returns, name="relative_branch_return")
    baseline = returns.gather(1, reference[:, None])
    target = returns - baseline
    gathered_prediction = predicted_advantage.gather(1, reference[:, None])
    if not bool(torch.allclose(gathered_prediction, torch.zeros_like(gathered_prediction))):
        raise RuntimeError("public_mt50_relative_model_reference_must_be_zero")
    loss = functional.huber_loss(predicted_advantage, target, delta=float(delta))
    return {"loss": loss, "target": target}


def _success_pairwise_loss(torch, logits, success):
    """Binary pairwise loss on every success/failure candidate pair."""
    upper = torch.triu_indices(5, 5, offset=1, device=logits.device)
    difference = success[:, upper[0]] - success[:, upper[1]]
    mask = difference != 0
    if not bool(mask.any()):
        return logits.sum() * 0.0
    prediction = logits[:, upper[0]] - logits[:, upper[1]]
    target = (difference[mask] > 0).to(logits.dtype)
    return torch.nn.functional.binary_cross_entropy_with_logits(prediction[mask], target)


def _listwise(
    predicted: torch.Tensor, target: torch.Tensor, temperature: float = 0.25
) -> torch.Tensor:
    soft = F.softmax(target / temperature, dim=-1)
    return -(soft * F.log_softmax(predicted / temperature, dim=-1)).sum(-1).mean()


def _loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    success_priority: float,
    positive_weight: float,
    expert_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    delta = F.smooth_l1_loss(output["predicted_delta_standard"], batch["delta_standard"])
    relative_delta = F.smooth_l1_loss(
        output["predicted_delta_standard"]
        - output["predicted_delta_standard"].mean(1, keepdim=True),
        batch["delta_standard"] - batch["delta_standard"].mean(1, keepdim=True),
    )
    future_cosine = 1.0 - (output["predicted_future"] * batch["future"]).sum(-1).mean()
    reward = F.smooth_l1_loss(output["macro_reward_std"], batch["return_standard"])
    value = F.binary_cross_entropy_with_logits(
        output["terminal_value_logit"],
        batch["success"],
        pos_weight=torch.tensor(positive_weight, device=batch["success"].device),
    )
    branch_target = batch["return_standard"] + success_priority * batch["success"]
    rank_branch = _listwise(output["score"], branch_target)
    rank_expert = _listwise(output["score"], batch["expert_standard"])
    total = (
        delta
        + relative_delta
        + future_cosine
        + reward
        + 0.25 * value
        + rank_branch
        + expert_weight * rank_expert
    )
    return (
        total,
        {
            "delta": float(delta.detach()),
            "relative_delta": float(relative_delta.detach()),
            "future_cosine": float(future_cosine.detach()),
            "reward": float(reward.detach()),
            "value": float(value.detach()),
            "rank_branch": float(rank_branch.detach()),
            "rank_expert": float(rank_expert.detach()),
        },
    )
