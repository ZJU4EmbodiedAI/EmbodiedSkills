"""Spatial context encoder, action-conditioned latent predictor and goal value head."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

CLIPORT_ACTION_DIM = WAM_ACTION_DIM = 20
CLIPORT_CANDIDATES = WAM_CANDIDATES = 5
WAM_VISUAL_DIM = 1024
WAM_VISUAL_TOKENS = 64
WAM_CONTEXT_DIM = 768
WAM_LATENT_DIM = 32
WAM_HIDDEN_DIM = 128
WAM_PROJECTION_SEED = 20260919
CONSEQUENCE_VALUE_HIDDEN_DIM = 64
CONSEQUENCE_VALUE_ARCHITECTURES = ("mean_max_delta",)
CONSEQUENCE_DECISION_BASE_MODES = ("direct_candidate",)
MAX_RESIDUAL_SCALE = 8.0


def build_cliport_mt50_transfer_ranker(
    *,
    visual_dim: int = 1024,
    language_dim: int = 384,
    hidden_dim: int = 256,
    state_mean: Sequence[float] | None = None,
    state_std: Sequence[float] | None = None,
    action_mean: Sequence[float] | None = None,
    action_std: Sequence[float] | None = None,
    direct_relative: bool = False,
    direct_relative_score_weight: float = 1.0,
):
    """Build the MT50 scorer with the minimal CLIPort action/state boundary."""
    import torch
    from torch import nn

    if visual_dim <= 0 or language_dim <= 0 or hidden_dim < 32 or hidden_dim % 8:
        raise ValueError("cliport_mt50_transfer_dimensions_invalid")

    class CLIPortMT50TransferRanker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual_dim = int(visual_dim)
            self.language_dim = int(language_dim)
            self.hidden_dim = int(hidden_dim)
            self.direct_relative = bool(direct_relative)
            if (
                not math.isfinite(float(direct_relative_score_weight))
                or direct_relative_score_weight < 0
            ):
                raise ValueError("cliport_mt50_direct_relative_weight_invalid")
            self.direct_relative_score_weight = float(direct_relative_score_weight)
            self.visual_projection = nn.Sequential(
                nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim), nn.GELU()
            )
            self.language_encoder = nn.Sequential(
                nn.LayerNorm(language_dim),
                nn.Linear(language_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.state_encoder = nn.Sequential(
                nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.progress_encoder = nn.Sequential(
                nn.Linear(5, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, hidden_dim)
            )
            self.native_pick_place_encoder = nn.Sequential(
                nn.LayerNorm(CLIPORT_ACTION_DIM),
                nn.Linear(CLIPORT_ACTION_DIM, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.direct_relative:
                self.relative_pick_place_encoder = nn.Sequential(
                    nn.LayerNorm(CLIPORT_ACTION_DIM),
                    nn.Linear(CLIPORT_ACTION_DIM, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            self.context_fusion = nn.Sequential(
                nn.LayerNorm(hidden_dim * 4),
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.cross_attention = nn.MultiheadAttention(
                hidden_dim, num_heads=8, dropout=0.05, batch_first=True
            )
            self.candidate_fusion = nn.Sequential(
                nn.LayerNorm(hidden_dim * 4),
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.score_head = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if self.direct_relative:
                self.relative_value_head = nn.Sequential(
                    nn.LayerNorm(hidden_dim * 3),
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )

            def statistic(
                values: Sequence[float] | None, size: int, default: float
            ) -> torch.Tensor:
                array = (
                    np.full(size, default, dtype=np.float32)
                    if values is None
                    else np.asarray(values, dtype=np.float32)
                )
                if array.shape != (size,) or not np.isfinite(array).all():
                    raise ValueError("cliport_mt50_transfer_statistics_invalid")
                return torch.from_numpy(array.copy())

            self.register_buffer("state_mean", statistic(state_mean, 4, 0.0))
            self.register_buffer("state_std", statistic(state_std, 4, 1.0))
            self.register_buffer("action_mean", statistic(action_mean, CLIPORT_ACTION_DIM, 0.0))
            self.register_buffer("action_std", statistic(action_std, CLIPORT_ACTION_DIM, 1.0))
            if bool((self.state_std <= 0).any()) or bool((self.action_std <= 0).any()):
                raise ValueError("cliport_mt50_transfer_std_must_be_positive")

        def forward(
            self,
            visual_features,
            state4,
            language_features,
            progress,
            candidate_actions,
            *,
            direct_reference_index=None,
            return_components: bool = False,
        ):
            if visual_features.ndim == 2:
                visual_features = visual_features[:, None]
            batch = visual_features.shape[0]
            expected = {
                "visual": (batch, visual_features.shape[1], self.visual_dim),
                "state": (batch, 4),
                "language": (batch, self.language_dim),
                "progress": (batch, 1),
                "actions": (batch, CLIPORT_CANDIDATES, CLIPORT_ACTION_DIM),
            }
            if visual_features.shape != expected["visual"]:
                raise ValueError("cliport_mt50_visual_features_shape_invalid")
            if state4.shape != expected["state"]:
                raise ValueError("cliport_mt50_state4_shape_invalid")
            if language_features.shape != expected["language"]:
                raise ValueError("cliport_mt50_language_features_shape_invalid")
            if progress.shape != expected["progress"]:
                raise ValueError("cliport_mt50_progress_shape_invalid")
            if candidate_actions.shape != expected["actions"]:
                raise ValueError("cliport_mt50_candidate_actions_shape_invalid")
            for name, value in (
                ("visual", visual_features),
                ("state4", state4),
                ("language", language_features),
                ("progress", progress),
                ("actions", candidate_actions),
            ):
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"cliport_mt50_{name}_must_be_finite")
            if bool((progress < 0).any()) or bool((progress > 1).any()):
                raise ValueError("cliport_mt50_progress_must_be_in_0_1")
            state = (state4 - self.state_mean) / self.state_std
            actions = (candidate_actions - self.action_mean) / self.action_std
            visual_tokens = self.visual_projection(visual_features)
            visual_global = visual_tokens.mean(dim=1)
            language = self.language_encoder(language_features)
            state_embedding = self.state_encoder(state)
            progress_features = torch.cat(
                (
                    progress,
                    progress.square(),
                    torch.sin(math.pi * progress),
                    torch.cos(math.pi * progress),
                    torch.sin(2.0 * math.pi * progress),
                ),
                dim=1,
            )
            progress_embedding = self.progress_encoder(progress_features)
            context = self.context_fusion(
                torch.cat((visual_global, language, state_embedding, progress_embedding), dim=1)
            )
            action_embedding = self.native_pick_place_encoder(actions)
            reference = None
            if self.direct_relative:
                if direct_reference_index is None:
                    reference = torch.zeros(batch, dtype=torch.long, device=actions.device)
                else:
                    reference = torch.as_tensor(
                        direct_reference_index, dtype=torch.long, device=actions.device
                    )
                if reference.shape != (batch,):
                    raise ValueError("cliport_mt50_direct_reference_index_must_be_B")
                if bool((reference < 0).any() | (reference >= CLIPORT_CANDIDATES).any()):
                    raise ValueError("cliport_mt50_direct_reference_index_out_of_range")
                reference_action = actions[torch.arange(batch, device=actions.device), reference][
                    :, None
                ]
                action_embedding = action_embedding + self.relative_pick_place_encoder(
                    actions - reference_action
                )
            query = action_embedding + context[:, None]
            attended, _ = self.cross_attention(
                query, visual_tokens, visual_tokens, need_weights=False
            )
            candidates = self.candidate_fusion(
                torch.cat(
                    (
                        action_embedding,
                        attended,
                        context[:, None].expand(-1, CLIPORT_CANDIDATES, -1),
                        action_embedding * attended,
                    ),
                    dim=-1,
                )
            )
            set_mean = candidates.mean(dim=1, keepdim=True)
            score_features = torch.cat(
                (candidates, candidates - set_mean, set_mean.expand(-1, CLIPORT_CANDIDATES, -1)),
                dim=-1,
            )
            branch_scores = self.score_head(score_features).squeeze(-1)
            relative_advantage = None
            scores = branch_scores
            if self.direct_relative:
                assert reference is not None
                raw_relative = self.relative_value_head(score_features).squeeze(-1)
                relative_advantage = raw_relative - raw_relative.gather(1, reference[:, None])
                scores = branch_scores + self.direct_relative_score_weight * relative_advantage
            if scores.shape != (batch, CLIPORT_CANDIDATES):
                raise RuntimeError("cliport_mt50_internal_scores_shape_invalid")
            if return_components:
                return {
                    "scores": scores,
                    "branch_scores": branch_scores,
                    "relative_advantage": relative_advantage,
                    "action_embedding": action_embedding,
                    "score_features": score_features,
                }
            return scores

    return CLIPortMT50TransferRanker()


def fixed_visual_delta_projection() -> np.ndarray:
    """Return the fixed orthonormal 1024 -> 32 consequence projection.

    The projection defines the transition target space and is stored with
    each member's weights.
    """
    generator = np.random.default_rng(WAM_PROJECTION_SEED)
    raw = generator.normal(size=(WAM_VISUAL_DIM, WAM_LATENT_DIM))
    orthonormal, _ = np.linalg.qr(raw)
    return orthonormal[:, :WAM_LATENT_DIM].astype(np.float32)


def _as_statistic(value: Sequence[float] | np.ndarray, *, name: str, positive: bool = False):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (WAM_LATENT_DIM,) or not np.isfinite(array).all():
        raise ValueError(f"cliport_wam_sidecar_{name}_invalid")
    if positive and bool((array <= 0).any()):
        raise ValueError(f"cliport_wam_sidecar_{name}_must_be_positive")
    return array


def build_latent_wam(
    *,
    target_mean: Sequence[float] | np.ndarray,
    target_std: Sequence[float] | np.ndarray,
    projection: np.ndarray | None = None,
    hidden_dim: int = WAM_HIDDEN_DIM,
):
    """Build the action-conditioned token-delta predictor.

    ``candidate_context`` is the frozen ranker's detached ``score_features``:
    it is upstream of both ``score_head`` and ``relative_value_head`` and is
    derived solely from the five declared deployment inputs.
    """
    import torch
    from torch import nn

    hidden_dim = int(hidden_dim)
    if hidden_dim <= 0:
        raise ValueError("cliport_wam_sidecar_hidden_dim_invalid")
    mean = _as_statistic(target_mean, name="target_mean")
    std = _as_statistic(target_std, name="target_std", positive=True)
    matrix = (
        fixed_visual_delta_projection()
        if projection is None
        else np.asarray(projection, dtype=np.float32)
    )
    if matrix.shape != (WAM_VISUAL_DIM, WAM_LATENT_DIM):
        raise ValueError("cliport_wam_sidecar_projection_shape_invalid")
    if not np.isfinite(matrix).all():
        raise ValueError("cliport_wam_sidecar_projection_nonfinite")

    class CLIPortLatentWAM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.scene_projection = nn.Sequential(
                nn.LayerNorm(WAM_VISUAL_DIM), nn.Linear(WAM_VISUAL_DIM, hidden_dim), nn.GELU()
            )
            self.candidate_film = nn.Linear(WAM_CONTEXT_DIM, hidden_dim * 2)
            self.delta_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, WAM_LATENT_DIM),
            )
            self.register_buffer("target_mean", torch.from_numpy(mean.copy()))
            self.register_buffer("target_std", torch.from_numpy(std.copy()))
            self.register_buffer("visual_delta_projection", torch.from_numpy(matrix.copy()))

        def normalize_target(self, value):
            return (value - self.target_mean) / self.target_std

        def denormalize_prediction(self, value):
            return value * self.target_std + self.target_mean

        def forward(self, current_visual, candidate_context):
            batch = current_visual.shape[0]
            if current_visual.shape != (batch, WAM_VISUAL_TOKENS, WAM_VISUAL_DIM):
                raise ValueError("cliport_wam_sidecar_visual_shape_invalid")
            if candidate_context.shape != (batch, WAM_CANDIDATES, WAM_CONTEXT_DIM):
                raise ValueError("cliport_wam_sidecar_candidate_context_shape_invalid")
            if not bool(torch.isfinite(current_visual).all()):
                raise ValueError("cliport_wam_sidecar_visual_nonfinite")
            if not bool(torch.isfinite(candidate_context).all()):
                raise ValueError("cliport_wam_sidecar_candidate_context_nonfinite")
            scene = self.scene_projection(current_visual)[:, None]
            gamma, beta = self.candidate_film(candidate_context).chunk(2, dim=-1)
            hidden = scene * (1.0 + 0.1 * torch.tanh(gamma[:, :, None]))
            hidden = hidden + beta[:, :, None]
            normalized_delta = self.delta_head(hidden)
            raw_delta = self.denormalize_prediction(normalized_delta)
            current_projected = torch.matmul(current_visual, self.visual_delta_projection)
            predicted_future = current_projected[:, None] + raw_delta
            return {
                "predicted_visual_delta_normalized": normalized_delta,
                "predicted_visual_delta": raw_delta,
                "predicted_future_visual": predicted_future,
                "predicted_pooled_consequence": predicted_future.mean(dim=2),
            }

    return CLIPortLatentWAM()


def _validate_deployment_tensors(
    visual_features, state4, language_features, progress, candidate_actions
) -> None:
    batch = visual_features.shape[0]
    expected = {
        "visual": (batch, WAM_VISUAL_TOKENS, WAM_VISUAL_DIM),
        "state4": (batch, 4),
        "language": (batch, 384),
        "progress": (batch, 1),
        "actions": (batch, WAM_CANDIDATES, WAM_ACTION_DIM),
    }
    observed = {
        "visual": tuple(visual_features.shape),
        "state4": tuple(state4.shape),
        "language": tuple(language_features.shape),
        "progress": tuple(progress.shape),
        "actions": tuple(candidate_actions.shape),
    }
    for name, expected_shape in expected.items():
        if observed[name] != expected_shape:
            raise ValueError(
                f"cliport_consequence_lookahead_{name}_shape_invalid:{observed[name]}!={expected_shape}"
            )
    for name, value in (
        ("visual", visual_features),
        ("state4", state4),
        ("language", language_features),
        ("progress", progress),
        ("actions", candidate_actions),
    ):
        if not bool(value.isfinite().all()):
            raise ValueError(f"cliport_consequence_lookahead_{name}_nonfinite")
    if bool((progress < 0).any()) or bool((progress > 1).any()):
        raise ValueError("cliport_consequence_lookahead_progress_out_of_range")


def build_goal_conditioned_consequence_value_head(
    *, hidden_dim: int = CONSEQUENCE_VALUE_HIDDEN_DIM, architecture: str = "mean_max_delta"
):
    """Build a shared candidate-wise value head over predicted consequences.

    The candidate-specific inputs are the predicted future pooled latent, its
    predicted delta from the current latent, and its difference from candidate
    zero.  Language is shared across candidates and can only affect selection
    through its interaction with those predicted consequence features.
    """
    if hidden_dim <= 0:
        raise ValueError("cliport_consequence_value_hidden_dim_invalid")
    if architecture not in CONSEQUENCE_VALUE_ARCHITECTURES:
        raise ValueError("cliport_consequence_value_architecture_invalid")
    import torch
    from torch import nn

    class GoalConditionedConsequenceValueHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_dim = int(hidden_dim)
            self.architecture = str(architecture)

            def latent_encoder():
                return nn.Sequential(
                    nn.LayerNorm(WAM_LATENT_DIM),
                    nn.Linear(WAM_LATENT_DIM, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )

            self.future_encoder = latent_encoder()
            self.delta_encoder = latent_encoder()
            self.direct_relative_encoder = latent_encoder()
            self.max_delta_encoder = latent_encoder()
            self.language_encoder = nn.Sequential(
                nn.LayerNorm(384),
                nn.Linear(384, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            fusion_count = 8
            self.value = nn.Sequential(
                nn.LayerNorm(hidden_dim * fusion_count),
                nn.Linear(hidden_dim * fusion_count, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, 1),
            )

        def forward(
            self,
            predicted_future_visual,
            current_projected_visual,
            language_features,
            *,
            predicted_visual_delta=None,
        ):
            batch = predicted_future_visual.shape[0]
            if predicted_future_visual.shape != (
                batch,
                WAM_CANDIDATES,
                WAM_VISUAL_TOKENS,
                WAM_LATENT_DIM,
            ):
                raise ValueError("cliport_consequence_value_predicted_future_shape_invalid")
            if current_projected_visual.shape != (batch, WAM_VISUAL_TOKENS, WAM_LATENT_DIM):
                raise ValueError("cliport_consequence_value_current_visual_shape_invalid")
            if language_features.shape != (batch, 384):
                raise ValueError("cliport_consequence_value_language_shape_invalid")
            pooled = predicted_future_visual.mean(dim=2)
            current = current_projected_visual.mean(dim=1)
            if (
                predicted_visual_delta is not None
                and predicted_visual_delta.shape != predicted_future_visual.shape
            ):
                raise ValueError("cliport_consequence_value_predicted_delta_shape_invalid")
            delta_tokens = (
                predicted_future_visual - current_projected_visual[:, None]
                if predicted_visual_delta is None
                else predicted_visual_delta
            )
            spatial_max_delta = delta_tokens.amax(dim=2)
            result = self.forward_pooled(
                pooled, current, language_features, spatial_max_delta=spatial_max_delta
            )
            result["predicted_pooled_consequence"] = pooled
            return result

        def forward_pooled(
            self,
            predicted_pooled_consequence,
            current_pooled_consequence,
            language_features,
            *,
            spatial_max_delta=None,
        ):
            """Score already-pooled predicted latents from precomputed latent tensors."""
            batch = predicted_pooled_consequence.shape[0]
            if predicted_pooled_consequence.shape != (batch, WAM_CANDIDATES, WAM_LATENT_DIM):
                raise ValueError("cliport_consequence_value_predicted_pooled_shape_invalid")
            if current_pooled_consequence.shape != (batch, WAM_LATENT_DIM):
                raise ValueError("cliport_consequence_value_current_pooled_shape_invalid")
            if language_features.shape != (batch, 384):
                raise ValueError("cliport_consequence_value_language_shape_invalid")
            if spatial_max_delta is None or spatial_max_delta.shape != (
                batch,
                WAM_CANDIDATES,
                WAM_LATENT_DIM,
            ):
                raise ValueError("cliport_consequence_value_spatial_max_delta_shape_invalid")
            pooled = predicted_pooled_consequence
            current = current_pooled_consequence[:, None]
            direct = pooled[:, :1]
            future = self.future_encoder(pooled)
            delta = self.delta_encoder(pooled - current)
            relative = self.direct_relative_encoder(pooled - direct)
            language = self.language_encoder(language_features)[:, None].expand(
                -1, WAM_CANDIDATES, -1
            )
            features = [future, delta, relative, language, future * language, delta * language]
            maximum = self.max_delta_encoder(spatial_max_delta)
            features.extend((maximum, maximum * language))
            fused = torch.cat(features, dim=-1)
            raw_value = self.value(fused).squeeze(-1)
            residual = torch.tanh(raw_value - raw_value[:, :1])
            return {"consequence_value_logits": raw_value, "bounded_relative_residual": residual}

    return GoalConditionedConsequenceValueHead()


def build_cliport_consequence_lookahead(
    ranker,
    *,
    target_mean: Sequence[float] | np.ndarray,
    target_std: Sequence[float] | np.ndarray,
    residual_scale: float,
    projection: np.ndarray | None = None,
    value_hidden_dim: int = CONSEQUENCE_VALUE_HIDDEN_DIM,
    wam_hidden_dim: int = WAM_HIDDEN_DIM,
    value_architecture: str = "mean_max_delta",
    detach_wam_for_value: bool = False,
    decision_base_mode: str = "direct_candidate",
):
    """Build the frozen-ranker -> WAM -> consequence-value score path."""
    import torch
    from torch import nn

    scale = float(residual_scale)
    if not np.isfinite(scale) or not 0.0 <= scale <= MAX_RESIDUAL_SCALE:
        raise ValueError("cliport_consequence_lookahead_scale_invalid")
    if decision_base_mode not in CONSEQUENCE_DECISION_BASE_MODES:
        raise ValueError("cliport_consequence_lookahead_decision_base_mode_invalid")
    ranker.requires_grad_(False)
    ranker.eval()
    wam = build_latent_wam(
        target_mean=target_mean,
        target_std=target_std,
        projection=projection,
        hidden_dim=wam_hidden_dim,
    )
    value_head = build_goal_conditioned_consequence_value_head(
        hidden_dim=value_hidden_dim, architecture=value_architecture
    )

    class CLIPortConsequenceLookahead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ranker = ranker
            self.wam = wam
            self.value_head = value_head
            self.detach_wam_for_value = bool(detach_wam_for_value)
            self.decision_base_mode = str(decision_base_mode)
            self.register_buffer("residual_scale", torch.tensor(scale, dtype=torch.float32))

        def train(self, mode: bool = True):
            super().train(mode)
            self.ranker.eval()
            return self

        def set_residual_scale(self, value: float) -> None:
            new_scale = float(value)
            if not np.isfinite(new_scale) or not 0.0 <= new_scale <= MAX_RESIDUAL_SCALE:
                raise ValueError("cliport_consequence_lookahead_scale_invalid")
            self.residual_scale.fill_(new_scale)

        def forward(self, visual_features, state4, language_features, progress, candidate_actions):
            _validate_deployment_tensors(
                visual_features, state4, language_features, progress, candidate_actions
            )
            with torch.no_grad():
                base = self.ranker(
                    visual_features,
                    state4,
                    language_features,
                    progress,
                    candidate_actions,
                    return_components=True,
                )
            consequence = self.wam(visual_features, base["score_features"].detach())
            predicted_future = consequence["predicted_future_visual"]
            value_future = (
                predicted_future.detach() if self.detach_wam_for_value else predicted_future
            )
            current_projected = torch.matmul(visual_features, self.wam.visual_delta_projection)
            if self.detach_wam_for_value:
                current_projected = current_projected.detach()
            value = self.value_head(
                value_future,
                current_projected,
                language_features,
                predicted_visual_delta=consequence["predicted_visual_delta"].detach()
                if self.detach_wam_for_value
                else consequence["predicted_visual_delta"],
            )
            residual = value["bounded_relative_residual"]
            decision_base = torch.zeros_like(base["scores"])
            final_scores = decision_base + self.residual_scale * residual
            return {
                "scores": final_scores,
                "final_scores": final_scores,
                "base_scores": base["scores"],
                "decision_base_scores": decision_base,
                "branch_scores": base["branch_scores"],
                "relative_advantage": base["relative_advantage"],
                "bounded_relative_residual": residual,
                "scaled_residual": self.residual_scale * residual,
                "consequence_value_logits": value["consequence_value_logits"],
                **consequence,
            }

    return CLIPortConsequenceLookahead()
