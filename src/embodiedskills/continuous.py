"""Recurrent continuous-action latent world model."""

from __future__ import annotations

import numpy as np


def build_strict_latent_wam_mt50(
    *,
    delta_mean: np.ndarray,
    delta_std: np.ndarray,
    hidden: int = 192,
    success_priority: float = 4.0,
    outcome_change_standardized: bool = True,
    detach_consequence_for_score: bool = True,
    state_dim: int = 4,
    action_dim: int = 4,
    execution_horizon: int = 8,
):
    """Build the consequence-only scorer used by training and deployment."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class StrictLatentWAMMT50(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if hidden < 64 or hidden % 8:
                raise ValueError("hidden_must_be_at_least_64_and_divisible_by_8")
            self.hidden = int(hidden)
            self.success_priority = float(success_priority)
            self.outcome_change_standardized = bool(outcome_change_standardized)
            self.detach_consequence_for_score = bool(detach_consequence_for_score)
            self.current_encoder = nn.Sequential(
                nn.LayerNorm(1024), nn.Linear(1024, hidden), nn.SiLU()
            )
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden)
            )
            self.language_encoder = nn.Sequential(
                nn.LayerNorm(384), nn.Linear(384, hidden), nn.SiLU()
            )
            self.progress_encoder = nn.Sequential(nn.Linear(1, hidden), nn.SiLU())
            self.action_gru = nn.GRU(action_dim + 1, hidden, batch_first=True)
            self.transition = nn.Sequential(
                nn.LayerNorm(hidden * 5),
                nn.Linear(hidden * 5, hidden * 2),
                nn.SiLU(),
                nn.Linear(hidden * 2, hidden),
                nn.SiLU(),
            )
            self.future_delta = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1024))
            self.outcome_current = nn.Sequential(
                nn.LayerNorm(1024), nn.Linear(1024, hidden), nn.SiLU()
            )
            self.outcome_future = nn.Sequential(
                nn.LayerNorm(1024), nn.Linear(1024, hidden), nn.SiLU()
            )
            if self.outcome_change_standardized:
                self.outcome_change = nn.Sequential(
                    nn.LayerNorm(1024), nn.Linear(1024, hidden), nn.SiLU()
                )
                outcome_width = hidden * 6
            else:
                self.outcome_change = None
                outcome_width = hidden * 5
            self.outcome = nn.Sequential(
                nn.LayerNorm(outcome_width),
                nn.Linear(outcome_width, hidden * 2),
                nn.SiLU(),
                nn.Linear(hidden * 2, hidden),
                nn.SiLU(),
            )
            self.reward_head = nn.Linear(hidden, 1)
            self.value_head = nn.Linear(hidden, 1)
            self.register_buffer("delta_mean", torch.as_tensor(delta_mean, dtype=torch.float32))
            self.register_buffer("delta_std", torch.as_tensor(delta_std, dtype=torch.float32))
            nn.init.zeros_(self.future_delta[-1].weight)
            with torch.no_grad():
                self.future_delta[-1].bias.copy_(-self.delta_mean / self.delta_std)

        def score_from_consequence(
            self, current_latent, predicted_future, state, language, progress
        ):
            _, candidates = predicted_future.shape[:2]
            current = self.outcome_current(current_latent)[:, None].expand(-1, candidates, -1)
            future = self.outcome_future(predicted_future)
            state_feature = self.state_encoder(state)[:, None].expand(-1, candidates, -1)
            language_feature = self.language_encoder(language)[:, None].expand(-1, candidates, -1)
            progress_feature = self.progress_encoder(progress)[:, None].expand(-1, candidates, -1)
            consequence_inputs = [
                current,
                future,
                state_feature,
                language_feature,
                progress_feature,
            ]
            if self.outcome_change is not None:
                current_expanded = current_latent[:, None].expand_as(predicted_future)
                predicted_change_standard = (
                    predicted_future - current_expanded - self.delta_mean
                ) / self.delta_std
                consequence_inputs.append(self.outcome_change(predicted_change_standard))
            consequence = self.outcome(torch.cat(consequence_inputs, dim=-1))
            macro_reward_std = self.reward_head(consequence).squeeze(-1)
            terminal_value_logit = self.value_head(consequence).squeeze(-1)
            score = macro_reward_std + self.success_priority * terminal_value_logit.sigmoid()
            return {
                "macro_reward_std": macro_reward_std,
                "terminal_value_logit": terminal_value_logit,
                "score": score,
            }

        def forward(self, current_latent, state, language, progress, actions, action_mask):
            if actions.ndim != 4 or actions.shape[1:] != (5, execution_horizon, action_dim):
                raise ValueError("actions_must_match_candidate_horizon_action_contract")
            if state.shape != (actions.shape[0], state_dim):
                raise ValueError("state_must_match_public_state_contract")
            if action_mask.shape != actions.shape[:3]:
                raise ValueError("action_mask_must_be_B_5_8")
            batch, candidates = actions.shape[:2]
            mask = action_mask.unsqueeze(-1).to(actions.dtype)
            action_input = torch.cat((actions * mask, mask), dim=-1)
            _, action_hidden = self.action_gru(
                action_input.reshape(batch * candidates, execution_horizon, action_dim + 1)
            )
            action_hidden = action_hidden[-1].reshape(batch, candidates, self.hidden)
            transition_input = torch.cat(
                (
                    self.current_encoder(current_latent)[:, None].expand(-1, candidates, -1),
                    self.state_encoder(state)[:, None].expand(-1, candidates, -1),
                    self.language_encoder(language)[:, None].expand(-1, candidates, -1),
                    self.progress_encoder(progress)[:, None].expand(-1, candidates, -1),
                    action_hidden,
                ),
                dim=-1,
            )
            delta_standard = self.future_delta(self.transition(transition_input))
            delta = delta_standard * self.delta_std + self.delta_mean
            predicted_future = F.normalize(current_latent[:, None] + delta, dim=-1)
            result = {
                "predicted_future": predicted_future,
                "predicted_delta_standard": delta_standard,
            }
            result.update(
                self.score_from_consequence(
                    current_latent,
                    predicted_future.detach()
                    if self.detach_consequence_for_score
                    else predicted_future,
                    state,
                    language,
                    progress,
                )
            )
            return result

    return StrictLatentWAMMT50()
