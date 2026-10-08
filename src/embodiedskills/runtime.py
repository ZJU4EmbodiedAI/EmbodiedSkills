"""Observe, propose, predict, select and execute through a shared skill interface."""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from .checkpoint import load_bundle


@dataclass(frozen=True)
class ObservationFeatures:
    visual: np.ndarray
    state: np.ndarray
    language: np.ndarray
    progress: np.ndarray


@dataclass(frozen=True)
class Candidates:
    """Five executable proposals; slot zero is the frozen policy's direct action."""

    actions: np.ndarray
    executable: tuple[Any, ...]
    mask: np.ndarray | None = None

    def __post_init__(self):
        if len(self.executable) != 5 or self.actions.shape[0] != 5:
            raise ValueError("Expected five proposals with matching executable actions")


def member_forward(model, family, statistics, inputs):
    visual, state, language, progress, actions = (
        inputs[key] for key in ("visual", "state", "language", "progress", "actions")
    )
    if family == "spatial":
        return model(visual, state, language, progress, actions)

    def normalized(value, name):
        mean = value.new_tensor(statistics[f"{name}_mean"])
        std = value.new_tensor(statistics[f"{name}_std"])
        return (value - mean) / std

    return model(
        visual,
        normalized(state, "state"),
        normalized(language, "language"),
        progress,
        normalized(actions, "action"),
        inputs["mask"],
    )


class Selector:
    def __init__(self, checkpoint, *, device="cpu"):
        self.device = torch.device(device)
        self.payload, self.members, self.head = load_bundle(checkpoint, self.device)
        self.family = self.payload["family"]

    @torch.inference_mode()
    def predict(self, *, visual, state, language, progress, actions, mask=None):
        """Batched inference. Future observations and outcome labels are not inputs."""
        inputs = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in {
                "visual": visual,
                "state": state,
                "language": language,
                "progress": progress,
                "actions": actions,
            }.items()
        }
        if any(not bool(value.isfinite().all()) for value in inputs.values()):
            raise ValueError("Deployment inputs must be finite")
        if self.family == "continuous":
            if mask is None:
                mask = np.ones(inputs["actions"].shape[:3], dtype=np.float32)
            inputs["mask"] = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
            if not bool(((inputs["mask"] == 0) | (inputs["mask"] == 1)).all()):
                raise ValueError("Action mask must contain zero or one")
        scores, futures = [], []
        for model, payload in zip(self.members, self.payload["members"], strict=True):
            result = member_forward(model, self.family, payload["statistics"], inputs)
            if self.family == "spatial":
                scores.append(result["scores"])
                futures.append(result["predicted_visual_delta"].mean(dim=2))
            else:
                scores.append(result["score"])
                futures.append(result["predicted_future"])
        result = self.head(torch.stack(scores, dim=1), torch.stack(futures, dim=1))
        return {
            "scores": result["scores"].cpu().numpy(),
            "selected": result["scores"].argmax(dim=1).cpu().numpy(),
        }

    def select(self, observation: ObservationFeatures, candidates: Candidates) -> int:
        result = self.predict(
            visual=observation.visual[None],
            state=observation.state[None],
            language=observation.language[None],
            progress=observation.progress[None],
            actions=candidates.actions[None],
            mask=None if candidates.mask is None else candidates.mask[None],
        )
        return int(result["selected"][0])


class Skills(Protocol):
    """A backend supplies sensing, its frozen policy, and native execution."""

    def observe(self) -> ObservationFeatures: ...
    def propose(self, observation: ObservationFeatures) -> Candidates: ...
    def execute(self, action: Any) -> bool:
        """Execute one native action/chunk and return whether the episode ended."""
        ...


def run_loop(
    skills: Skills,
    selector: Selector,
    *,
    max_decisions: int,
    intervention_steps: set[int] | None = None,
) -> list[int]:
    """None enables selection at every step; {0} selects only the first action."""
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    selections = []
    for step in range(max_decisions):
        observation = skills.observe()
        candidates = skills.propose(observation)
        selected = (
            selector.select(observation, candidates)
            if intervention_steps is None or step in intervention_steps
            else 0
        )
        selections.append(selected)
        if skills.execute(candidates.executable[selected]):
            break
    return selections
