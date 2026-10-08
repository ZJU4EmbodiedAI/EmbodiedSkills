"""Feature archives and episode-disjoint training inputs."""

from pathlib import Path

import numpy as np
import torch

INPUTS = ("visual", "state", "language", "progress", "actions", "mask")


def load_archive(path, family, *, training=False):
    with np.load(Path(path), allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    required = {"visual", "state", "language", "progress", "actions", "task", "seed", "anchor"}
    if training:
        required |= {"future_visual", "success"}
        if family == "continuous":
            required |= {"return", "expert_utility", "mask"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing archive fields: {sorted(missing)}")
    count = len(data["visual"])
    if count == 0 or any(len(data[key]) != count for key in required):
        raise ValueError("Archive fields must have the same nonzero row count")
    for key, value in data.items():
        if key not in {"task", "seed", "anchor"}:
            data[key] = np.asarray(value, dtype=np.float32)
            if not np.isfinite(data[key]).all():
                raise ValueError(f"Nonfinite archive field: {key}")
    spatial = family == "spatial"
    visual_shape = (count, 64, 1024) if spatial else (count, 1024)
    if data["visual"].shape != visual_shape:
        raise ValueError(f"Expected visual shape {visual_shape}")
    for key, shape in {"language": (count, 384), "progress": (count, 1)}.items():
        if data[key].shape != shape:
            raise ValueError(f"Expected {key} shape {shape}")
    if np.any((data["progress"] < 0) | (data["progress"] > 1)):
        raise ValueError("Progress must be between zero and one")
    if spatial and (data["actions"].shape != (count, 5, 20) or data["state"].shape != (count, 4)):
        raise ValueError("Spatial inputs require actions [N,5,20] and state [N,4]")
    if not spatial:
        if data["actions"].ndim != 4 or data["actions"].shape[:2] != (count, 5):
            raise ValueError("Continuous actions must have shape [N,5,horizon,action_dim]")
        if "mask" not in data:
            data["mask"] = np.ones(data["actions"].shape[:3], dtype=np.float32)
        if (
            data["mask"].shape != data["actions"].shape[:3]
            or not np.isin(data["mask"], [0, 1]).all()
        ):
            raise ValueError("Action mask must match [N,5,horizon] and contain zero or one")
    if "future_visual" in data and data["future_visual"].shape != (count, 5, *visual_shape[1:]):
        raise ValueError("Future features must add the five-candidate axis to current features")
    if "success" in data and (
        data["success"].shape != (count, 5) or not np.isin(data["success"], [0, 1]).all()
    ):
        raise ValueError("Success labels must be binary [N,5]")
    for key in ("return", "expert_utility"):
        if key in data and data[key].shape != (count, 5):
            raise ValueError(f"{key} must have shape [N,5]")
    identities = list(zip(data["task"].astype(str), data["seed"].tolist(), data["anchor"].tolist()))
    if len(set(identities)) != count:
        raise ValueError("Duplicate task/seed/anchor identity")
    return data


def ensure_disjoint(train, validation):
    def identities(data):
        return set(zip(data["task"].astype(str), data["seed"].tolist()))

    if identities(train) & identities(validation):
        raise ValueError("Training and validation overlap at the task/episode-seed level")


def batch(data, indices, device, *, labels=False):
    keys = INPUTS + (("future_visual", "success", "return", "expert_utility") if labels else ())
    return {
        key: torch.as_tensor(data[key][indices], dtype=torch.float32, device=device)
        for key in keys
        if key in data
    }


def task_balanced_indices(tasks, generator, samples_per_task=None):
    groups = [np.flatnonzero(tasks == task) for task in np.unique(tasks)]
    count = samples_per_task or max(1, int(np.ceil(len(tasks) / len(groups))))
    indices = np.concatenate([generator.choice(group, count, replace=True) for group in groups])
    generator.shuffle(indices)
    return indices


def mean_std(values, axis, floor=1e-5):
    return (
        values.mean(axis=axis, dtype=np.float64).astype(np.float32),
        np.maximum(values.std(axis=axis, dtype=np.float64), floor).astype(np.float32),
    )


def outcome_metrics(success, selections, tasks):
    def summarize(indices):
        direct = success[indices, 0] > 0
        chosen = success[indices, selections[indices]] > 0
        return {
            "rows": len(indices),
            "direct_successes": int(direct.sum()),
            "selected_successes": int(chosen.sum()),
            "direct_rate": float(direct.mean()),
            "selected_rate": float(chosen.mean()),
            "rescued": int((~direct & chosen).sum()),
            "harmed": int((direct & ~chosen).sum()),
            "switches": int((selections[indices] != 0).sum()),
        }

    return {
        "overall": summarize(np.arange(len(success))),
        "per_task": {
            str(task): summarize(np.flatnonzero(tasks == task)) for task in np.unique(tasks)
        },
    }
