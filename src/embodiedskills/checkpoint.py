"""Self-contained member and ensemble checkpoints."""

from pathlib import Path

import numpy as np
import torch

from .consensus import build_future_consensus_head
from .continuous import build_strict_latent_wam_mt50
from .continuous_consensus import build_future_consequence_critic
from .spatial import build_cliport_consequence_lookahead, build_cliport_mt50_transfer_ranker

SCHEMA = "embodiedskills.bundle.v1"


def build_member(family, config, statistics):
    if family == "spatial":
        ranker = build_cliport_mt50_transfer_ranker(
            direct_relative=True,
            state_mean=statistics["state_mean"],
            state_std=statistics["state_std"],
            action_mean=statistics["action_mean"],
            action_std=statistics["action_std"],
        )
        return build_cliport_consequence_lookahead(
            ranker,
            target_mean=statistics["delta_mean"],
            target_std=statistics["delta_std"],
            projection=np.asarray(statistics["projection"], dtype=np.float32),
            residual_scale=config.get("residual_scale", 0.1),
            wam_hidden_dim=config.get("hidden", 128),
            value_hidden_dim=config.get("value_hidden", 64),
        )
    if family == "continuous":
        return build_strict_latent_wam_mt50(
            delta_mean=np.asarray(statistics["delta_mean"], dtype=np.float32),
            delta_std=np.asarray(statistics["delta_std"], dtype=np.float32),
            **config,
        )
    raise ValueError(f"Unknown member family: {family}")


def build_head(family, state):
    if family == "spatial":
        model = build_future_consensus_head(effect_rms=state["effect_rms"])
    elif family == "continuous":
        model = build_future_consequence_critic(
            effect_rms=state["effect_rms"],
            effect_mean=state["effect_mean"],
            pca_basis=state["pca_basis"],
        )
    else:
        raise ValueError(f"Unknown head family: {family}")
    model.load_state_dict(state, strict=True)
    return model


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_bundle(path, family, members, head, *, encoder_config=None):
    if len(members) != 3:
        raise ValueError("The consensus architecture requires three members")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA,
            "family": family,
            "members": members,
            "head": cpu_state(head),
            "encoders": encoder_config or {},
        },
        path,
    )


def load_bundle(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA or len(payload.get("members", [])) != 3:
        raise ValueError("Expected an EmbodiedSkills bundle containing three WAM members")
    family = payload["family"]
    members = []
    for member in payload["members"]:
        model = build_member(family, member["config"], member["statistics"])
        model.load_state_dict(member["state_dict"], strict=True)
        members.append(model.to(device).eval().requires_grad_(False))
    head = build_head(family, payload["head"]).to(device).eval().requires_grad_(False)
    return payload, members, head
