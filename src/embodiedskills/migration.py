"""Convert existing research checkpoints into a portable inference bundle."""

import torch

from .checkpoint import build_head, build_member, cpu_state, save_bundle


def convert(family, member_paths, head_path, output):
    """Read trusted local PyTorch files; discard histories, machine paths and hashes."""
    if len(member_paths) != 3:
        raise ValueError("Provide the three member checkpoints used with this head")
    members = []
    for path in member_paths:
        old = torch.load(path, map_location="cpu", weights_only=False)
        if family == "spatial":
            cfg = old["model_config"]
            if (
                cfg["value_architecture"] != "mean_max_delta"
                or cfg["decision_base_mode"] != "direct_candidate"
                or cfg["detach_wam_for_value"]
            ):
                raise ValueError("This converter accepts the final spatial WAM architecture")
            context = old["frozen_ranker"]
            if not context["model_config"].get("direct_relative"):
                raise ValueError("Expected the direct-relative context encoder")
            statistics = dict(context["statistics"])
            statistics.update(
                {
                    "delta_mean": old["wam"]["target_mean"].tolist(),
                    "delta_std": old["wam"]["target_std"].tolist(),
                    "projection": old["wam"]["visual_delta_projection"].tolist(),
                }
            )
            config = {
                "hidden": old["wam"]["scene_projection.1.weight"].shape[0],
                "value_hidden": cfg["value_hidden_dim"],
                "residual_scale": cfg["residual_scale"],
            }
            model = build_member(family, config, statistics)
            model.ranker.load_state_dict(context["model"], strict=True)
            model.wam.load_state_dict(old["wam"], strict=True)
            model.value_head.load_state_dict(old["value_head"], strict=True)
        elif family == "continuous":
            cfg = old["model_config"]
            action_shape = cfg.get("action_shape", [8, 4])
            config = {
                "hidden": cfg["hidden"],
                "success_priority": cfg["success_priority"],
                "outcome_change_standardized": cfg["outcome_change_standardized"],
                "detach_consequence_for_score": cfg["detach_consequence_for_score"],
                "state_dim": old["model"]["state_encoder.0.weight"].shape[1],
                "action_dim": action_shape[1],
                "execution_horizon": action_shape[0],
            }
            statistics = {
                key: value
                for key, value in old["statistics"].items()
                if key
                in {
                    "state_mean",
                    "state_std",
                    "language_mean",
                    "language_std",
                    "action_mean",
                    "action_std",
                    "delta_mean",
                    "delta_std",
                }
            }
            model = build_member(family, config, statistics)
            model.load_state_dict(old["model"], strict=True)
        else:
            raise ValueError("Unknown model family")
        members.append({"config": config, "statistics": statistics, "state_dict": cpu_state(model)})
    old_head = torch.load(head_path, map_location="cpu", weights_only=False)
    state = old_head.get(
        "model_state_dict", old_head.get("critic_state_dict", old_head.get("model"))
    )
    if state is None:
        raise ValueError("No compatible consensus state dictionary found")
    head = build_head(family, state)
    save_bundle(output, family, members, head)
