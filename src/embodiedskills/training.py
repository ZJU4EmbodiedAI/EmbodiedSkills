"""Train the context encoder, three WAM members and shared consensus head."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoint import build_member, cpu_state, save_bundle
from .consensus import build_future_consensus_head, fit_effect_rms, joint_targets
from .continuous_consensus import build_future_consequence_critic, fit_effect_projection
from .data import batch, ensure_disjoint, load_archive, mean_std, task_balanced_indices
from .losses import (
    _loss,
    _success_pairwise_loss,
    candidate_ranking_loss,
    direct_relative_branch_loss,
)
from .runtime import member_forward
from .spatial import build_cliport_mt50_transfer_ranker, fixed_visual_delta_projection


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_statistics(data, family):
    stats = {}
    for name, key, axis in (
        ("state", "state", 0),
        ("language", "language", 0),
        ("action", "actions", (0, 1) if family == "spatial" else (0, 1, 2)),
    ):
        stats[f"{name}_mean"], stats[f"{name}_std"] = mean_std(data[key], axis)
    delta = data["future_visual"] - data["visual"][:, None]
    if family == "spatial":
        stats["projection"] = fixed_visual_delta_projection()
        delta = delta @ stats["projection"]
    stats["delta_mean"], stats["delta_std"] = mean_std(
        delta,
        (0, 1, 2) if family == "spatial" else (0, 1),
        floor=1e-4 if family == "spatial" else 1e-5,
    )
    result = {key: value.tolist() for key, value in stats.items()}
    if family == "continuous":
        result["returns"] = {}
        for task in np.unique(data["task"]):
            mean, std = mean_std(data["return"][data["task"] == task], None)
            result["returns"][str(task)] = [float(mean), float(std)]
    return result


def context_loss(model, b, *, permute=False):
    actions, success = b["actions"], b["success"]
    reference = torch.zeros(len(actions), dtype=torch.long, device=actions.device)
    if permute:
        order = torch.stack([torch.randperm(5, device=actions.device) for _ in actions])
        actions = actions.gather(1, order[..., None].expand_as(actions))
        success = success.gather(1, order)
        reference = (order == 0).long().argmax(1)
    out = model(
        b["visual"],
        b["state"],
        b["language"],
        b["progress"],
        actions,
        direct_reference_index=reference,
        return_components=True,
    )
    informative = success.max(1).values > success.min(1).values
    loss = out["scores"].sum() * 0
    if bool(informative.any()):
        targets = success[informative] / success[informative].sum(1, keepdim=True)
        loss = candidate_ranking_loss(out["scores"][informative], targets, pairwise_weight=0.25)[
            "loss"
        ]
    return loss + direct_relative_branch_loss(out["relative_advantage"], success, reference)["loss"]


def train_context(train, validation, statistics, config, device):
    seed_everything(config["seed"])
    model = build_cliport_mt50_transfer_ranker(
        direct_relative=True,
        **{k: statistics[k] for k in ("state_mean", "state_std", "action_mean", "action_std")},
    ).to(device)
    settings = config["context"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=1e-4
    )
    best, best_state = float("inf"), None
    rng = np.random.default_rng(config["seed"])
    for epoch in range(settings["epochs"]):
        model.train()
        order = task_balanced_indices(train["task"], rng)
        for start in range(0, len(order), config["batch_size"]):
            b = batch(train, order[start : start + config["batch_size"]], device, labels=True)
            loss = context_loss(model, b, permute=True)
            update(optimizer, model, loss)
        model.eval()
        total = 0.0
        with torch.no_grad():
            for indices in chunks(len(validation["task"]), config["batch_size"]):
                loss = context_loss(model, batch(validation, indices, device, labels=True))
                total += float(loss) * len(indices)
        metric = total / len(validation["task"])
        print(
            json.dumps({"stage": "context", "epoch": epoch + 1, "validation_loss": metric}),
            flush=True,
        )
        if metric < best:
            best, best_state = metric, cpu_state(model)
    model.load_state_dict(best_state)
    return model.eval().requires_grad_(False)


def chunks(count, batch_size):
    for start in range(0, count, batch_size):
        yield np.arange(start, min(start + batch_size, count))


def update(optimizer, model, loss):
    if not bool(loss.isfinite()):
        raise ValueError("Nonfinite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def member_loss(model, family, statistics, b, tasks, positive_weight, settings, *, warmup=False):
    output = member_forward(model, family, statistics, b)
    if family == "spatial":
        raw_delta = (b["future_visual"] - b["visual"][:, None]) @ model.wam.visual_delta_projection
        delta_loss = F.smooth_l1_loss(
            output["predicted_visual_delta_normalized"], model.wam.normalize_target(raw_delta)
        )
        if warmup:
            return delta_loss
        logits, relative = output["consequence_value_logits"], output["bounded_relative_residual"]
        bce = F.binary_cross_entropy_with_logits(
            logits, b["success"], pos_weight=logits.new_tensor(positive_weight)
        )
        pairwise = _success_pairwise_loss(torch, logits, b["success"])
        advantage = direct_relative_branch_loss(
            relative, b["success"], torch.zeros(len(logits), dtype=torch.long, device=logits.device)
        )["loss"]
        return delta_loss + 0.25 * bce + 0.5 * pairwise + advantage

    target_delta = b["future_visual"] - b["visual"][:, None]
    return_stats = b["return"].new_tensor([statistics["returns"][str(t)] for t in tasks])
    expert = b["expert_utility"] - b["expert_utility"].mean(1, keepdim=True)
    expert = expert / expert.std(1, keepdim=True, unbiased=False).clamp_min(1e-5)
    labels = {
        "future": b["future_visual"],
        "success": b["success"],
        "delta_standard": (target_delta - model.delta_mean) / model.delta_std,
        "return_standard": (b["return"] - return_stats[:, :1]) / return_stats[:, 1:],
        "expert_standard": expert,
    }
    return _loss(
        output,
        labels,
        success_priority=model.success_priority,
        positive_weight=positive_weight,
        expert_weight=settings.get("expert_weight", 0.5),
    )[0]


@torch.no_grad()
def member_validation(model, family, statistics, data, device, batch_size):
    model.eval()
    successful, harms, cosine_gain, expert_correct = 0, 0, 0.0, 0
    for indices in chunks(len(data["task"]), batch_size):
        b = batch(data, indices, device, labels=True)
        output = member_forward(model, family, statistics, b)
        score = output["scores"] if family == "spatial" else output["score"]
        selected = b["success"].gather(1, score.argmax(1)[:, None]).squeeze(1)
        successful += int(selected.sum())
        harms += int((b["success"][:, 0] > selected).sum())
        if family == "continuous":
            cosine_gain += float(
                ((output["predicted_future"] - b["visual"][:, None]) * b["future_visual"])
                .sum(-1)
                .mean(1)
                .sum()
            )
            expert_correct += int((score.argmax(1) == b["expert_utility"].argmax(1)).sum())
    if family == "spatial":
        return (successful, -harms)
    return ((cosine_gain + 0.01 * expert_correct) / len(data["task"]),)


def train_member(train, validation, family, statistics, config, model_seed, device, context=None):
    seed_everything(model_seed)
    model = build_member(family, config["model"], statistics).to(device)
    if context is not None:
        model.ranker.load_state_dict(context.state_dict())
    settings = config["member"]
    warmup_epochs = settings.get("warmup_epochs", 0)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=settings.get("warmup_learning_rate", settings["learning_rate"]),
        weight_decay=1e-4,
    )
    positives = float(train["success"].sum())
    positive_weight = (train["success"].size - positives) / max(positives, 1)
    if family == "continuous":
        positive_weight = min(max(positive_weight, 1), 30)
    best_key, best_state = None, None
    rng = np.random.default_rng(model_seed)
    for epoch in range(warmup_epochs + settings["epochs"]):
        warmup = epoch < warmup_epochs
        if epoch == warmup_epochs:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=settings["learning_rate"],
                weight_decay=1e-4,
            )
        model.train()
        order = (
            task_balanced_indices(train["task"], rng)
            if family == "spatial"
            else rng.permutation(len(train["task"]))
        )
        total = 0.0
        for start in range(0, len(order), config["batch_size"]):
            indices = order[start : start + config["batch_size"]]
            b = batch(train, indices, device, labels=True)
            loss = member_loss(
                model,
                family,
                statistics,
                b,
                train["task"][indices],
                positive_weight,
                settings,
                warmup=warmup,
            )
            update(optimizer, model, loss)
            total += float(loss.detach()) * len(indices)
        record = {
            "stage": "warmup" if warmup else "member",
            "seed": model_seed,
            "epoch": epoch + 1,
            "loss": total / len(order),
        }
        if not warmup:
            key = member_validation(
                model, family, statistics, validation, device, config["batch_size"]
            )
            record["validation_objective"] = key
            if best_key is None or key > best_key:
                best_key, best_state = key, cpu_state(model)
        print(json.dumps(record), flush=True)
    model.load_state_dict(best_state)
    return model.eval().requires_grad_(False)


@torch.no_grad()
def predict_members(models, family, statistics, data, device, batch_size):
    scores, effects = [], []
    for indices in chunks(len(data["task"]), batch_size):
        b = batch(data, indices, device)
        outputs = [member_forward(model, family, statistics, b) for model in models]
        scores.append(
            torch.stack(
                [o["scores"] if family == "spatial" else o["score"] for o in outputs], dim=1
            ).cpu()
        )
        effects.append(
            torch.stack(
                [
                    o["predicted_visual_delta"].mean(2)
                    if family == "spatial"
                    else o["predicted_future"]
                    for o in outputs
                ],
                dim=1,
            ).cpu()
        )
    return torch.cat(scores), torch.cat(effects)


def train_consensus(fit, val, train, validation, config, device):
    seed_everything(config["seed"])
    scores, effects = fit
    if config["family"] == "spatial":
        head = build_future_consensus_head(fit_effect_rms(torch, effects, np.arange(len(effects))))
    else:
        rms, mean, basis, _ = fit_effect_projection(torch, effects, np.arange(len(effects)))
        head = build_future_consequence_critic(effect_rms=rms, effect_mean=mean, pca_basis=basis)
    head.to(device)
    settings = config["consensus"]
    targets = joint_targets(torch, torch.as_tensor(train["success"]))
    val_targets = joint_targets(torch, torch.as_tensor(validation["success"]))
    counts = torch.bincount(targets[:, 1:].reshape(-1), minlength=4).float().clamp_min(1)
    weight = counts.rsqrt()
    weight = (weight / weight.mean()).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=settings["learning_rate"], weight_decay=1e-4
    )
    rng = np.random.default_rng(config["seed"])
    best, best_state = float("inf"), None
    for epoch in range(settings["epochs"]):
        head.train()
        order = task_balanced_indices(train["task"], rng, settings.get("samples_per_task"))
        for start in range(0, len(order), settings["batch_size"]):
            indices = order[start : start + settings["batch_size"]]
            output = head(scores[indices].to(device), effects[indices].to(device))
            loss = F.cross_entropy(
                output["logits"][:, 1:].reshape(-1, 4),
                targets[indices, 1:].reshape(-1).to(device),
                weight=weight,
            )
            update(optimizer, head, loss)
        head.eval()
        numerator, denominator = 0.0, 0.0
        with torch.no_grad():
            for indices in chunks(len(val_targets), settings["batch_size"]):
                output = head(val[0][indices].to(device), val[1][indices].to(device))
                target = val_targets[indices, 1:].reshape(-1).to(device)
                numerator += float(
                    F.cross_entropy(
                        output["logits"][:, 1:].reshape(-1, 4),
                        target,
                        weight=weight,
                        reduction="sum",
                    )
                )
                denominator += float(weight[target].sum())
        metric = numerator / denominator
        print(
            json.dumps({"stage": "consensus", "epoch": epoch + 1, "validation_loss": metric}),
            flush=True,
        )
        if metric < best:
            best, best_state = metric, cpu_state(head)
    head.load_state_dict(best_state)
    return head.eval()


def train(config, train_path, validation_path, output, *, device="cpu", context_checkpoint=None):
    family = config["family"]
    if family not in {"spatial", "continuous"}:
        raise ValueError("family must be spatial or continuous")
    if len(config["member_seeds"]) != 3 or len(set(config["member_seeds"])) != 3:
        raise ValueError("Provide three different member initialization seeds")
    for stage in ("member", "consensus") + (("context",) if family == "spatial" else ()):
        if config[stage]["epochs"] <= 0:
            raise ValueError("Training stages require a positive epoch count")
    training = load_archive(train_path, family, training=True)
    validation = load_archive(validation_path, family, training=True)
    ensure_disjoint(training, validation)
    statistics = fit_statistics(training, family)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    context = None
    if family == "spatial":
        if context_checkpoint:
            saved = torch.load(context_checkpoint, map_location="cpu", weights_only=True)
            context = build_cliport_mt50_transfer_ranker(direct_relative=True).to(device)
            context.load_state_dict(saved["state_dict"], strict=True)
            for key in ("state_mean", "state_std", "action_mean", "action_std"):
                statistics[key] = getattr(context, key).detach().cpu().tolist()
        else:
            context = train_context(training, validation, statistics, config, device)
        torch.save({"state_dict": cpu_state(context)}, output / "context.pt")
    models, payloads = [], []
    for index, member_seed in enumerate(config["member_seeds"]):
        model = train_member(
            training, validation, family, statistics, config, member_seed, device, context
        )
        member = {
            "config": config["model"],
            "statistics": statistics,
            "state_dict": cpu_state(model),
        }
        torch.save(member, output / f"member_{index}.pt")
        models.append(model)
        payloads.append(member)
    fit = predict_members(models, family, statistics, training, device, config["batch_size"])
    val = predict_members(models, family, statistics, validation, device, config["batch_size"])
    head = train_consensus(fit, val, training, validation, config, device)
    save_bundle(
        output / "selector.pt", family, payloads, head, encoder_config=config.get("encoders")
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
