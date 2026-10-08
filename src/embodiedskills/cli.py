"""Command-line training, feature extraction and cached-branch evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .data import batch, load_archive, outcome_metrics
from .runtime import Selector


def encode(args):
    from .language import LanguageEncoder
    from .vision import FrozenPublicVisualEncoder

    with np.load(args.input, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    visual = FrozenPublicVisualEncoder(
        backbone=args.backbone,
        model=args.visual_model,
        device=args.device,
        spatial_grid=8 if args.family == "spatial" else 1,
    )
    language = LanguageEncoder(args.language_model, device=args.device)

    def images_to_features(images):
        features = visual.encode_rgb(images, batch_size=args.batch_size)
        if args.family == "continuous":
            features = features[:, 0]
            features /= np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-12)
        return features

    data["visual"] = images_to_features(data.pop("rgb"))
    data["language"] = language.encode(data.pop("instruction").tolist(), batch_size=args.batch_size)
    if "future_rgb" in data:
        frames = data.pop("future_rgb")
        if frames.ndim != 5 or frames.shape[1:] != (5, 256, 256, 3):
            raise ValueError("future_rgb must have shape [N,5,256,256,3]")
        features = images_to_features(frames.reshape(-1, 256, 256, 3))
        data["future_visual"] = features.reshape(len(frames), 5, *features.shape[1:])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        np.savez_compressed(stream, **data)


def infer(args):
    selector = Selector(args.checkpoint, device=args.device)
    data = load_archive(args.input, selector.family)
    if args.command == "evaluate" and "success" not in data:
        raise ValueError("Evaluation requires recorded branch success labels")
    count = len(data["visual"])
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        count = min(count, args.limit)
    outputs = []
    for start in range(0, count, args.batch_size):
        indices = np.arange(start, min(count, start + args.batch_size))
        inputs = batch(data, indices, args.device)
        outputs.append(selector.predict(**inputs))
    selected = np.concatenate([out["selected"] for out in outputs])
    scores = np.concatenate([out["scores"] for out in outputs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "predict":
        with args.output.open("wb") as stream:
            np.savez_compressed(stream, selected=selected, scores=scores)
    else:
        report = {
            "evaluation": "cached_branch_outcomes",
            **outcome_metrics(
                data["success"][:count],
                selected,
                data["task"][:count],
            ),
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train three WAM members and the consensus head")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--train", type=Path, required=True)
    training.add_argument("--validation", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--context-checkpoint", type=Path)
    training.add_argument("--device", default="cpu")
    for name in ("predict", "evaluate"):
        item = commands.add_parser(
            name,
            help="Score feature archives" if name == "predict" else "Evaluate recorded branches",
        )
        item.add_argument("--checkpoint", type=Path, required=True)
        item.add_argument("--input", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--device", default="cpu")
        item.add_argument("--batch-size", type=int, default=32)
        item.add_argument("--limit", type=int)
    encoding = commands.add_parser(
        "encode", help="Encode RGB and instructions using frozen backbones"
    )
    encoding.add_argument("--input", type=Path, required=True)
    encoding.add_argument("--output", type=Path, required=True)
    encoding.add_argument("--family", choices=("spatial", "continuous"), required=True)
    encoding.add_argument("--backbone", choices=("vjepa2", "dinov2"), default="vjepa2")
    encoding.add_argument("--visual-model", required=True)
    encoding.add_argument("--language-model", required=True)
    encoding.add_argument("--device", default="cpu")
    encoding.add_argument("--batch-size", type=int, default=16)
    conversion = commands.add_parser("convert", help="Convert trusted local research checkpoints")
    conversion.add_argument("--family", choices=("spatial", "continuous"), required=True)
    conversion.add_argument("--members", nargs=3, type=Path, required=True)
    conversion.add_argument("--head", type=Path, required=True)
    conversion.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, "batch_size", 1) <= 0:
        parser.error("batch-size must be positive")
    if args.command == "train":
        from .training import train

        config = yaml.safe_load(args.config.read_text())
        train(
            config,
            args.train,
            args.validation,
            args.output,
            device=args.device,
            context_checkpoint=args.context_checkpoint,
        )
    elif args.command == "encode":
        encode(args)
    elif args.command == "convert":
        from .migration import convert

        convert(args.family, args.members, args.head, args.output)
    else:
        infer(args)


if __name__ == "__main__":
    main()
