"""Frozen RGB feature extraction shared by training and deployment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

VJEPA2_BACKBONE = "vjepa2"
DINOV2_BACKBONE = "dinov2"
VISUAL_BACKBONES = (VJEPA2_BACKBONE, DINOV2_BACKBONE)


def visual_preprocessing_contract(backbone: str) -> str:
    """Return the stable preprocessing identity stored in feature metadata."""
    if backbone == VJEPA2_BACKBONE:
        return "resize_292_center_crop_256_imagenet_two_identical_frames"
    if backbone == DINOV2_BACKBONE:
        return "pinned_huggingface_auto_image_processor_spatial_patch_tokens"
    raise ValueError(f"public_mt50_visual_backbone_unknown:{backbone}")


def spatial_patch_tokens(last_hidden_state, *, image_shape: tuple[int, int], patch_size: int):
    """Remove model prefix tokens and return the rectangular patch grid.

    DINO-family checkpoints may contain a class token and optional register
    tokens.  The image geometry determines the patch-token count, so taking
    the final ``H/P * W/P`` tokens is explicit and remains correct with either
    prefix contract.
    """
    height, width = (int(value) for value in image_shape)
    patch_size = int(patch_size)
    if height <= 0 or width <= 0 or patch_size <= 0:
        raise ValueError("public_mt50_visual_patch_geometry_must_be_positive")
    if height % patch_size or width % patch_size:
        raise ValueError("public_mt50_visual_image_must_align_to_patch_size")
    patch_height, patch_width = (height // patch_size, width // patch_size)
    patch_count = patch_height * patch_width
    if last_hidden_state.ndim != 3 or last_hidden_state.shape[1] <= patch_count:
        raise ValueError("public_mt50_visual_tokens_missing_prefix_or_patches")
    return (last_hidden_state[:, -patch_count:], patch_height, patch_width)


class FrozenPublicVisualEncoder:
    """Backbone-neutral frozen RGB-to-spatial-token adapter."""

    def __init__(
        self, *, backbone: str, model: str | Path, device: str = "cuda", spatial_grid: int = 8
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if backbone not in VISUAL_BACKBONES:
            raise ValueError(f"public_mt50_visual_backbone_unknown:{backbone}")
        if not 1 <= int(spatial_grid) <= 16:
            raise ValueError("public_mt50_spatial_grid_must_be_in_1_16")
        self.torch = torch
        self.backbone = str(backbone)
        self.model_path = str(model)
        self.device = str(device)
        self.spatial_grid = int(spatial_grid)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = (
            AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .to(self.device)
            .eval()
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.image_processor = (
            AutoImageProcessor.from_pretrained(
                self.model_path, local_files_only=True, use_fast=False
            )
            if self.backbone == DINOV2_BACKBONE
            else None
        )
        self.feature_dim = int(self.model.config.hidden_size)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    @property
    def preprocessing_contract(self) -> str:
        return visual_preprocessing_contract(self.backbone)

    @staticmethod
    def _validate_images(images: np.ndarray, batch_size: int) -> np.ndarray:
        images = np.asarray(images)
        if images.ndim != 4 or images.shape[1:] != (256, 256, 3) or images.dtype != np.uint8:
            raise ValueError("public_mt50_encoder_rgb_must_be_B_256_256_3_uint8")
        if batch_size <= 0:
            raise ValueError("public_mt50_encoder_batch_size_must_be_positive")
        return images

    def _pool_tokens(self, tokens, *, height: int, width: int) -> np.ndarray:
        from torch.nn import functional

        if tokens.ndim != 3 or tokens.shape[1] != height * width:
            raise ValueError("public_mt50_visual_tokens_must_match_spatial_grid")
        grid = tokens.reshape(len(tokens), height, width, tokens.shape[-1]).permute(0, 3, 1, 2)
        pooled = functional.adaptive_avg_pool2d(
            grid.float(), (self.spatial_grid, self.spatial_grid)
        )
        return (
            pooled.permute(0, 2, 3, 1)
            .reshape(len(tokens), self.spatial_grid**2, tokens.shape[-1])
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    def _encode_vjepa2(self, images: np.ndarray) -> np.ndarray:
        from torch.nn import functional

        torch = self.torch
        frames = (
            torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div_(255.0)
        )
        frames = functional.interpolate(
            frames, size=(292, 292), mode="bilinear", align_corners=False
        )[:, :, 18:274, 18:274]
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        video = ((frames - mean) / std).to(self.device).unsqueeze(1).repeat(1, 2, 1, 1, 1)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")
            ),
        ):
            tokens = self.model.get_vision_features(video)
        side = round(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise ValueError("public_mt50_vjepa_tokens_must_form_square_grid")
        return self._pool_tokens(tokens, height=side, width=side)

    def _encode_dinov2(self, images: np.ndarray) -> np.ndarray:
        torch = self.torch
        if self.image_processor is None:
            raise RuntimeError("public_mt50_dinov2_image_processor_missing")
        processed = self.image_processor(images=list(images), return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self.device)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")
            ),
        ):
            encoded = self.model(pixel_values=pixel_values)
        tokens, height, width = spatial_patch_tokens(
            encoded.last_hidden_state,
            image_shape=(int(pixel_values.shape[-2]), int(pixel_values.shape[-1])),
            patch_size=int(self.model.config.patch_size),
        )
        return self._pool_tokens(tokens, height=height, width=width)

    def encode_rgb(self, images: np.ndarray, *, batch_size: int = 32) -> np.ndarray:
        images = self._validate_images(images, batch_size)
        outputs: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            if self.backbone == VJEPA2_BACKBONE:
                outputs.append(self._encode_vjepa2(batch))
            elif self.backbone == DINOV2_BACKBONE:
                outputs.append(self._encode_dinov2(batch))
            else:
                raise AssertionError(self.backbone)
        if not outputs:
            return np.empty((0, self.spatial_grid**2, self.feature_dim), dtype=np.float32)
        return np.concatenate(outputs, axis=0)
