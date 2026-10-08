"""Observation and action adapters for frozen policy proposals."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

CLIPORT_CANDIDATES = 5


def cliport_step_progress(step_index: int, max_steps: int) -> np.ndarray:
    """Encode deployment-visible time progress without task-state leakage."""
    step = int(step_index)
    budget = int(max_steps)
    if step < 0 or budget <= 0 or step > budget:
        raise ValueError("cliport_step_progress_values_invalid")
    fraction = step / budget
    return np.asarray((fraction, 1.0 - fraction), dtype=np.float32)


def cliport_fused_rgb(fused_image: np.ndarray, *, size: int = 256) -> np.ndarray:
    """Convert CLIPort's fused RGB-D heightmap to encoder-ready RGB."""
    from PIL import Image

    fused = np.asarray(fused_image)
    if fused.ndim != 3 or fused.shape[2] != 6:
        raise ValueError("cliport_fused_image_must_be_H_W_6")
    if not np.isfinite(fused).all():
        raise ValueError("cliport_fused_image_contains_nonfinite_values")
    rgb = np.clip(fused[:, :, :3], 0, 255).astype(np.uint8)
    if size <= 0:
        raise ValueError("cliport_rgb_size_must_be_positive")
    if rgb.shape[:2] != (size, size):
        rgb = np.asarray(
            Image.fromarray(rgb).resize((size, size), resample=Image.Resampling.BILINEAR)
        )
    return np.ascontiguousarray(rgb)


def cliport_candidate_vector(
    candidate_pose: np.ndarray,
    candidate_pixel_pose: np.ndarray,
    *,
    image_height: int = 320,
    image_width: int = 160,
) -> np.ndarray:
    """Build a physical action vector without policy scores or branch values."""
    pose = np.asarray(candidate_pose, dtype=np.float32)
    pixel = np.asarray(candidate_pixel_pose, dtype=np.float32)
    if pose.shape != (CLIPORT_CANDIDATES, 2, 7):
        raise ValueError("cliport_candidate_pose_must_be_5_2_7")
    if pixel.shape != (CLIPORT_CANDIDATES, 6):
        raise ValueError("cliport_candidate_pixel_pose_must_be_5_6")
    if not all(np.isfinite(value).all() for value in (pose, pixel)):
        raise ValueError("cliport_candidate_vector_contains_nonfinite_values")
    pixel_scale = np.asarray(
        [
            max(image_height - 1, 1),
            max(image_width - 1, 1),
            2 * math.pi,
            max(image_height - 1, 1),
            max(image_width - 1, 1),
            2 * math.pi,
        ],
        dtype=np.float32,
    )
    pixel_normalized = pixel / pixel_scale[None]
    return np.concatenate((pose.reshape(CLIPORT_CANDIDATES, -1), pixel_normalized), axis=1).astype(
        np.float32, copy=False
    )


def cliport_public_height_state4(fused_image: np.ndarray) -> np.ndarray:
    """Return four deployment-visible statistics of CLIPort's height channel."""
    fused = np.asarray(fused_image)
    if fused.ndim != 3 or fused.shape[2] != 6:
        raise ValueError("cliport_mt50_fused_image_must_be_H_W_6")
    if not np.isfinite(fused).all():
        raise ValueError("cliport_mt50_fused_image_must_be_finite")
    height = fused[..., 3].astype(np.float64, copy=False)
    return np.asarray(
        (height.mean(), height.std(), height.max(), np.count_nonzero(height) / float(height.size)),
        dtype=np.float32,
    )


@dataclass(frozen=True)
class SpatialPeak:
    """One local maximum in a row-column-rotation probability volume."""

    row: int
    col: int
    rotation: int
    probability: float

    @property
    def pixel(self) -> tuple[int, int]:
        return (self.row, self.col)


@dataclass(frozen=True)
class JointHypothesis:
    """A native CLIPort pick/place hypothesis and its joint log score."""

    pick: SpatialPeak
    place: SpatialPeak
    joint_log_probability: float


def top_spatial_local_maxima(
    probabilities: np.ndarray, *, limit: int, min_pixel_distance: float, min_rotation_bins: int = 1
) -> list[SpatialPeak]:
    """Return stable non-maximum-suppressed peaks from an ``H x W x R`` map.

    Suppression is joint in image position and circular rotation. Peaks at the
    same pixel remain distinct only when their rotations are sufficiently far
    apart. The global argmax is therefore always the first returned item.
    """
    volume = np.asarray(probabilities, dtype=np.float64)
    if volume.ndim == 2:
        volume = volume[..., None]
    if volume.ndim != 3:
        raise ValueError(f"expected_probability_volume_hwr_got_{volume.shape}")
    if limit <= 0:
        raise ValueError("limit_must_be_positive")
    if min_pixel_distance < 0:
        raise ValueError("min_pixel_distance_must_be_nonnegative")
    if min_rotation_bins < 0:
        raise ValueError("min_rotation_bins_must_be_nonnegative")
    if not np.all(np.isfinite(volume)):
        raise ValueError("probability_volume_contains_nonfinite_values")
    if np.any(volume < 0):
        raise ValueError("probability_volume_contains_negative_values")
    order = np.argsort(-volume.reshape(-1), kind="stable")
    rotations = volume.shape[2]
    selected: list[SpatialPeak] = []
    for flat_index in order:
        row, col, rotation = np.unravel_index(int(flat_index), volume.shape)
        candidate = SpatialPeak(
            row=int(row),
            col=int(col),
            rotation=int(rotation),
            probability=float(volume[row, col, rotation]),
        )
        suppressed = False
        for peak in selected:
            pixel_distance = math.hypot(candidate.row - peak.row, candidate.col - peak.col)
            rotation_distance = abs(candidate.rotation - peak.rotation)
            rotation_distance = min(rotation_distance, rotations - rotation_distance)
            if pixel_distance < min_pixel_distance and rotation_distance < min_rotation_bins:
                suppressed = True
                break
        if not suppressed:
            selected.append(candidate)
            if len(selected) == limit:
                break
    return selected


def make_joint_hypothesis(pick: SpatialPeak, place: SpatialPeak) -> JointHypothesis:
    """Combine native attention and transport probabilities in log space."""
    tiny = np.finfo(np.float64).tiny
    score = math.log(max(pick.probability, tiny)) + math.log(max(place.probability, tiny))
    return JointHypothesis(pick=pick, place=place, joint_log_probability=score)


def select_fixed_joint_hypotheses(
    *,
    canonical: JointHypothesis,
    hypotheses: Sequence[JointHypothesis],
    count: int = 5,
    min_pick_distance: float = 4.0,
    min_place_distance: float = 4.0,
    min_place_rotation_bins: int = 2,
    place_rotation_count: int = 36,
) -> list[JointHypothesis]:
    """Keep exact argmax first, then the best distinct native beam entries."""
    if count <= 0:
        raise ValueError("count_must_be_positive")
    if place_rotation_count <= 0:
        raise ValueError("place_rotation_count_must_be_positive")
    ordered = sorted(
        hypotheses,
        key=lambda item: (
            -item.joint_log_probability,
            item.pick.row,
            item.pick.col,
            item.place.row,
            item.place.col,
            item.place.rotation,
        ),
    )
    selected = [canonical]
    for candidate in ordered:
        if candidate == canonical:
            continue
        too_similar = False
        for retained in selected:
            pick_distance = math.hypot(
                candidate.pick.row - retained.pick.row, candidate.pick.col - retained.pick.col
            )
            place_distance = math.hypot(
                candidate.place.row - retained.place.row, candidate.place.col - retained.place.col
            )
            rotation_distance = abs(candidate.place.rotation - retained.place.rotation)
            rotation_distance = min(rotation_distance, place_rotation_count - rotation_distance)
            if (
                pick_distance < min_pick_distance
                and place_distance < min_place_distance
                and (rotation_distance < min_place_rotation_bins)
            ):
                too_similar = True
                break
        if not too_similar:
            selected.append(candidate)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(f"insufficient_distinct_native_hypotheses:{len(selected)}_of_{count}")
    return selected
