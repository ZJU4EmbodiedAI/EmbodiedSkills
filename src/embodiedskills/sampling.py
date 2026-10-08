"""Independent proposal seeds with the direct action in slot zero."""

from __future__ import annotations

import hashlib

CANDIDATE_COUNT = 5


def _require_integer(value: int, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name}_must_be_an_integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name}_must_be_at_least_{minimum}")
    return result


def paired_direct_sampling_seed(seed_base: int, environment_seed: int, anchor_index: int) -> int:
    """Return the seeded direct-pi0.5 request for one public decision.

    This is intentionally the same arithmetic identity as the direct policy
    evaluator.  It is *not* a candidate-specific namespace: every paired
    direct run must issue this exact request at the corresponding decision.
    """
    base = _require_integer(seed_base, name="seed_base", minimum=0)
    environment = _require_integer(environment_seed, name="environment_seed", minimum=0)
    anchor = _require_integer(anchor_index, name="anchor_index", minimum=0)
    return base + environment * 1000 + anchor


def _seed_digest(*parts: object) -> int:
    """Stable seed in the existing medoid independent-candidate namespace."""
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(
        hashlib.blake2s(payload, digest_size=8, person=b"esmedoid").digest()[:4], "big"
    )
    return int(value % 2000000000) + 1


def candidate_sampling_seeds(
    *, seed_base: int, environment_seed: int, anchor_index: int, count: int = CANDIDATE_COUNT
) -> list[int]:
    """Return physical candidate seeds: paired direct first, four independent.

    Inputs are registered scalar identities only.  No observation, action,
    reward, success flag, or privileged simulator state can affect the pool.
    """
    count = _require_integer(count, name="candidate_count", minimum=1)
    if count != CANDIDATE_COUNT:
        raise ValueError("public_mt50_requires_exactly_five_candidates")
    base = _require_integer(seed_base, name="seed_base", minimum=0)
    environment = _require_integer(environment_seed, name="environment_seed", minimum=0)
    anchor = _require_integer(anchor_index, name="anchor_index", minimum=0)
    seeds = [paired_direct_sampling_seed(base, environment, anchor)]
    for candidate_index in range(1, count):
        candidate_seed = _seed_digest(
            "public-pi05-medoid-candidate-v1", base, environment, anchor, candidate_index
        )
        while candidate_seed in seeds:
            candidate_seed = 1 if candidate_seed >= 2000000000 else candidate_seed + 1
        seeds.append(candidate_seed)
    if len(set(seeds)) != count:
        raise RuntimeError("public_mt50_candidate_sampling_seeds_not_unique")
    return seeds
