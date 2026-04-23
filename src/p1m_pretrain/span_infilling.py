from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence


ATTACHMENT_TOKEN = "[At]"
NUM_SENTINEL_TOKENS = 100
SENTINEL_TOKENS = [f"<extra_id_{idx}>" for idx in range(NUM_SENTINEL_TOKENS)]


@dataclass(frozen=True)
class SpanInfillingExample:
    input_tokens: list[str]
    target_tokens: list[str]
    masked_region_indices: list[int]


def split_attachment_regions(tokens: Sequence[str], attachment_token: str = ATTACHMENT_TOKEN) -> tuple[list[list[str]], list[str]]:
    regions: list[list[str]] = [[]]
    attachments: list[str] = []
    for token in tokens:
        if token == attachment_token:
            attachments.append(token)
            regions.append([])
        else:
            regions[-1].append(token)
    return regions, attachments


def build_span_infilling_example(
    tokens: Sequence[str],
    *,
    noise_density: float,
    rng: random.Random | None = None,
    attachment_token: str = ATTACHMENT_TOKEN,
    sentinel_tokens: Sequence[str] = SENTINEL_TOKENS,
) -> SpanInfillingExample:
    if not 0.0 <= noise_density <= 1.0:
        raise ValueError(f"noise_density must be in [0, 1], got {noise_density}")

    rng = rng or random
    regions, attachments = split_attachment_regions(tokens, attachment_token=attachment_token)
    mask_flags = [rng.random() < noise_density for _ in regions]
    if not any(mask_flags):
        non_empty_regions = [idx for idx, region in enumerate(regions) if region]
        fallback = non_empty_regions or list(range(len(regions)))
        if fallback:
            mask_flags[rng.choice(fallback)] = True

    masked_region_indices = [idx for idx, is_masked in enumerate(mask_flags) if is_masked]
    if len(masked_region_indices) + 1 > len(sentinel_tokens):
        raise ValueError(
            f"Need {len(masked_region_indices) + 1} sentinel tokens but only {len(sentinel_tokens)} are available"
        )

    input_tokens: list[str] = []
    target_tokens: list[str] = []
    sentinel_idx = 0
    for region_idx, region_tokens in enumerate(regions):
        if mask_flags[region_idx]:
            sentinel = sentinel_tokens[sentinel_idx]
            sentinel_idx += 1
            input_tokens.append(sentinel)
            target_tokens.append(sentinel)
            target_tokens.extend(region_tokens)
        else:
            input_tokens.extend(region_tokens)
        if region_idx < len(attachments):
            input_tokens.append(attachments[region_idx])

    target_tokens.append(sentinel_tokens[sentinel_idx])
    return SpanInfillingExample(
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        masked_region_indices=masked_region_indices,
    )
