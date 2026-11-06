"""Pinned SmolVLM2 and Qwen3-VL-2B revisions.

The revisions are the ones already recorded for this study. This module does
not load either checkpoint. Qwen remains a scoring target of the prospective
protocol; naming it here does not adopt it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPin:
    key: str
    model_id: str
    revision: str


SMOLVLM2 = ModelPin(
    key="smolvlm2",
    model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467",
)
QWEN3_VL_2B = ModelPin(
    key="qwen3-vl-2b",
    model_id="Qwen/Qwen3-VL-2B-Instruct",
    revision="89644892e4d85e24eaac8bacfd4f463576704203",
)
MODELS: tuple[ModelPin, ...] = (SMOLVLM2, QWEN3_VL_2B)
BY_KEY: dict[str, ModelPin] = {model.key: model for model in MODELS}


def require_model(key: str) -> ModelPin:
    """Return the pin for ``smolvlm2`` or ``qwen3-vl-2b``."""
    try:
        return BY_KEY[key]
    except KeyError:
        known = ", ".join(model.key for model in MODELS)
        raise ValueError(f"unknown model {key!r}; expected {known}") from None
