"""Toy-data statistics for cross-modal attack transfer.

This package scores tables. It does not load a vision-language model, does not
differentiate through one, and does not run an attack.
"""

from .bootstrap import paired_mean_difference, scene_bootstrap_mean, scene_bootstrap_rate
from .metrics import (
    additional_error_by_scene,
    attack_cells,
    attack_success_indicators,
    attack_success_rate,
    clean_by_context,
    conditional_transfer,
    cross_context_attack_transfer,
    mean_alignment,
    mean_loss_by_context,
    sign_alignment,
)
from .schemas import (
    CONTEXTS,
    SchemaError,
    epsilon_key,
    load_jsonl,
    validate_attacks,
    validate_gradient_summaries,
    validate_gradient_vectors,
    validate_losses,
    validate_predictions,
)

__version__ = "0.0.1"

__all__ = [
    "CONTEXTS",
    "SchemaError",
    "additional_error_by_scene",
    "attack_cells",
    "attack_success_indicators",
    "attack_success_rate",
    "clean_by_context",
    "conditional_transfer",
    "cross_context_attack_transfer",
    "epsilon_key",
    "load_jsonl",
    "mean_alignment",
    "mean_loss_by_context",
    "paired_mean_difference",
    "scene_bootstrap_mean",
    "scene_bootstrap_rate",
    "sign_alignment",
    "validate_attacks",
    "validate_gradient_summaries",
    "validate_gradient_vectors",
    "validate_losses",
    "validate_predictions",
]
