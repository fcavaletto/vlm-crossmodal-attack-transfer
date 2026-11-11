"""Checkpoint 1.6 closed-vocabulary scoring.

The official primary metric is the strict parse of greedy text. The class loss
is the differentiable closed-vocabulary scalar. Its argmin is stored beside
that parse. Nothing here loads weights or runs an attack.
"""

from .decision import argmin_class, class_decision, is_disagreement
from .generation import score_free_generation
from .loss import class_nll_from_surfaces, finite_difference_positions, single_position_class_nll, word_only_nll
from .protocol import (
    CLASS_ARGMIN_IS_PRIMARY,
    DESIGN_BAND,
    DISAGREEMENT_TOLERANCE,
    EMBEDDED_READING_ADOPTED,
    FREE_GENERATION_ROLE,
    OFFICIAL_METRIC,
)
from .vocabulary import task_vocabulary

__all__ = [
    "CLASS_ARGMIN_IS_PRIMARY",
    "DESIGN_BAND",
    "DISAGREEMENT_TOLERANCE",
    "EMBEDDED_READING_ADOPTED",
    "FREE_GENERATION_ROLE",
    "OFFICIAL_METRIC",
    "argmin_class",
    "class_decision",
    "class_nll_from_surfaces",
    "finite_difference_positions",
    "is_disagreement",
    "score_free_generation",
    "single_position_class_nll",
    "task_vocabulary",
    "word_only_nll",
]
