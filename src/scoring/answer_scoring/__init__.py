"""Checkpoint 1.5 answer scoring.

The strict parser and the canonical-word loss stay available. Alias surfaces
and the format-instruction prompt are additional records. Nothing here selects
a rule by accuracy, runs an attack, or fits H*.
"""

from .class_loss import class_nll, word_only_nll
from .instruction import FORMAT_INSTRUCTION, format_user_prompt
from .reading import side_by_side

__all__ = [
    "FORMAT_INSTRUCTION",
    "class_nll",
    "format_user_prompt",
    "side_by_side",
    "word_only_nll",
]
