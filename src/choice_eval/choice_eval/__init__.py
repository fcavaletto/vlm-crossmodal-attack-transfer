"""Prospective closed-vocabulary scoring for Checkpoint 1.7.

The primary score is the teacher-forced summed NLL of one full color
sequence. Free-generation parsing stays a secondary diagnostic. This package
does not load weights and does not run an attack.
"""

from .decision import choose_color, rank_colors
from .diagnostic import diagnose_free_generation
from .likelihood import mean_nll, summed_nll, summed_nll_from_causal_logits
from .models import MODELS, QWEN3_VL_2B, SMOLVLM2
from .prompt import INSTRUCTION, build_prompt, continuation_for, scoring_inputs
from .protocol import PRIMARY_REDUCTION, PROTOCOL_ID, PROTOCOL_STATUS
from .score import score_example, summarize
from .tokenize import TokenizationError, verify_model_tokenization

__all__ = [
    "INSTRUCTION",
    "MODELS",
    "PRIMARY_REDUCTION",
    "PROTOCOL_ID",
    "PROTOCOL_STATUS",
    "QWEN3_VL_2B",
    "SMOLVLM2",
    "TokenizationError",
    "build_prompt",
    "choose_color",
    "continuation_for",
    "diagnose_free_generation",
    "mean_nll",
    "rank_colors",
    "score_example",
    "scoring_inputs",
    "summarize",
    "summed_nll",
    "summed_nll_from_causal_logits",
    "verify_model_tokenization",
]
