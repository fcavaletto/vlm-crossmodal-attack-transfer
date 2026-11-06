"""Vision-language scoring, input-image gradients, and generation parsing.

The module-level functions are the Checkpoint 0 contract:

- ``answer_logprob(image, prompt, answer_text) -> float``
- ``image_loss_gradient(image, prompt, answer_text) -> tensor``
- ``generate(image, prompt, task_family=...) -> {"text", "parsed_answer", "invalid"}``

They use a process-wide session that loads the pinned SmolVLM2 revision on
first call. Tests and scripts can pass ``session=`` instead.
"""

from __future__ import annotations

from .parsing import parse_generation
from .pins import MODEL_ID, PINNED_REVISION
from .session import (
    VLMSession,
    answer_logprob,
    configure_default_session,
    generate,
    get_default_session,
    image_loss_gradient,
)

__all__ = [
    "MODEL_ID",
    "PINNED_REVISION",
    "VLMSession",
    "answer_logprob",
    "configure_default_session",
    "generate",
    "get_default_session",
    "image_loss_gradient",
    "parse_generation",
]
