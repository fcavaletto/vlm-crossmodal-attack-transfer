"""One-token closed-vocabulary sums and the generation-cue check.

The forward that produces the logits lives in the runner. This module turns a
logit row into the four sums and checks the cue before any development scene.
"""

from __future__ import annotations

from typing import Mapping

from choice_eval.colors import COLORS
from choice_eval.likelihood import mean_nll, summed_nll
from choice_eval.prompt import build_prompt
from choice_eval.tokenize import verify_model_tokenization

from .lock import ANSWER_PREFIX, EXPECTED_TOKEN_IDS, PROBE_CONTEXT, PROBE_QUESTION


def nlls_from_first_logit(logits, token_ids: Mapping[str, tuple[int, ...] | list[int]]) -> tuple[dict[str, float], dict[str, float]]:
    """Summed and mean NLL of four one-token colors from the first-answer logit row."""
    import torch

    if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
        raise ValueError("the first continuation logit must be a vector")
    predicting = logits.detach().to(dtype=torch.float32, device="cpu").unsqueeze(0)
    if not torch.isfinite(predicting).all():
        raise RuntimeError("continuation logits are not finite")
    summed: dict[str, float] = {}
    mean: dict[str, float] = {}
    for color in COLORS:
        ids = [int(token) for token in token_ids[color]]
        if len(ids) != 1:
            raise RuntimeError("the locked continuations are one token")
        summed[color] = float(summed_nll(predicting, ids).detach().cpu())
        mean[color] = float(mean_nll(predicting, ids).detach().cpu())
    return summed, mean


def probe_user_text() -> str:
    """A sentence that is not a development_v4 question."""
    return build_prompt(PROBE_CONTEXT, PROBE_QUESTION).user_text()


def generation_suffix(processor, user_text: str) -> str:
    """Suffix after ``user_text`` in the image-turn generation prompt."""
    from PIL import Image

    image = Image.new("RGB", (8, 8), "black")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if isinstance(rendered, list):
        rendered = rendered[0] if rendered else ""
    if not isinstance(rendered, str) or rendered == "":
        raise RuntimeError(f"the chat template did not return a string ({type(rendered).__name__})")
    index = rendered.rfind(user_text)
    if index < 0:
        raise RuntimeError("the probe user text was not in the rendered prompt")
    suffix = rendered[index + len(user_text) :]
    if suffix == "":
        raise RuntimeError("the generation cue is empty")
    return suffix


def assert_locked_tokenization(processor, model_key: str) -> dict[str, object]:
    """Stop before any development scene if the cue or the color ids differ."""
    suffix = generation_suffix(processor, probe_user_text())
    expected_prefix = ANSWER_PREFIX[model_key]
    if suffix != expected_prefix:
        raise RuntimeError(
            "generation cue does not match the protocol. "
            f"expected {expected_prefix!r}, got {suffix!r}"
        )
    tokenizer = processor.tokenizer
    report = verify_model_tokenization(tokenizer, model_key=model_key, answer_prefix=suffix)
    expected_ids = EXPECTED_TOKEN_IDS[model_key]
    selected = {item.color: item.token_ids for item in report.selected}
    if selected != expected_ids:
        raise RuntimeError(f"selected token ids {selected} do not match the protocol {expected_ids}")
    image_processor = getattr(processor, "image_processor", None)
    return {
        "answer_prefix": suffix,
        "token_ids": {color: list(ids) for color, ids in selected.items()},
        "continuations": {item.color: item.text for item in report.selected},
        "processor_class": type(processor).__name__,
        "image_processor_class": None if image_processor is None else type(image_processor).__name__,
        "any_selected_multi_token": report.any_selected_multi_token,
    }
