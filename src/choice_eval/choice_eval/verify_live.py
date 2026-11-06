"""Tokenize the pinned SmolVLM2 and Qwen3-VL-2B checkpoints.

Downloads tokenizer files only. Refuses weight files. Does not run a forward
pass, a generation, or an image gradient. The unit tests do not call this
module's ``main``.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import torch

from .models import MODELS, ModelPin
from .prompt import build_prompt, continuation_for
from .tokenize import encode_text, verify_model_tokenization

ALLOW_PATTERNS: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "config.json",
    "generation_config.json",
    "vocab.txt",
    "spiece.model",
)
IGNORE_PATTERNS: tuple[str, ...] = (
    "*.safetensors",
    "*.safetensors.index.json",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.gguf",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "*.ot",
)
WEIGHT_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".gguf",
    ".onnx",
    ".h5",
    ".msgpack",
    ".ot",
}
MAX_FILE_BYTES = 40_000_000
USER_CONTEXT = "Several simple objects appear on a plain background."
USER_QUESTION = "What color is the circle in the top row?"


def assert_no_weight_files(root: Path, *, max_file_bytes: int = MAX_FILE_BYTES) -> None:
    """Raise if a snapshot contains a weight file or an unexpectedly large file."""
    if not root.exists():
        raise RuntimeError(f"snapshot directory is missing: {root}")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if path.suffix.lower() in WEIGHT_SUFFIXES or name.endswith(".safetensors.index.json"):
            raise RuntimeError(f"weight file was downloaded: {path.name}")
        size = path.stat().st_size
        if size > max_file_bytes:
            raise RuntimeError(f"file exceeds tokenizer-sized cap: {path.name} ({size} bytes)")


def main(argv: list[str] | None = None) -> int:
    """Print one JSON object and return 0 when both tokenizers verify."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        raise SystemExit("verify_live takes no arguments and does not load weights")
    payload: dict[str, object] = {"weights_loaded": False, "models": []}
    failed = False
    models: list[dict[str, object]] = []
    for model in MODELS:
        try:
            models.append(_survey_model(model))
        except Exception as exc:
            failed = True
            models.append(
                {
                    "model_key": model.key,
                    "model_id": model.model_id,
                    "revision": model.revision,
                    "weights_loaded": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
    payload["models"] = models
    json.dump(_jsonable(payload), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 1 if failed else 0


def _survey_model(model: ModelPin) -> dict[str, object]:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    with tempfile.TemporaryDirectory(prefix=f"choice-eval-{model.key}-") as temporary:
        local = Path(temporary) / "snapshot"
        snapshot_download(
            repo_id=model.model_id,
            revision=model.revision,
            allow_patterns=list(ALLOW_PATTERNS),
            ignore_patterns=list(IGNORE_PATTERNS),
            local_dir=str(local),
        )
        assert_no_weight_files(local)
        files = sorted(
            path.name
            for path in local.rglob("*")
            if path.is_file() and _reportable_file(path.name)
        )
        tokenizer = AutoTokenizer.from_pretrained(str(local), local_files_only=True)
        user_text = build_prompt(USER_CONTEXT, USER_QUESTION).user_text()
        template = _generation_suffix(tokenizer, user_text)
        result: dict[str, object] = {
            "model_key": model.key,
            "model_id": model.model_id,
            "revision": model.revision,
            "weights_loaded": False,
            "files": files,
            "vocab_size": int(len(tokenizer)),
            "template": template,
        }
        suffix = template.get("generation_suffix")
        messages = template.pop("messages", None)
        if template.get("suffixes_agree") is False:
            raise RuntimeError("generation suffix depends on the chat message shape: " + repr(template.get("suffixes_by_attempt")))
        if isinstance(suffix, str) and suffix != "":
            report = verify_model_tokenization(tokenizer, model_key=model.key, answer_prefix=suffix)
            result["tokenization"] = _report_dict(report)
            result["template_continuation"] = _continuation_match(
                tokenizer, messages, suffix, template.get("rendered")
            )
        else:
            bare = {color: list(encode_text(tokenizer, color)) for color in ("red", "blue", "green", "yellow")}
            spaced = {
                color: list(encode_text(tokenizer, " " + color)) for color in ("red", "blue", "green", "yellow")
            }
            result["bare_ids"] = bare
            result["leading_space_ids"] = spaced
            result["tokenization"] = None
            raise RuntimeError("chat template did not yield a generation suffix; candidate convention was not selected")
        return result


def _reportable_file(name: str) -> bool:
    if name.startswith(".") or name in {"CACHEDIR.TAG"}:
        return False
    return not name.endswith((".lock", ".metadata"))


_ATTEMPT_RANK = ("image-marker", "pil-image", "text-part", "string")


def prefer_generation_suffix(found: list[dict[str, object]]) -> dict[str, object]:
    """Prefer the image chat shape. Fall back to a text-only render."""
    if not found:
        raise ValueError("no rendered prompts")

    def rank(item: dict[str, object]) -> int:
        attempt = str(item.get("attempt"))
        return _ATTEMPT_RANK.index(attempt) if attempt in _ATTEMPT_RANK else len(_ATTEMPT_RANK)

    chosen = dict(min(found, key=rank))
    suffixes = {str(item["attempt"]): item["generation_suffix"] for item in found}
    chosen["suffixes_by_attempt"] = suffixes
    chosen["suffixes_agree"] = len(set(suffixes.values())) == 1
    return chosen


def _generation_suffix(tokenizer, user_text: str) -> dict[str, object]:
    errors: list[str] = []
    found: list[dict[str, object]] = []
    for name, messages in _message_attempts(user_text):
        try:
            rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if isinstance(rendered, list):
            rendered = rendered[0] if rendered else ""
        if not isinstance(rendered, str):
            errors.append(f"{name}: render returned {type(rendered).__name__}")
            continue
        index = rendered.rfind(user_text)
        if index < 0:
            errors.append(f"{name}: user text was not in the rendered prompt")
            continue
        suffix = rendered[index + len(user_text) :]
        found.append(
            {
                "attempt": name,
                "rendered": rendered[:2000],
                "generation_suffix": suffix,
                "suffix_ends_with_whitespace": bool(suffix) and suffix[-1].isspace(),
                "messages": messages,
            }
        )
    if not found:
        return {"attempt": None, "generation_suffix": None, "errors": errors, "messages": None}
    chosen = prefer_generation_suffix(found)
    chosen["errors"] = errors
    return chosen


def _message_attempts(user_text: str):
    yield "string", [{"role": "user", "content": user_text}]
    yield "text-part", [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    yield "image-marker", [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
    try:
        from PIL import Image
    except ImportError:
        return
    image = Image.new("RGB", (8, 8), "black")
    yield (
        "pil-image",
        [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]}],
    )


def _continuation_match(tokenizer, messages: object, suffix: str, rendered: object) -> dict[str, object]:
    continuation = continuation_for("red", answer_prefix=suffix)
    separate = list(encode_text(tokenizer, continuation))
    report: dict[str, object] = {"continuation": continuation, "separate_ids": separate}
    if isinstance(rendered, str) and rendered != "":
        raw_prompt = list(encode_text(tokenizer, rendered))
        joint = list(encode_text(tokenizer, rendered + continuation))
        report["raw_prompt_plus_continuation_equals_joint"] = raw_prompt + separate == joint
        report["joint_tail_ids"] = joint[len(raw_prompt) :] if joint[: len(raw_prompt)] == raw_prompt else None
    if not isinstance(messages, list):
        report["assistant_turn_error"] = "no user messages were recorded"
        return report
    try:
        prompt_ids = _template_ids(tokenizer, messages, add_generation_prompt=True)
        full_ids = _template_ids(
            tokenizer,
            [*messages, {"role": "assistant", "content": continuation}],
            add_generation_prompt=False,
        )
    except Exception as exc:
        report["assistant_turn_error"] = f"{type(exc).__name__}: {exc}"
        return report
    prompt_is_prefix = full_ids[: len(prompt_ids)] == prompt_ids
    tail = full_ids[len(prompt_ids) :] if prompt_is_prefix else []
    continuation_is_prefix = prompt_is_prefix and tail[: len(separate)] == separate
    extra = tail[len(separate) :] if continuation_is_prefix else tail
    report.update(
        {
            "assistant_turn_prompt_is_prefix": prompt_is_prefix,
            "assistant_turn_includes_continuation": continuation_is_prefix,
            "assistant_turn_tail_ids": tail,
            "assistant_turn_extra_ids": extra,
            "prompt_length": len(prompt_ids),
            "assistant_turn_length": len(full_ids),
        }
    )
    return report


def _template_ids(tokenizer, messages: list[dict[str, object]], *, add_generation_prompt: bool) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_tensors="pt",
    )
    if isinstance(rendered, torch.Tensor):
        tensor = rendered
    elif isinstance(rendered, list):
        row = rendered[0] if rendered and isinstance(rendered[0], list) else rendered
        return [int(token) for token in row]
    else:
        tensor = rendered["input_ids"]
    if tensor.ndim == 2:
        tensor = tensor[0]
    return [int(token) for token in tensor.tolist()]


def _report_dict(report) -> dict[str, object]:
    def encoded(items) -> list[dict[str, object]]:
        return [
            {
                "color": item.color,
                "text": item.text,
                "token_ids": list(item.token_ids),
                "pieces": None if item.pieces is None else list(item.pieces),
                "decoded": item.decoded,
                "multi_token": item.multi_token,
            }
            for item in items
        ]

    return {
        "answer_prefix": report.answer_prefix,
        "prefix_ends_with_whitespace": report.prefix_ends_with_whitespace,
        "selected": encoded(report.selected),
        "bare": encoded(report.bare),
        "leading_space": encoded(report.leading_space),
        "leading_space_changes_ids": list(report.leading_space_changes_ids),
        "leading_space_changes_every_color": report.leading_space_changes_every_color,
        "any_selected_multi_token": report.any_selected_multi_token,
        "joint_matches_appended": list(report.joint_matches_appended),
        "joint_matches_every_color": report.joint_matches_every_color,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
