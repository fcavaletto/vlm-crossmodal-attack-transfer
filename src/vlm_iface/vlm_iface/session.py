"""Session that scores an answer, differentiates it, and parses a generation."""

from __future__ import annotations

from typing import Any

import torch

from .devices import resolve_device
from .images import image_01_to_pil, to_image_01
from .load import load_pinned_model
from .logprob import sequence_logprob
from .parsing import parse_generation
from .pins import MODEL_ID, PINNED_REVISION
from .preprocess import differentiable_pixel_values
from .tokens import answer_token_ids


class VLMSession:
    """Frozen eval-mode session. One image per call. Not thread-safe.

    Passing ``model`` and ``processor`` skips the download. That hook is for
    tests. The default constructor loads the pinned SmolVLM2 revision.
    """

    def __init__(
        self,
        *,
        model=None,
        processor=None,
        device: str = "auto",
        do_image_splitting: bool = False,
        max_new_tokens: int = 16,
        leading_space: bool = True,
        model_id: str = MODEL_ID,
        revision: str = PINNED_REVISION,
    ) -> None:
        if (model is None) ^ (processor is None):
            raise ValueError("Pass both model and processor, or neither.")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.device = resolve_device(device)
        self.do_image_splitting = do_image_splitting
        self.max_new_tokens = max_new_tokens
        self.leading_space = leading_space
        self.model_id = model_id
        self.revision = revision
        if model is None:
            self.processor, self.model = load_pinned_model(
                self.device, model_id=model_id, revision=revision
            )
        else:
            self.processor = processor
            self.model = model
            self.model.to(self.device)
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def answer_logprob(self, image, prompt: str, answer_text: str) -> float:
        """Joint log-probability of the full answer, as a Python float."""
        with torch.no_grad():
            logprob, _n_tokens = self._logprob_tensor(image, prompt, answer_text, enable_grad=False)
        value = float(logprob)
        if value != value:  # NaN
            raise RuntimeError("answer log-probability is not finite")
        return value

    def image_loss_gradient(self, image, prompt: str, answer_text: str) -> torch.Tensor:
        """Gradient of ``L = -answer_logprob`` with respect to the ``[0, 1]`` RGB image.

        The returned tensor is detached float32 on CPU, shaped ``(3, height, width)``.
        It is the derivative in that canonical space, not in processor-normalized space.
        """
        logprob, _n_tokens = self._logprob_tensor(image, prompt, answer_text, enable_grad=True)
        loss = -logprob.sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Loss is not finite; image gradient was not computed.")
        loss.backward()
        gradient = self._leaf.grad
        if gradient is None:
            raise RuntimeError("Image gradient is missing. The preprocess graph does not reach the image.")
        if not torch.isfinite(gradient).all():
            raise RuntimeError("Image gradient contains a non-finite value.")
        result = gradient.detach().to(dtype=torch.float32, device="cpu").contiguous()
        self._leaf = None
        return result

    def generate(self, image, prompt: str, *, task_family: str) -> dict[str, str | bool | None]:
        """Greedy generation parsed with the task vocabulary.

        The dict keys are ``text``, ``parsed_answer``, and ``invalid``.
        ``task_family`` selects the Checkpoint 1 vocabulary. Scoring and this
        call share ``_bind``, so they use the same float32 pixels.
        """
        return {
            key: self.generate_detailed(image, prompt, task_family=task_family)[key]
            for key in ("text", "parsed_answer", "invalid")
        }

    def generate_detailed(self, image, prompt: str, *, task_family: str) -> dict[str, Any]:
        """Greedy generation plus parse reasons, token ids, and the truncation flag."""
        bound = self._bind(image, prompt, enable_grad=False)
        inputs = {
            "input_ids": bound["input_ids"],
            "attention_mask": bound["attention_mask"],
            "pixel_values": bound["pixel_values"].detach(),
            "pixel_attention_mask": bound["pixel_attention_mask"],
        }
        eos_ids = _end_token_ids(self.model, self.processor.tokenizer)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        new_ids = _continuation_ids(generated, inputs["input_ids"])
        text = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        token_ids = [int(token) for token in new_ids.tolist()]
        truncated = not any(token in eos_ids for token in token_ids)
        parsed = parse_generation(text, truncated=truncated, task_family=task_family)
        detailed = parsed.as_contract()
        detailed["reasons"] = list(parsed.reasons)
        detailed["token_ids"] = token_ids
        detailed["truncated"] = truncated
        return detailed

    def score_details(self, image, prompt: str, answer_text: str) -> dict[str, Any]:
        """Log-probability, token ids, and the library mean NLL on the same logits."""
        with torch.no_grad():
            bound = self._bind(image, prompt, enable_grad=False)
            answer_ids = answer_token_ids(
                self.processor.tokenizer, answer_text, leading_space=self.leading_space
            )
            full_ids = torch.cat([bound["input_ids"], answer_ids.view(1, -1).to(self.device)], dim=-1)
            full_mask = torch.cat(
                [bound["attention_mask"], torch.ones(1, answer_ids.numel(), dtype=torch.long, device=self.device)],
                dim=-1,
            )
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=full_mask,
                pixel_values=bound["pixel_values"],
                pixel_attention_mask=bound["pixel_attention_mask"],
                use_cache=False,
            )
            prompt_length = int(bound["input_ids"].shape[-1])
            logprob = sequence_logprob(outputs.logits, full_ids, prompt_length)
            labels = full_ids.clone()
            labels[:, :prompt_length] = -100
            vocab_size = _vocab_size(self.model)
            library_loss = self.model.loss_function(
                logits=outputs.logits, labels=labels, vocab_size=vocab_size
            )
        n_tokens = int(answer_ids.numel())
        mean_from_sum = float((-logprob / n_tokens).detach().cpu())
        library_mean = float(library_loss.detach().cpu())
        return {
            "logprob": float(logprob.detach().cpu()),
            "n_tokens": n_tokens,
            "token_ids": answer_ids.tolist(),
            "tokens": self.processor.tokenizer.convert_ids_to_tokens(answer_ids.tolist()),
            "mean_nll_from_sum": mean_from_sum,
            "library_mean_nll": library_mean,
            "mean_nll_abs_diff": abs(mean_from_sum - library_mean),
            "prompt_tokens": prompt_length,
        }

    def prompt_last_logits(
        self, image, prompt: str, prefix_token_ids: list[int] | None = None
    ) -> torch.Tensor:
        """Float32 CPU logits at the last position of the prompt plus ``prefix_token_ids``.

        With no prefix, these logits predict the first answer token. With a prefix,
        they predict the next answer token under teacher forcing. The forward uses
        the same float32 pixels as scoring and generation. No image gradient is
        taken. This is not an attack.
        """
        bound = self._bind(image, prompt, enable_grad=False)
        input_ids = bound["input_ids"]
        attention_mask = bound["attention_mask"]
        if prefix_token_ids:
            prefix = torch.tensor([prefix_token_ids], dtype=torch.long, device=self.device)
            input_ids = torch.cat([input_ids, prefix], dim=-1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(1, prefix.shape[-1], dtype=attention_mask.dtype, device=self.device),
                ],
                dim=-1,
            )
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=bound["pixel_values"],
                pixel_attention_mask=bound["pixel_attention_mask"],
                use_cache=False,
            )
        logits = outputs.logits[0, -1, :].detach().to(dtype=torch.float32, device="cpu").contiguous()
        if not torch.isfinite(logits).all():
            raise RuntimeError("Answer-position logits are not finite.")
        return logits

    def _logprob_tensor(self, image, prompt: str, answer_text: str, *, enable_grad: bool):
        bound = self._bind(image, prompt, enable_grad=enable_grad)
        answer_ids = answer_token_ids(
            self.processor.tokenizer, answer_text, leading_space=self.leading_space
        ).to(self.device)
        full_ids = torch.cat([bound["input_ids"], answer_ids.view(1, -1)], dim=-1)
        full_mask = torch.cat(
            [bound["attention_mask"], torch.ones(1, answer_ids.numel(), dtype=torch.long, device=self.device)],
            dim=-1,
        )
        outputs = self.model(
            input_ids=full_ids,
            attention_mask=full_mask,
            pixel_values=bound["pixel_values"],
            pixel_attention_mask=bound["pixel_attention_mask"],
            use_cache=False,
        )
        logprob = sequence_logprob(outputs.logits, full_ids, int(bound["input_ids"].shape[-1]))
        return logprob, int(answer_ids.numel())

    def _bind(self, image, prompt: str, *, enable_grad: bool) -> dict[str, Any]:
        if not isinstance(prompt, str) or prompt.strip() == "":
            raise ValueError("prompt must be a non-empty string")
        canonical = to_image_01(image)
        leaf = canonical.detach()
        self._leaf = leaf
        if enable_grad:
            leaf.requires_grad_(True)
        pil = image_01_to_pil(leaf)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt},
            ],
        }]
        batch = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"do_image_splitting": self.do_image_splitting},
        )
        pixels, mask = differentiable_pixel_values(
            self.processor.image_processor,
            leaf,
            do_image_splitting=self.do_image_splitting,
        )
        if tuple(pixels.shape) != tuple(batch["pixel_values"].shape):
            raise RuntimeError(
                "Float preprocess shape "
                f"{tuple(pixels.shape)} does not match the chat-template pixel shape "
                f"{tuple(batch['pixel_values'].shape)}. Refusing to mix those layouts."
            )
        if tuple(mask.shape) != tuple(batch["pixel_attention_mask"].shape):
            raise RuntimeError(
                "Float attention-mask shape does not match the chat-template mask shape."
            )
        return {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "pixel_values": pixels.to(self.device),
            "pixel_attention_mask": mask.to(self.device),
        }


_default_session: VLMSession | None = None


def configure_default_session(**kwargs) -> VLMSession:
    """Install the session used by the module-level functions."""
    global _default_session
    _default_session = VLMSession(**kwargs)
    return _default_session


def get_default_session() -> VLMSession:
    """Return the installed session, loading the pinned checkpoint on first use."""
    global _default_session
    if _default_session is None:
        _default_session = VLMSession()
    return _default_session


def answer_logprob(image, prompt: str, answer_text: str, *, session: VLMSession | None = None) -> float:
    """Joint log-probability of ``answer_text`` given ``image`` and ``prompt``."""
    return _session(session).answer_logprob(image, prompt, answer_text)


def image_loss_gradient(image, prompt: str, answer_text: str, *, session: VLMSession | None = None) -> torch.Tensor:
    """CPU gradient of ``L = -log P(answer_text | image, prompt)``."""
    return _session(session).image_loss_gradient(image, prompt, answer_text)


def generate(
    image, prompt: str, *, task_family: str, session: VLMSession | None = None
) -> dict[str, str | bool | None]:
    """Greedy generation parsed into ``text``, ``parsed_answer``, and ``invalid``."""
    return _session(session).generate(image, prompt, task_family=task_family)


def _session(session: VLMSession | None) -> VLMSession:
    return get_default_session() if session is None else session


def _end_token_ids(model, tokenizer) -> set[int]:
    found: set[int] = set()
    sources = [getattr(tokenizer, "eos_token_id", None)]
    generation = getattr(model, "generation_config", None)
    if generation is not None:
        sources.append(getattr(generation, "eos_token_id", None))
    for source in sources:
        if source is None:
            continue
        if isinstance(source, (list, tuple, set)):
            found.update(int(item) for item in source)
        else:
            found.add(int(source))
    if not found:
        raise RuntimeError("No end-of-sequence id is available, so truncation cannot be judged.")
    return found


def _continuation_ids(generated: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    if generated.ndim != 2 or generated.shape[0] != 1:
        raise RuntimeError(f"Expected generate() to return shape [1, time], got {tuple(generated.shape)}")
    prompt = input_ids[0].to(generated.device)
    if generated.shape[1] >= prompt.shape[0] and torch.equal(generated[0, : prompt.shape[0]], prompt):
        return generated[0, prompt.shape[0] :]
    return generated[0]


def _vocab_size(model) -> int:
    config = model.config
    text = getattr(config, "text_config", None)
    if text is not None and getattr(text, "vocab_size", None):
        return int(text.vocab_size)
    return int(config.vocab_size)
