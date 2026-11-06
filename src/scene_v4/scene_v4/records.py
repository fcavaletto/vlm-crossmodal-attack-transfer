"""Turn a design spec into one manifest record."""

from __future__ import annotations

import hashlib
import json

from .checks import ambiguity_flags
from .contexts import HELPFUL_TEXT, NEUTRAL_TEXT, build_contexts
from .design import SceneSpec
from .geometry import center
from .normalize import leakage_aliases, normalize_for_leakage
from .oracle import answer_for
from .seeds import development_seed
from .vocabulary import COLORS


def materialize(spec: SceneSpec) -> dict[str, object]:
    """Build one record. Raises if the scene is ambiguous or the text leaks."""
    graph_answer = answer_for(spec.task_family, spec.question, spec.objects)
    if graph_answer != spec.intended_answer:
        raise ValueError(
            f"{spec.scene_id}: scene graph answers {graph_answer!r}, "
            f"design says {spec.intended_answer!r}"
        )
    aliases = [spec.intended_answer]
    contexts = build_contexts(
        question=spec.question,
        answer=spec.intended_answer,
        foil=spec.foil,
    )
    for key in ("neutral", "helpful_nonrevealing", "contradictory"):
        leaked = leakage_aliases(str(contexts[key]["text"]), aliases)
        if leaked:
            raise ValueError(f"{spec.scene_id}: {key} contains {leaked}")
    if leakage_aliases(spec.question, aliases):
        raise ValueError(f"{spec.scene_id}: question contains the answer")
    joined = str(contexts["helpful_nonrevealing"]["text"]) + "\n" + spec.question
    if leakage_aliases(joined, aliases):
        raise ValueError(f"{spec.scene_id}: helpful prompt join contains the answer")
    revealing = str(contexts["answer_revealing"]["text"])
    if leakage_aliases(revealing, aliases) != aliases:
        raise ValueError(f"{spec.scene_id}: revealing text does not contain the answer")
    for text in (NEUTRAL_TEXT, HELPFUL_TEXT):
        norm = normalize_for_leakage(text)
        for color in COLORS:
            if color in norm:
                raise ValueError(f"{spec.scene_id}: non-revealing text contains {color}")

    objects = []
    for obj in spec.objects:
        cx, cy = center(obj.col, obj.row)
        objects.append(
            {
                "object_id": obj.object_id,
                "category": obj.shape,
                "attributes": {"color": obj.color, "size": obj.size},
                "position": {"col": obj.col, "row": obj.row, "cx": cx, "cy": cy},
            }
        )
    record: dict[str, object] = {
        "scene_id": spec.scene_id,
        "seed": development_seed(spec.scene_id),
        "split": "development",
        "task_family": spec.task_family,
        "counterfactual_group_id": spec.group_id,
        "question": spec.question,
        "answer": spec.intended_answer,
        "answer_aliases": aliases,
        "contexts": contexts,
        "objects": objects,
        "image_path": f"images/{spec.scene_id}.png",
        "ambiguity_flags": [],
    }
    flags = ambiguity_flags(record)
    if flags:
        raise ValueError(f"{spec.scene_id}: ambiguity flags {flags}")
    return record


def scene_graph_payload(record: dict[str, object]) -> dict[str, object]:
    """Canonical graph used for the development-exclusion list."""
    return {
        "question": record["question"],
        "answer": record["answer"],
        "objects": record["objects"],
    }


def scene_graph_hash(record: dict[str, object]) -> str:
    payload = json.dumps(scene_graph_payload(record), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
