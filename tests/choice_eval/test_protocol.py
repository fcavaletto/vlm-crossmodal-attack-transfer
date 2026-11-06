"""Protocol flags and the package boundary. No checkpoint is loaded."""

from __future__ import annotations

import json
from pathlib import Path

from choice_eval.colors import COLORS
from choice_eval.models import QWEN3_VL_2B, SMOLVLM2
from choice_eval.protocol import CORRECTS_CHECKPOINT_1_6, PROTOCOL_STATUS
from choice_eval.score import score_example
from vlm_iface.pins import MODEL_ID, PINNED_REVISION, QWEN_CANDIDATE_ID

REPO = Path(__file__).resolve().parents[2]


def test_protocol_is_prospective_and_does_not_correct_1_6(peaked):
    assert PROTOCOL_STATUS == "prospective"
    assert CORRECTS_CHECKPOINT_1_6 is False
    logits = {color: peaked(1, 1.0) for color in COLORS}
    token_ids = {color: [1] for color in COLORS}
    row = score_example(logits, token_ids, label="red")
    assert row["protocol_status"] == "prospective"
    assert row["corrects_checkpoint_1_6"] is False
    assert row["free_generation"] is None


def test_pins_match_the_recorded_checkpoints():
    assert SMOLVLM2.model_id == MODEL_ID
    assert SMOLVLM2.revision == PINNED_REVISION
    assert QWEN3_VL_2B.model_id == QWEN_CANDIDATE_ID
    recorded = json.loads((REPO / "docs" / "checkpoint-1.6" / "qwen_feasibility.json").read_text())
    assert QWEN3_VL_2B.revision == recorded["qwen_candidate"]["revision"]
    assert QWEN3_VL_2B.model_id == recorded["qwen_candidate"]["repo_id"]


def test_package_source_does_not_load_weights_or_step_an_image():
    root = REPO / "src" / "choice_eval" / "choice_eval"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "AutoModel" not in text
        assert "image_loss_gradient" not in text
        if path.name != "verify_live.py":
            assert "from_pretrained" not in text
            assert "huggingface_hub" not in text
            assert "AutoTokenizer" not in text
