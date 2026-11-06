"""Weight-memory guard. These tests do not contact the Hub."""

from __future__ import annotations

import json
import struct

from vlm_iface.candidate import decide_load, parse_safetensors_prefix
from vlm_iface.pins import FLOAT32_WEIGHT_ABORT_BYTES


def test_two_billion_bf16_parameters_are_rejected():
    decision = decide_load(parameter_count=2_000_000_000, float32_weight_bytes=2_000_000_000 * 4)
    assert decision["load_permitted_by_weight_guard"] is False
    assert decision["float32_weight_bytes"] > FLOAT32_WEIGHT_ABORT_BYTES


def test_five_hundred_million_float32_parameters_are_allowed():
    count = 507_497_656
    decision = decide_load(parameter_count=count, float32_weight_bytes=count * 4)
    assert decision["load_permitted_by_weight_guard"] is True


def test_safetensors_header_counts_bf16_parameters_without_tensor_bytes():
    header = {
        "__metadata__": {"format": "pt"},
        "weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
    }
    encoded = json.dumps(header).encode("utf-8")
    blob = struct.pack("<Q", len(encoded)) + encoded
    summary = parse_safetensors_prefix(blob)
    assert summary["parameter_count"] == 6
    assert summary["dtype_counts"] == {"BF16": 6}
    assert summary["float32_weight_bytes"] == 24
