"""Scoring roles for Checkpoint 1.6.

The official primary metric is the Checkpoint 1 strict parser of greedy text.
The class-loss argmin is a stored closed-vocabulary readout. It is not the
primary outcome. The embedded single-label reading is unselected.
"""

from __future__ import annotations

OFFICIAL_METRIC = "strict_free_generation"
FREE_GENERATION_ROLE = "official"
CLASS_ARGMIN_IS_PRIMARY = False
CLASS_LOSS_ROLE = "differentiated_closed_vocabulary_loss"
CLASS_DECISION_ROLE = "stored_readout"
EMBEDDED_READING_ADOPTED = False
DISAGREEMENT_TOLERANCE = 1e-6
DESIGN_BAND = (0.60, 0.80)
PRIMARY_CONTEXTS = ("neutral", "helpful_nonrevealing")
CONTEXTS = (
    "neutral",
    "helpful_nonrevealing",
    "contradictory",
    "answer_revealing",
)
FAMILIES = ("attribute", "counting", "spatial")
N_DEVELOPMENT_ROWS = 288
SCENES_PER_FAMILY = 24
