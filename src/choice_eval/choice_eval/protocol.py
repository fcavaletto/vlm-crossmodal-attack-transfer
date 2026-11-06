"""Prospective Checkpoint 1.7 protocol.

This is not a correction of Checkpoint 1.6. The 1.6 primary outcome remains
the strict free-generation parser. The score defined here is the one a later
image attack would differentiate: the summed negative log-probability of one
full color sequence. Free generation stays a separate diagnostic.
"""

from __future__ import annotations

PROTOCOL_STATUS = "prospective"
PROTOCOL_ID = "checkpoint-1.7-choice-scoring"
CORRECTS_CHECKPOINT_1_6 = False
PRIMARY_METRIC = "teacher_forced_summed_nll"
PRIMARY_REDUCTION = "sum"
MEAN_REDUCTION_ROLE = "diagnostic_only"
FREE_GENERATION_ROLE = "secondary_diagnostic"
FREE_GENERATION_IS_PRIMARY = False
ATTACK_LOSS = "summed_nll_of_labeled_color"
TIE_BREAK = "lowest_summed_nll_then_color_name"
# parse_generation has no row_color family. Attribute and spatial already
# share the four color names, so the unchanged parser is called as attribute.
COLOR_PARSER_TASK_FAMILY = "attribute"
