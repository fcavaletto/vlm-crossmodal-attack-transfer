"""Differentiable RGB path into the pinned Qwen3-VL-2B processor.

Importing this package does not load the model.
"""

from .pins import MODEL_ID, PINNED_REVISION, SCENE_ID

__all__ = ["MODEL_ID", "PINNED_REVISION", "SCENE_ID"]
