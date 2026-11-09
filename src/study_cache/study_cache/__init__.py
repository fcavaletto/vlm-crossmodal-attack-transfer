"""Opt-in model-cache path for reproducing model-dependent runs on another machine.

Why a launcher and not an edit: every recorded run refused any Hub cache except the
study path (``REQUIRED_HF_HOME`` in ``qwen_rgb/pins.py`` and ``dev_pilot/lock.py``).
Those packages are listed as preserved by the Checkpoint 2.1 attack protocol
(``development_attack.protocol.PRESERVED_PATHS``) and ``dev_pilot/lock.py`` is hashed in
the Checkpoint 2.0 manifest, so their files stay byte-identical. This package changes
nothing unless it is asked to:

    VLM_STUDY_MODEL_CACHE=/abs/path/to/hf-cache python -m study_cache path_probe confirm

Without ``VLM_STUDY_MODEL_CACHE`` the launcher runs the module exactly as
``python -m <module>`` would, guard included. With it, the launcher points the cache
guard at that directory for this process only. The model id and revision stay pinned in
code, so the snapshot read is still
``<cache>/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/<pinned revision>/``.
Setting ``HF_HOME`` alone never selects another cache.
"""

from .resolve import GUARDED_MODULES, OVERRIDE_ENV, apply_override, resolve_override

__all__ = ["GUARDED_MODULES", "OVERRIDE_ENV", "apply_override", "resolve_override"]
