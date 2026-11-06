"""Development scene generator for Checkpoint 1.7.

The package draws original geometric scenes for one size-shape-row color task
and writes a development manifest. It does not load a vision-language model,
run an attack, or emit a held-out split. It does not overwrite
``fixtures/development``, ``fixtures/development_v2``, or
``fixtures/development_simple``.
"""

from .generate import SealedSplitError, generate_development
from .version import __version__

__all__ = ["SealedSplitError", "__version__", "generate_development"]
