"""Development scene generator for Checkpoint 1.6.

The package draws original geometric scenes for one row-and-shape color task
and writes a development manifest. It does not load a vision-language model,
run an attack, or emit a held-out split. It does not overwrite
``fixtures/development`` or ``fixtures/development_v2``.
"""

from .generate import SealedSplitError, generate_development
from .version import __version__

__all__ = ["SealedSplitError", "__version__", "generate_development"]
