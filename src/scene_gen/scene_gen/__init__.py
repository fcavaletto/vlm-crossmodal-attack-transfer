"""Development scene generator for Checkpoint 0.

The package draws original geometric scenes and writes a development manifest.
It does not load a vision-language model, run an attack, or emit a held-out split.
"""

from .generate import SealedSplitError, generate_development
from .version import __version__

__all__ = ["SealedSplitError", "__version__", "generate_development"]
