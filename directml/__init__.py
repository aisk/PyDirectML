"""Python binding for DirectML.

The package splits in two along the line drawn in docs/api-design.md: ``_core``
is the compiled extension and owns resources, execution and the data hot path;
this wrapper layer owns signature shaping, defaults, validation and error
messages. The boundary of the library is ``import directml``, not the ``.pyd``.
"""

import importlib.metadata

from ._core import *  # noqa: F401,F403 -- the classes, enums and operators

try:
    __version__ = importlib.metadata.version("directml")
except importlib.metadata.PackageNotFoundError:
    # Running from a source tree that was never pip-installed.
    __version__ = "0.0.0"
