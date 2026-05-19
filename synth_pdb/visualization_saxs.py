"""Visualization for SAXS Profiles (Compatibility Shim)."""

import logging
from typing import Any

try:
    import synth_saxs as _saxs

    HAS_SYNTH_SAXS = True
except ImportError:
    HAS_SYNTH_SAXS = False

logger = logging.getLogger(__name__)

if HAS_SYNTH_SAXS:
    from synth_saxs import plot_saxs_results
else:

    def plot_saxs_results(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        raise ImportError("plot_saxs_results requires synth-saxs. Run: pip install synth-saxs")
