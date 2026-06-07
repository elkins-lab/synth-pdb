"""
Shared Numba JIT support for geometry operations.
"""

import os

try:
    if os.environ.get("NUMBA_DISABLE_JIT") == "1":
        raise ImportError("JIT disabled by environment variable")
    from numba import njit
except ImportError:  # pragma: no cover
    # Fallback to no-op decorator if numba is not installed or disabled
    def njit(func=None, **kwargs):  # type: ignore[no-untyped-def]
        if func is None:
            return lambda f: f
        return func
