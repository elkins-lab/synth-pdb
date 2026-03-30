"""
Shared Numba JIT support for geometry operations.
"""

try:
    from numba import njit
except ImportError:
    # Fallback to no-op decorator if numba is not installed
    def njit(func=None, **kwargs):  # type: ignore[no-untyped-def]
        if func is None:
            return lambda f: f
        return func
