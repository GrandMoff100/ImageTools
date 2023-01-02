import numpy as np

from .misc import _sliding_window, _wrap_around_sliding_window


__all__ = ("apply_kernel",)


def apply_kernel(
    array: np.ndarray,
    kernel: np.ndarray,
    wrap_around: bool = False,
) -> np.ndarray:
    """
    Convolve a 2D array with a 2D kernel.
    The kernel is moved (left-to-right top-to-bottom) along the array.
    """
    if wrap_around:
        return np.sum(
            _wrap_around_sliding_window(array, kernel.shape) * kernel, axis=(1, 2)
        )
    return np.sum(_sliding_window(array, kernel.shape) * kernel, axis=(1, 2))
