from __future__ import annotations

import numpy as np

__all__ = ("gaussian_noise",)


def gaussian_noise(
    size: tuple[int, int], sigma: float, center: int = 128
) -> np.ndarray:
    """Generate an array of gaussian noise with a given size and center and standard deviation."""
    return np.random.normal(center, sigma, size[0] * size[1]).reshape(size)
