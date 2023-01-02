import numpy as np


__all__ = (
    "_sliding_window",
    "_wrap_around_sliding_window",
)


def _cartesian_product(*arrays):
    """Generate a cartesian product of the given arrays."""
    return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))


def _sliding_window(array: np.ndarray, window_shape: tuple[int, int]) -> np.ndarray:
    """
    Generate a sliding views of 2D array with a given shape along a extra dimension.
    Window moves (left-to-right top-to-bottom) along the array.
    It does not wrap around the array.
    """
    # Generate an array of indices for the first window
    result = np.sum(
        _cartesian_product(
            np.arange(window_shape[1]) * array.shape[0],
            np.arange(window_shape[0]),
        ),
        axis=1,
    ).reshape(window_shape)
    # Manipulate the indices to generate the rest of the windows
    steps = (
        np.add(
            np.arange(array.shape[0] - window_shape[0] + 1)
            .reshape(-1, 1)
            .repeat(array.shape[1] - window_shape[1] + 1, axis=1),
            (np.arange(array.shape[1] - window_shape[1] + 1) * array.shape[0])
            .reshape(-1, 1)
            .repeat(array.shape[0] - window_shape[0] + 1, axis=1)
            .T,
        )
        .reshape(-1, 1, 1)
        .repeat(window_shape[0], axis=1)
        .repeat(window_shape[1], axis=2)
    )
    return array.take(result + steps)


def _wrap_around_sliding_window(
    array: np.ndarray, window_shape: tuple[int, int]
) -> np.ndarray:
    """
    Generate a sliding views of 2D array with a given shape along a extra dimension.
    Window moves (left-to-right top-to-bottom) along the array.
    It wraps around the array.
    """
    # Generate an array of indices for the first window
    result = np.sum(
        _cartesian_product(
            np.arange(window_shape[1]) * array.shape[0],
            np.arange(window_shape[0]),
        ),
        axis=1,
    ).reshape(window_shape) - (array.shape[0] + 1)
    # Manipulate the indices to generate the rest of the windows
    steps = (
        np.arange(array.size)
        .reshape(-1, 1, 1)
        .repeat(window_shape[0], axis=1)
        .repeat(window_shape[1], axis=2)
    )
    return array.take(result + steps, mode="wrap")
