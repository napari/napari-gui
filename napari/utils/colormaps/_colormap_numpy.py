from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from napari.utils import DirectLabelColormap


MAPPING_OF_UNKNOWN_VALUE = 0
# For direct mode, we map all unknown values to single value
# for simplicity of implementation we select 0

__all__ = (
    'zero_preserving_modulo_numpy',
    'labels_raw_to_texture_direct_numpy',
    'minimum_dtype_for_labels',
)


def zero_preserving_modulo_numpy(
    values: np.ndarray, n: int, dtype: np.dtype, to_zero: int = 0
) -> np.ndarray:
    """``(values - 1) % n + 1``, but with one specific value mapped to 0.

    This ensures (1) an output value in [0, n] (inclusive), and (2) that
    no nonzero values in the input are zero in the output, other than the
    ``to_zero`` value.

    Parameters
    ----------
    values : np.ndarray
        The dividend of the modulo operator.
    n : int
        The divisor.
    dtype : np.dtype
        The desired dtype for the output array.
    to_zero : int, optional
        A specific value to map to 0. (By default, 0 itself.)

    Returns
    -------
    np.ndarray
        The result: 0 for the ``to_zero`` value, ``values % n + 1``
        everywhere else.
    """
    res = ((values - 1) % n + 1).astype(dtype)
    res[values == to_zero] = 0
    return res


def labels_raw_to_texture_direct_numpy(
    data: np.ndarray, direct_colormap: 'DirectLabelColormap'
) -> np.ndarray:
    """Convert labels data to the data type used in the texture.

    This implementation uses numpy vectorized operations.

    See `_cast_labels_data_to_texture_dtype_direct` for more details.
    """
    if direct_colormap.use_selection:
        return (data == direct_colormap.selection).astype(np.uint8)
    mapper = direct_colormap._array_map
    if any(x < 0 for x in direct_colormap.color_dict if x is not None):
        half_shape = mapper.shape[0] // 2 - 1
        data = np.clip(data, -half_shape, half_shape)
    else:
        data = np.clip(data, 0, mapper.shape[0] - 1)

    return mapper[data]


def minimum_dtype_for_labels(num_colors: int) -> np.dtype:
    """Return the minimum texture dtype that can hold given number of colors.

    Parameters
    ----------
    num_colors : int
        Number of unique colors in the data.

    Returns
    -------
    np.dtype
        Minimum dtype that can hold the number of colors.
    """
    if num_colors <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if num_colors <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.float32)
