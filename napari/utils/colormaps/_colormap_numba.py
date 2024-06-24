from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange, typed

from napari.utils.colormaps._colormap_numpy import (
    MAPPING_OF_UNKNOWN_VALUE,
    minimum_dtype_for_labels,
)

if TYPE_CHECKING:
    from napari.utils import DirectLabelColormap


__all__ = (
    'zero_preserving_modulo_numba',
    'labels_raw_to_texture_direct_numba',
)


def zero_preserving_modulo_numba(
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
    result = np.empty_like(values, dtype=dtype)
    # need to preallocate numpy array for asv memory benchmarks
    return _zero_preserving_modulo_inner_loop(values, n, to_zero, out=result)


@njit(parallel=True, cache=True)
def _zero_preserving_modulo_inner_loop(
    values: np.ndarray, n: int, to_zero: int, out: np.ndarray
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
    to_zero : int
        A specific value to map to 0. (Usually, 0 itself.)
    out : np.ndarray
        Preallocated the output array

    Returns
    -------
    np.ndarray
        The result: 0 for the ``to_zero`` value, ``values % n + 1``
        everywhere else.
    """
    for i in prange(values.size):
        if values.flat[i] == to_zero:
            out.flat[i] = 0
        else:
            out.flat[i] = (values.flat[i] - 1) % n + 1

    return out


def labels_raw_to_texture_direct_numba(
    data: np.ndarray, direct_colormap: 'DirectLabelColormap'
) -> np.ndarray:
    """
    Cast direct labels to the minimum type.

    Parameters
    ----------
    data : np.ndarray
        The input data array.
    direct_colormap : DirectLabelColormap
        The direct colormap.

    Returns
    -------
    np.ndarray
        The cast data array.
    """
    if direct_colormap.use_selection:
        return (data == direct_colormap.selection).astype(np.uint8)

    dkt = direct_colormap._get_typed_dict_mapping(data.dtype)
    target_dtype = minimum_dtype_for_labels(
        direct_colormap._num_unique_colors + 2
    )
    result_array = np.full_like(
        data, MAPPING_OF_UNKNOWN_VALUE, dtype=target_dtype
    )
    return _labels_raw_to_texture_direct_inner_loop(data, dkt, result_array)


@njit(parallel=True, cache=True)
def _labels_raw_to_texture_direct_inner_loop(
    data: np.ndarray, dkt: 'typed.Dict', out: np.ndarray
) -> np.ndarray:
    """
    Relabel data using typed dict with mapping unknown labels to default value
    """
    # The numba typed dict does not provide official Api for
    # determine key and value types
    for i in prange(data.size):
        val = data.flat[i]
        if val in dkt:
            out.flat[i] = dkt[data.flat[i]]

    return out
