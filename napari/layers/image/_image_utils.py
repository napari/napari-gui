"""guess_rgb, guess_multiscale, guess_labels."""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from napari.layers._data_protocols import LayerDataProtocol
from napari.layers._multiscale_data import MultiScaleData
from napari.layers.image._image_constants import ImageProjectionMode
from napari.utils.translations import trans


def guess_rgb(shape: tuple[int, ...], min_side_len: int = 30) -> bool:
    """Guess if the passed shape comes from rgb data.

    If last dim is 3 or 4 and other dims are larger (>30), assume the data is
    rgb, including rgba.

    Parameters
    ----------
    shape : list of int
        Shape of the data that should be checked.

    Returns
    -------
    bool
        If data is rgb or not.
    """
    ndim = len(shape)
    last_dim = shape[-1]
    viewed_dims = shape[-3:-1]

    return (
        ndim > 2
        and last_dim in (3, 4)
        and all(d > min_side_len for d in viewed_dims)
    )


def guess_multiscale(
    data: MultiScaleData | list | tuple,
) -> tuple[bool, LayerDataProtocol | Sequence[LayerDataProtocol]]:
    """Guess whether the passed data is multiscale, process it accordingly.

    If shape of arrays along first axis is strictly decreasing, the data is
    multiscale. If it is the same shape everywhere, it is not. Various
    ambiguous conditions in between will result in a ValueError being raised,
    or in an "unwrapping" of data, if data contains only one element.

    Parameters
    ----------
    data : array or list of array
        Data that should be checked.

    Returns
    -------
    multiscale : bool
        True if the data is thought to be multiscale, False otherwise.
    data : list or array
        The input data, perhaps with the leading axis removed.
    """
    # If the data has ndim and is not one-dimensional then cannot be multiscale
    # If data is a zarr array, this check ensure that subsets of it are not
    # instantiated. (`for d in data` instantiates `d` as a NumPy array if
    # `data` is a zarr array.)
    if isinstance(data, MultiScaleData):
        return True, data

    if hasattr(data, 'ndim') and data.ndim > 1:
        return False, data

    if isinstance(data, (list, tuple)) and len(data) == 1:
        # pyramid with only one level, unwrap
        return False, data[0]

    sizes = [d.size for d in data]
    if len(sizes) <= 1:
        return False, data

    consistent = all(
        s1 > s2 for s1, s2 in zip(sizes[:-1], sizes[1:], strict=False)
    )
    if all(s == sizes[0] for s in sizes):
        # note: the individual array case should be caught by the first
        # code line in this function, hasattr(ndim) and ndim > 1.
        raise ValueError(
            trans._(
                'Input data should be an array-like object, or a sequence of arrays of decreasing size. Got arrays of single size: {size}',
                deferred=True,
                size=sizes[0],
            )
        )
    if not consistent:
        raise ValueError(
            trans._(
                'Input data should be an array-like object, or a sequence of arrays of decreasing size. Got arrays in incorrect order, sizes: {sizes}',
                deferred=True,
                sizes=sizes,
            )
        )

    return True, MultiScaleData(data)


def guess_labels(data: Any) -> Literal['labels', 'image']:
    """Guess if array contains labels data."""

    if hasattr(data, 'dtype') and data.dtype in (
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ):
        return 'labels'

    return 'image'


def project_slice(
    data: npt.NDArray, axis: tuple[int, ...], mode: ImageProjectionMode
) -> npt.NDArray:
    """Project a thick slice along axis based on mode."""
    func: Callable
    if mode == ImageProjectionMode.SUM:
        func = np.sum
    elif mode == ImageProjectionMode.MEAN:
        func = np.mean
    elif mode == ImageProjectionMode.MAX:
        func = np.max
    elif mode == ImageProjectionMode.MIN:
        func = np.min
    else:
        raise NotImplementedError(f'unimplemented projection: {mode}')
    return func(data, tuple(axis))
