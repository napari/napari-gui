import itertools
import os
from collections.abc import Callable, Sequence
from functools import lru_cache
from types import ModuleType
from typing import (
    Literal,
    overload,
)

import numpy as np
from skimage import morphology


def always_false(*_):
    return False


class Skip:
    def __init__(
        self,
        if_in_pr: Callable[..., bool] = always_false,
        if_on_ci: Callable[..., bool] = always_false,
        always: Callable[..., bool] = always_false,
    ):
        self.func_pr = if_in_pr if 'PR' in os.environ else always_false
        self.func_ci = if_on_ci if 'CI' in os.environ else always_false
        self.func_always = always

    def __contains__(self, item):
        return (
            self.func_pr(*item)
            or self.func_ci(*item)
            or self.func_always(*item)
        )


def _generate_ball(radius: int, ndim: int) -> np.ndarray:
    """Generate a ball of given radius and dimension.

    Parameters
    ----------
    radius : int
        Radius of the ball.
    ndim : int
        Dimension of the ball.

    Returns
    -------
    ball : ndarray of uint8
        Binary array of the hyper ball.
    """

    if ndim == 2:
        return morphology.disk(radius)
    if ndim == 3:
        return morphology.ball(radius)
    shape = (2 * radius + 1,) * ndim
    radius_sq = radius**2
    coords = np.indices(shape) - radius
    return (np.sum(coords**2, axis=0) <= radius_sq).astype(np.uint8)


def _generate_density(radius: int, ndim: int) -> np.ndarray:
    """Generate gaussian density of given radius and dimension."""
    shape = (2 * radius + 1,) * ndim
    coords = np.indices(shape) - radius
    dist = np.sqrt(np.sum(coords**2 / ((radius / 4) ** 2), axis=0))
    res = np.exp(-dist)
    res[res < 0.02] = 0
    return res


def _structure_at_coordinates(
    shape: tuple[int],
    coordinates: np.ndarray,
    structure: np.ndarray,
    *,
    multipliers: Sequence = itertools.repeat(1),
    dtype=None,
    reduce_fn: Callable[
        [np.ndarray, np.ndarray, np.ndarray | None], np.ndarray
    ],
):
    """Update data with structure at given coordinates.

    Parameters
    ----------
    data : ndarray
        Array to update.
    coordinates : ndarray
        Coordinates of the points. The structures will be added at these
        points (center).
    structure : ndarray
        Array with encoded structure. For example, ball (boolean) or density
        (0,1) float.
    multipliers : ndarray
        These values are multiplied by the values in the structure before
        updating the array. Can be used to generate different labels, or to
        vary the intensity of floating point gaussian densities.
    reduce_fn : function
        Function with which to update the array at a particular position. It
        should take two arrays as input and an optional output array.
    """
    radius = (structure.shape[0] - 1) // 2
    data = np.zeros(shape, dtype=dtype)

    for point, value in zip(coordinates, multipliers, strict=False):
        slice_im, slice_ball = _get_slices_at(shape, point, radius)
        reduce_fn(
            data[slice_im], value * structure[slice_ball], out=data[slice_im]
        )
    return data


def _get_slices_at(shape, point, radius):
    slice_im = []
    slice_ball = []
    for i, p in enumerate(point):
        slice_im.append(
            slice(max(0, p - radius), min(shape[i], p + radius + 1))
        )
        ball_start = max(0, radius - p)
        ball_stop = slice_im[-1].stop - slice_im[-1].start + ball_start
        slice_ball.append(slice(ball_start, ball_stop))
    return tuple(slice_im), tuple(slice_ball)


def _update_data_with_mask(data, struct, out=None):
    """Update ``data`` with ``struct`` where ``struct`` is nonzero."""
    # these branches are needed because np.where does not support
    # an out= keyword argument
    if out is None:
        return np.where(struct, struct, data)
    else:  # noqa: RET505
        nz = struct != 0
        out[nz] = struct[nz]
        return out


def _smallest_dtype(n: int) -> np.dtype:
    """Find the smallest dtype that can hold n values."""
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if np.iinfo(dtype).max >= n:
            return dtype
            break
    else:
        raise ValueError(f'{n=} is too large for any dtype.')


@overload
def labeled_particles(
    shape: Sequence[int],
    dtype: np.dtype | None = None,
    n: int = 144,
    seed: int | None = None,
    return_density: Literal[False] = False,
) -> np.ndarray: ...


@overload
def labeled_particles(
    shape: Sequence[int],
    dtype: np.dtype | None = None,
    n: int = 144,
    seed: int | None = None,
    return_density: Literal[True] = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@lru_cache
def labeled_particles(
    shape: Sequence[int],
    dtype: np.dtype | None = None,
    n: int = 144,
    seed: int | None = None,
    return_density: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate labeled blobs of given shape and dtype.

    Parameters
    ----------
    shape : Sequence[int]
        Shape of the resulting array.
    dtype : Optional[np.dtype]
        Dtype of the resulting array.
    n : int
        Number of blobs to generate.
    seed : Optional[int]
        Seed for the random number generator.
    return_density : bool
        Whether to return the density array and center coordinates.
    """
    if dtype is None:
        dtype = _smallest_dtype(n)
    rng = np.random.default_rng(seed)
    ndim = len(shape)
    points = rng.integers(shape, size=(n, ndim))
    values = rng.integers(
        np.iinfo(dtype).min, np.iinfo(dtype).max, size=n, dtype=dtype
    )
    sigma = int(max(shape) / (4.0 * n ** (1 / ndim)))
    ball = _generate_ball(sigma, ndim)

    labels = _structure_at_coordinates(
        shape,
        points,
        ball,
        multipliers=values,
        reduce_fn=_update_data_with_mask,
        dtype=dtype,
    )

    if return_density:
        dens = _generate_density(sigma * 2, ndim)
        densities = _structure_at_coordinates(
            shape, points, dens, reduce_fn=np.maximum, dtype=np.float32
        )

        return labels, densities, points
    else:  # noqa: RET505
        return labels


def run_benchmark_from_module(
    module: ModuleType, klass_name: str, method_name: str
):
    klass = getattr(module, klass_name)
    if getattr(klass, 'params', None):
        skip_if = getattr(klass, 'skip_params', {})
        if isinstance(klass.params[0], Sequence):
            params = itertools.product(*klass.params)
        else:
            params = ((i,) for i in klass.params)
        for param in params:
            if param in skip_if:
                continue
            obj = klass()
            try:
                obj.setup(*param)
            except NotImplementedError:
                continue
            getattr(obj, method_name)(*param)
            getattr(obj, 'teardown', lambda: None)()
    else:
        obj = klass()
        try:
            obj.setup()
        except NotImplementedError:
            return
        getattr(obj, method_name)()
        getattr(obj, 'teardown', lambda: None)()


def run_benchmark():
    import argparse
    import inspect

    parser = argparse.ArgumentParser(description='Run benchmark')
    parser.add_argument(
        'benchmark', type=str, help='Name of the benchmark to run', default=''
    )

    args = parser.parse_args()

    benchmark_selection = args.benchmark.split('.')

    # get module of parent frame
    call_module = inspect.getmodule(inspect.currentframe().f_back)
    run_benchmark_from_module(call_module, *benchmark_selection)
