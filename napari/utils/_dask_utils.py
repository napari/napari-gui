"""Dask cache utilities."""

import collections.abc
import contextlib
from collections.abc import Callable, Iterator
from typing import Any

import dask
import dask.array as da
from dask.cache import Cache

#: dask.cache.Cache, optional : A dask cache for opportunistic caching
#: use :func:`~.resize_dask_cache` to actually register and resize.
#: this is a global cache (all layers will use it), but individual layers
#: can opt out using Layer(..., cache=False)
_DASK_CACHE = Cache(1)
_DEFAULT_MEM_FRACTION = 0.25

DaskIndexer = Callable[
    [], contextlib.AbstractContextManager[tuple[dict, Cache] | None]
]


def resize_dask_cache(
    nbytes: int | None = None, mem_fraction: float | None = None
) -> Cache:
    """Create or resize the dask cache used for opportunistic caching.

    The cache object is an instance of a :class:`Cache`, (which
    wraps a :class:`cachey.Cache`).

    See `Dask opportunistic caching
    <https://docs.dask.org/en/latest/caching.html>`_

    Parameters
    ----------
    nbytes : int, optional
        The desired size of the cache, in bytes.  If ``None``, the cache size
        will autodetermined as fraction of the total memory in the system,
        using ``mem_fraction``.  If ``nbytes`` is 0. The cache is turned off.
        by default, cache size is autodetermined using ``mem_fraction``.
    mem_fraction : float, optional
        The fraction (from 0 to 1) of total memory to use for the dask cache.

    Returns
    -------
    dask_cache : dask.cache.Cache
        An instance of a Dask Cache

    Examples
    --------
    >>> from napari.utils import resize_dask_cache
    >>> cache = resize_dask_cache()  # use 25% of total memory by default

    >>> # dask.Cache wraps cachey.Cache
    >>> assert isinstance(cache.cache, cachey.Cache)

    >>> # useful attributes
    >>> cache.cache.available_bytes  # full size of cache
    >>> cache.cache.total_bytes   # currently used bytes
    """
    from psutil import virtual_memory

    if nbytes is None and mem_fraction is not None:
        nbytes = int(virtual_memory().total * mem_fraction)

    avail = _DASK_CACHE.cache.available_bytes
    # if we don't have a cache already, create one.
    if avail == 1:
        # If neither nbytes nor mem_fraction was provided, use default
        if nbytes is None:
            nbytes = int(virtual_memory().total * _DEFAULT_MEM_FRACTION)
        _DASK_CACHE.cache.resize(nbytes)
    elif nbytes is not None and nbytes != _DASK_CACHE.cache.available_bytes:
        # if the cache has already been registered, then calling
        # resize_dask_cache() without supplying either mem_fraction or nbytes
        # is a no-op:
        _DASK_CACHE.cache.resize(nbytes)
    return _DASK_CACHE


def _is_dask_data(data: Any) -> bool:
    """Return True if data is a dask array or a list/tuple of dask arrays."""
    return isinstance(data, da.Array) or (
        isinstance(data, collections.abc.Sequence)
        and any(isinstance(i, da.Array) for i in data)
    )


def configure_dask(data: Any, cache: bool = True) -> DaskIndexer:
    """Spin up cache and return context manager that optimizes Dask indexing.

    This function determines whether data is a dask array or list of dask
    arrays and prepares some optimizations if so.

    When a delayed dask array is given to napari, there are couple things that
    need to be done to optimize performance.

    1. Opportunistic caching needs to be enabled, such that we don't recompute
       (or "re-read") data that has already been computed or read.

    2. Dask task fusion must be turned off to prevent napari from triggering
       new io on data that has already been read from disk. For example, with a
       4D timelapse of 3D stacks, napari may actually *re-read* the entire 3D
       tiff file every time the Z plane index is changed. Turning of Dask task
       fusion with ``optimization.fuse.active == False`` prevents this.

       .. note::

          Turning off task fusion requires Dask version 2.15.0 or later.

    For background and context, see `napari/napari#718
    <https://github.com/napari/napari/issues/718>`_, `napari/napari#1124
    <https://github.com/napari/napari/pull/1124>`_, and `dask/dask#6084
    <https://github.com/dask/dask/pull/6084>`_.

    For details on Dask task fusion, see the documentation on `Dask
    Optimization <https://docs.dask.org/en/latest/optimize.html>`_.

    Parameters
    ----------
    data : Any
        data, as passed to a ``Layer.__init__`` method.

    Returns
    -------
    ContextManager
        A context manager that can be used to optimize dask indexing

    Examples
    --------
    >>> data = dask.array.ones((10,10,10))
    >>> optimized_slicing = configure_dask(data)
    >>> with optimized_slicing():
    ...    data[0, 2].compute()
    """
    if not _is_dask_data(data):
        return contextlib.nullcontext

    _cache = resize_dask_cache() if cache else contextlib.nullcontext()

    @contextlib.contextmanager
    def dask_optimized_slicing(
        memfrac: float = 0.5,
    ) -> Iterator[tuple[Any, Any]]:
        opts = {'optimization.fuse.active': False}
        with dask.config.set(opts) as cfg, _cache as c:
            yield cfg, c

    return dask_optimized_slicing
