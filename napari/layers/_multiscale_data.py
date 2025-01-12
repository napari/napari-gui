from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from napari.layers._data_protocols import LayerDataProtocol, assert_protocol
from napari.utils.translations import trans


# note: this also implements `LayerDataProtocol`, but we don't need to inherit.
class MultiScaleData(Sequence[LayerDataProtocol]):
    """Wrapper for multiscale data, to provide consistent API.

    :class:`LayerDataProtocol` is the subset of the python Array API that we
    expect array-likes to provide.  Multiscale data is just a sequence of
    array-likes (providing, e.g. `shape`, `dtype`, `__getitem__`).

    Parameters
    ----------
    data : Sequence[LayerDataProtocol]
        Levels of multiscale data, from larger to smaller.
    max_size : Sequence[int], optional
        Maximum size of a displayed tile in pixels, by default`data[-1].shape`

    Raises
    ------
    ValueError
        If `data` is empty or is not a list, tuple, or ndarray.
    TypeError
        If any of the items in `data` don't provide `LayerDataProtocol`.
    """

    def __init__(
        self,
        data: Sequence[LayerDataProtocol],
    ) -> None:
        self._data: list[LayerDataProtocol] = list(data)
        if not self._data:
            raise ValueError(
                trans._('Multiscale data must be a (non-empty) sequence')
            )
        for d in self._data:
            assert_protocol(d)

    @property
    def size(self) -> int:
        """Return size of the first scale.."""
        return self._data[0].size

    @property
    def ndim(self) -> int:
        """Return ndim of the first scale.."""
        return self._data[0].ndim

    @property
    def dtype(self) -> np.dtype:
        """Return dtype of the first scale.."""
        return self._data[0].dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of multiscale is just the biggest shape."""
        return self._data[0].shape

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        """Tuple shapes for all scales."""
        return tuple(im.shape for im in self._data)

    def __getitem__(  # type: ignore [override]
        self, key: int | tuple[slice, ...]
    ) -> LayerDataProtocol:
        """Multiscale indexing."""
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> bool:
        return self._data == other

    def __add__(self, other) -> bool:
        return self._data + other

    def __mul__(self, other) -> bool:
        return self._data * other

    def __rmul__(self, other) -> bool:
        return other * self._data

    def __array__(self) -> np.ndarray:
        return np.asarray(self._data[-1])

    def __repr__(self) -> str:
        return (
            f'<MultiScaleData at {hex(id(self))}. '
            f"{len(self)} levels, '{self.dtype}', shapes: {self.shapes}>"
        )
