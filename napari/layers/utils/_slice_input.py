from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Generic, List, Tuple, TypeVar, Union

import numpy as np

from napari.utils.misc import reorder_after_dim_reduction
from napari.utils.transforms import Affine
from napari.utils.translations import trans

_T = TypeVar('_T')


@dataclass(frozen=True)
class _ThickNDSlice(Generic[_T]):
    """Holds the point and the left and right margins of a thick nD slice."""

    point: Tuple[_T, ...]
    margin_left: Tuple[_T, ...]
    margin_right: Tuple[_T, ...]

    @classmethod
    def make_full(cls, ndim: int):
        return cls(
            point=tuple(np.nan for _ in range(ndim)),
            margin_left=tuple(np.nan for _ in range(ndim)),
            margin_right=tuple(np.nan for _ in range(ndim)),
        )

    def __getitem__(self, key):
        # this allows to use numpy-like slicing on the whole object
        return _ThickNDSlice(
            point=tuple(np.array(self.point)[key]),
            margin_left=tuple(np.array(self.margin_left)[key]),
            margin_right=tuple(np.array(self.margin_right)[key]),
        )

    def as_array(self):
        return np.array([self.point, self.margin_left, self.margin_right])

    @classmethod
    def from_array(self, arr):
        return _ThickNDSlice(
            point=tuple(arr[0]),
            margin_left=tuple(arr[1]),
            margin_right=tuple(arr[2]),
        )

    def __iter__(self):
        yield from zip(self.point, self.margin_left, self.margin_right)


@dataclass(frozen=True)
class _SliceInput:
    """Encapsulates the input needed for slicing a layer.

    An instance of this should be associated with a layer and some of the values
    in ``Viewer.dims`` when slicing a layer.
    """

    # The number of dimensions to be displayed in the slice.
    ndisplay: int
    # The thick slice in world coordinates.
    # Only the elements in the non-displayed dimensions have meaningful values.
    world_slice: _ThickNDSlice[float]
    # The layer dimension indices in the order they are displayed.
    # A permutation of the ``range(self.ndim)``.
    # The last ``self.ndisplay`` dimensions are displayed in the canvas.
    order: Tuple[int, ...]

    @property
    def ndim(self) -> int:
        """The dimensionality of the associated layer."""
        return len(self.order)

    @property
    def displayed(self) -> List[int]:
        """The layer dimension indices displayed in this slice."""
        return list(self.order[-self.ndisplay :])

    @property
    def not_displayed(self) -> List[int]:
        """The layer dimension indices not displayed in this slice."""
        return list(self.order[: -self.ndisplay])

    def with_ndim(self, ndim: int) -> _SliceInput:
        """Returns a new instance with the given number of layer dimensions."""
        old_ndim = self.ndim
        if old_ndim > ndim:
            point = self.world_slice.point[-ndim:]
            margin_left = self.world_slice.margin_left[-ndim:]
            margin_right = self.world_slice.margin_right[-ndim:]
            order = reorder_after_dim_reduction(self.order[-ndim:])
        elif old_ndim < ndim:
            point = (0,) * (ndim - old_ndim) + self.world_slice.point
            margin_left = (0,) * (
                ndim - old_ndim
            ) + self.world_slice.margin_left
            margin_right = (0,) * (
                ndim - old_ndim
            ) + self.world_slice.margin_right
            order = tuple(range(ndim - old_ndim)) + tuple(
                o + ndim - old_ndim for o in self.order
            )
        else:
            point = self.world_slice.point
            margin_left = self.world_slice.margin_left
            margin_right = self.world_slice.margin_right
            order = self.order

        world_slice = _ThickNDSlice(
            point=point, margin_left=margin_left, margin_right=margin_right
        )

        return _SliceInput(
            ndisplay=self.ndisplay, world_slice=world_slice, order=order
        )

    def data_slice(
        self, world_to_data: Affine, round_index: bool = True
    ) -> _ThickNDSlice[Union[float, int]]:
        """Transforms this thick_slice into data coordinates with only relevant dimensions.

        The elements in non-displayed dimensions will be real numbers.
        The elements in displayed dimensions will be ``slice(None)``.
        """
        if not self.is_orthogonal(world_to_data):
            warnings.warn(
                trans._(
                    'Non-orthogonal slicing is being requested, but is not fully supported. Data is displayed without applying an out-of-slice rotation or shear component.',
                    deferred=True,
                ),
                category=UserWarning,
            )

        slice_world_to_data = world_to_data.set_slice(self.not_displayed)
        world_slice_not_disp = self.world_slice.as_array()

        data_slice = slice_world_to_data(world_slice_not_disp)

        if round_index:
            # A round is taken to convert these values to slicing integers
            data_slice = np.round(data_slice).astype(int)

        full_data_slice = np.full((3, self.ndim), np.nan)

        for i, ax in enumerate(self.not_displayed):
            full_data_slice[:, ax] = data_slice[:, i]

        return _ThickNDSlice.from_array(full_data_slice)

    def is_orthogonal(self, world_to_data: Affine) -> bool:
        """Returns True if this slice represents an orthogonal slice through a layer's data, False otherwise."""
        # Subspace spanned by non displayed dimensions
        non_displayed_subspace = np.zeros(self.ndim)
        for d in self.not_displayed:
            non_displayed_subspace[d] = 1
        # Map subspace through inverse transform, ignoring translation
        world_to_data = Affine(
            ndim=self.ndim,
            linear_matrix=world_to_data.linear_matrix,
            translate=None,
        )
        mapped_nd_subspace = world_to_data(non_displayed_subspace)
        # Look at displayed subspace
        displayed_mapped_subspace = (
            mapped_nd_subspace[d] for d in self.displayed
        )
        # Check that displayed subspace is null
        return all(abs(v) < 1e-8 for v in displayed_mapped_subspace)
