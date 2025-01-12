from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from napari.utils.misc import reorder_after_dim_reduction
from napari.utils.transforms import Affine
from napari.utils.translations import trans

if TYPE_CHECKING:
    import numpy.typing as npt

    from napari.components.dims import Dims

_T = TypeVar('_T')


@dataclass(frozen=True)
class _ThickNDSlice(Generic[_T]):
    """Holds the point and the left and right margins of a thick nD slice."""

    point: tuple[_T, ...]
    margin_left: tuple[_T, ...]
    margin_right: tuple[_T, ...]

    @property
    def ndim(self):
        return len(self.point)

    @classmethod
    def make_full(
        cls,
        point=None,
        margin_left=None,
        margin_right=None,
        ndim=None,
    ):
        """
        Make a full slice based on minimal input.

        If ndim is provided, it will be used to crop or prepend zeros to the given values.
        Values not provided will be filled zeros.
        """
        for val in (point, margin_left, margin_right):
            if val is not None:
                val_ndim = len(val)
                break
        else:
            if ndim is None:
                raise ValueError(
                    'ndim must be provided if no other value is given'
                )
            val_ndim = ndim

        ndim = val_ndim if ndim is None else ndim

        # not provided arguments are just all zeros
        point = (0,) * ndim if point is None else tuple(point)
        margin_left = (
            (0,) * ndim if margin_left is None else tuple(margin_left)
        )
        margin_right = (
            (0,) * ndim if margin_right is None else tuple(margin_right)
        )

        # prepend zeros if ndim is bigger than the given values
        prepend = max(ndim - val_ndim, 0)

        point = (0,) * prepend + point
        margin_left = (0,) * prepend + margin_left
        margin_right = (0,) * prepend + margin_right

        # crop to ndim in case given values are longer (keeping last dims)
        return cls(
            point=point[-ndim:],
            margin_left=margin_left[-ndim:],
            margin_right=margin_right[-ndim:],
        )

    @classmethod
    def from_dims(cls, dims: Dims):
        """Generate from a Dims object's point and margins."""
        return cls.make_full(dims.point, dims.margin_left, dims.margin_right)

    def copy_with(
        self,
        point=None,
        margin_left=None,
        margin_right=None,
        ndim=None,
    ):
        """Create a copy, but modifying the given fields."""
        return self.make_full(
            point=point or self.point,
            margin_left=margin_left or self.margin_left,
            margin_right=margin_right or self.margin_right,
            ndim=ndim or self.ndim,
        )

    def as_array(self) -> npt.NDArray:
        """Return point and left and right margin as a (3, D) array."""
        return np.array([self.point, self.margin_left, self.margin_right])

    @classmethod
    def from_array(cls, arr: npt.NDArray) -> _ThickNDSlice:
        """Construct from a (3, D) array of point, left margin and right margin."""
        return cls(
            point=tuple(arr[0]),
            margin_left=tuple(arr[1]),
            margin_right=tuple(arr[2]),
        )

    def __getitem__(self, key):
        # this allows to use numpy-like slicing on the whole object
        return _ThickNDSlice(
            point=tuple(np.array(self.point)[key]),
            margin_left=tuple(np.array(self.margin_left)[key]),
            margin_right=tuple(np.array(self.margin_right)[key]),
        )

    def __iter__(self):
        # iterate all three fields dimension per dimension
        yield from zip(
            self.point, self.margin_left, self.margin_right, strict=False
        )


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
    order: tuple[int, ...]

    @property
    def ndim(self) -> int:
        """The dimensionality of the associated layer."""
        return len(self.order)

    @property
    def displayed(self) -> list[int]:
        """The layer dimension indices displayed in this slice."""
        return list(self.order[-self.ndisplay :])

    @property
    def not_displayed(self) -> list[int]:
        """The layer dimension indices not displayed in this slice."""
        return list(self.order[: -self.ndisplay])

    def with_ndim(self, ndim: int) -> _SliceInput:
        """Returns a new instance with the given number of layer dimensions."""
        old_ndim = self.ndim
        world_slice = self.world_slice.copy_with(ndim=ndim)
        if old_ndim > ndim:
            order = reorder_after_dim_reduction(self.order[-ndim:])
        elif old_ndim < ndim:
            order = tuple(range(ndim - old_ndim)) + tuple(
                o + ndim - old_ndim for o in self.order
            )
        else:
            order = self.order

        return _SliceInput(
            ndisplay=self.ndisplay, world_slice=world_slice, order=order
        )

    def data_slice(
        self,
        world_to_data: Affine,
    ) -> _ThickNDSlice[float | int]:
        """Transforms this thick_slice into data coordinates with only relevant dimensions.

        The elements in non-displayed dimensions will be real numbers.
        The elements in displayed dimensions will be ``slice(None)``.
        """
        if not self.is_orthogonal(world_to_data):
            warnings.warn(
                trans._(
                    'Non-orthogonal slicing is being requested, but is not fully supported. '
                    'Data is displayed without applying an out-of-slice rotation or shear component.',
                    deferred=True,
                ),
                category=UserWarning,
            )

        slice_world_to_data = world_to_data.set_slice(self.not_displayed)
        world_slice_not_disp = self.world_slice[self.not_displayed].as_array()

        data_slice = slice_world_to_data(world_slice_not_disp)

        full_data_slice = np.full((3, self.ndim), np.nan)

        for i, ax in enumerate(self.not_displayed):
            # we cannot have nan in non-displayed dims, so we default to 0
            full_data_slice[:, ax] = np.nan_to_num(data_slice[:, i], nan=0)

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
