from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from napari.layers.base._slice import _next_request_id
from napari.layers.image._image_constants import ImageProjectionMode
from napari.layers.image._image_utils import project_slice
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice
from napari.types import ArrayLike
from napari.utils._dask_utils import DaskIndexer
from napari.utils.misc import reorder_after_dim_reduction
from napari.utils.transforms import Affine


@dataclass(frozen=True)
class _ImageView:
    """A raw image and a potentially different viewable version of it.

    This is only needed for labels, and not other image layers, because sliced labels
    data are passed to vispy as floats in [0, 1] to use continuous colormaps.
    In that case, a conversion function is defined by `Labels._raw_to_displayed` to
    handle the desired colormapping behavior.

    For non-labels image layers the raw and viewable images should be the same instance
    and no conversion should be necessary.

    This is defined for images in general because `Labels` and `_ImageBase` share
    code through inheritance.

    Attributes
    ----------
    raw : array
        The raw image.
    view : array
        The viewable image, which should either be the same instance of raw, or a
        converted version of it.
    """

    raw: np.ndarray
    view: np.ndarray

    @classmethod
    def from_view(cls, view: np.ndarray) -> '_ImageView':
        """Makes an image view from the view where no conversion is needed."""
        return cls(raw=view, view=view)

    @classmethod
    def from_raw(
        cls, *, raw: np.ndarray, converter: Callable[[np.ndarray], np.ndarray]
    ) -> '_ImageView':
        """Makes an image view from the raw image and a conversion function."""
        view = converter(raw)
        return cls(raw=raw, view=view)


@dataclass(frozen=True)
class _ImageSliceResponse:
    """Contains all the output data of slicing an image layer.

    Attributes
    ----------
    image : _ImageView
        The sliced image data.
    thumbnail: _ImageView
        The thumbnail image data. This may come from a different resolution to the sliced image
        data for multi-scale images. Otherwise, it's the same instance as data.
    tile_to_data: Affine
        The affine transform from the sliced data to the full data at the highest resolution.
        For single-scale images, this will be the identity matrix.
    slice_input : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    request_id : int
        The identifier of the request from which this was generated.
    """

    image: _ImageView = field(repr=False)
    thumbnail: _ImageView = field(repr=False)
    tile_to_data: Affine = field(repr=False)
    slice_input: _SliceInput
    request_id: int
    empty: bool = False

    @classmethod
    def make_empty(
        cls,
        *,
        slice_input: _SliceInput,
        rgb: bool,
        request_id: int | None = None,
    ) -> '_ImageSliceResponse':
        """Returns an empty image slice response.

        An empty slice indicates that there is no valid slice data for an
        image layer, but allows other functionality that relies on slice
        data existing to continue to work without special casing.

        Parameters
        ----------
        slice_input : _SliceInput
            Describes the slicing plane or bounding box in the layer's dimensions.
        rgb : bool
            True if the underlying image is an RGB or RGBA image (i.e. that the
            last dimension represents a color channel that should not be sliced),
            False otherwise.
        request_id : int | None
            The request id for which we are responding with an empty slice.
            If None, a new request id will be returned, which guarantees that
            the empty slice never appears as loaded. (Used for layer
            initialisation before attempting data loading.)
        """
        shape = (1,) * slice_input.ndisplay
        if rgb:
            shape = shape + (3,)
        data = np.zeros(shape, dtype=np.uint8)
        image = _ImageView.from_view(data)
        ndim = slice_input.ndim
        tile_to_data = Affine(
            name='tile2data', linear_matrix=np.eye(ndim), ndim=ndim
        )
        if request_id is None:
            request_id = _next_request_id()
        return _ImageSliceResponse(
            image=image,
            thumbnail=image,
            tile_to_data=tile_to_data,
            slice_input=slice_input,
            request_id=request_id,
            empty=True,
        )

    def to_displayed(
        self, converter: Callable[[np.ndarray], np.ndarray]
    ) -> '_ImageSliceResponse':
        """
        Returns a raw slice converted for display,
        which is needed for Labels and Image.

        Parameters
        ----------
        converter : Callable[[np.ndarray], np.ndarray]
            A function that converts the raw image to a vispy viewable image.

        Returns
        -------
        _ImageSliceResponse
            Contains the converted image and thumbnail.
        """
        image = _ImageView.from_raw(raw=self.image.raw, converter=converter)
        thumbnail = image
        if self.thumbnail is not self.image:
            thumbnail = _ImageView.from_raw(
                raw=self.thumbnail.raw, converter=converter
            )
        return _ImageSliceResponse(
            image=image,
            thumbnail=thumbnail,
            tile_to_data=self.tile_to_data,
            slice_input=self.slice_input,
            request_id=self.request_id,
        )


@dataclass(frozen=True)
class _ImageSliceRequest:
    """A callable that stores all the input data needed to slice an image layer.

    This should be treated a deeply immutable structure, even though some
    fields can be modified in place. It is like a function that has captured
    all its inputs already.

    In general, the calling an instance of this may take a long time, so you may
    want to run it off the main thread.

    Attributes
    ----------
    slice_input : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    data : Any
        The layer's data field, which is the main input to slicing.
    data_slice : _ThickNDSlice
        The slicing coordinates and margins in data space.
    others
        See the corresponding attributes in `Layer` and `Image`.
    id : int
        The identifier of this slice request.
    """

    slice_input: _SliceInput
    data: Any = field(repr=False)
    dask_indexer: DaskIndexer
    data_slice: _ThickNDSlice
    projection_mode: ImageProjectionMode
    multiscale: bool = field(repr=False)
    corner_pixels: np.ndarray
    rgb: bool = field(repr=False)
    data_level: int = field(repr=False)
    thumbnail_level: int = field(repr=False)
    level_shapes: np.ndarray = field(repr=False)
    downsample_factors: np.ndarray = field(repr=False)
    id: int = field(default_factory=_next_request_id)

    def __call__(self) -> _ImageSliceResponse:
        if self._slice_out_of_bounds():
            return _ImageSliceResponse.make_empty(
                slice_input=self.slice_input, rgb=self.rgb, request_id=self.id
            )
        with self.dask_indexer():
            return (
                self._call_multi_scale()
                if self.multiscale
                else self._call_single_scale()
            )

    def _call_single_scale(self) -> _ImageSliceResponse:
        order = self._get_order()
        data = self._project_thick_slice(self.data, self.data_slice)
        data = np.transpose(data, order)
        image = _ImageView.from_view(data)
        # `Layer.multiscale` is mutable so we need to pass back the identity
        # transform to ensure `tile2data` is properly set on the layer.
        ndim = self.slice_input.ndim
        tile_to_data = Affine(
            name='tile2data', linear_matrix=np.eye(ndim), ndim=ndim
        )
        return _ImageSliceResponse(
            image=image,
            thumbnail=image,
            tile_to_data=tile_to_data,
            slice_input=self.slice_input,
            request_id=self.id,
        )

    def _call_multi_scale(self) -> _ImageSliceResponse:
        if self.slice_input.ndisplay == 3:
            level = len(self.data) - 1
        else:
            level = self.data_level

        # Calculate the tile-to-data transform.
        scale = np.ones(self.slice_input.ndim)
        for d in self.slice_input.displayed:
            scale[d] = self.downsample_factors[level][d]

        data = self.data[level]

        translate = np.zeros(self.slice_input.ndim)
        disp_slice = [slice(None) for _ in data.shape]
        if self.slice_input.ndisplay == 2:
            for d in self.slice_input.displayed:
                disp_slice[d] = slice(
                    self.corner_pixels[0, d],
                    self.corner_pixels[1, d] + 1,
                    1,
                )
            translate = self.corner_pixels[0] * scale

        # This only needs to be a ScaleTranslate but different types
        # of transforms in a chain don't play nicely together right now.
        tile_to_data = Affine(
            name='tile2data',
            scale=scale,
            translate=translate,
            ndim=self.slice_input.ndim,
        )

        # slice displayed dimensions to get the right tile data
        data = data[tuple(disp_slice)]

        # project the thick slice
        data_slice = self._thick_slice_at_level(level)
        data = self._project_thick_slice(data, data_slice)

        order = self._get_order()
        data = np.transpose(data, order)
        image = _ImageView.from_view(data)

        thumbnail_data_slice = self._thick_slice_at_level(self.thumbnail_level)
        thumbnail_data = self._project_thick_slice(
            self.data[self.thumbnail_level], thumbnail_data_slice
        )
        thumbnail_data = np.transpose(thumbnail_data, order)
        thumbnail = _ImageView.from_view(thumbnail_data)

        return _ImageSliceResponse(
            image=image,
            thumbnail=thumbnail,
            tile_to_data=tile_to_data,
            slice_input=self.slice_input,
            request_id=self.id,
        )

    def _thick_slice_at_level(self, level: int) -> _ThickNDSlice:
        """
        Get the data_slice rescaled for a specific level.
        """
        slice_arr = self.data_slice.as_array()
        # downsample based on level
        slice_arr /= self.downsample_factors[level]
        slice_arr[0] = np.clip(slice_arr[0], 0, self.level_shapes[level] - 1)
        return _ThickNDSlice.from_array(slice_arr)

    def _project_thick_slice(
        self, data: ArrayLike, data_slice: _ThickNDSlice
    ) -> np.ndarray:
        """
        Slice the given data with the given data slice and project the extra dims.

        This is also responsible for materializing the data if it is backed
        by a lazy store or compute graph (e.g. dask).
        """

        if self.projection_mode == 'none':
            # early return with only the dims point being used
            slices = self._point_to_slices(data_slice.point)
            return np.asarray(data[slices])

        slices = self._data_slice_to_slices(
            data_slice, self.slice_input.displayed
        )

        return project_slice(
            data=np.asarray(data[slices]),
            axis=tuple(self.slice_input.not_displayed),
            mode=self.projection_mode,
        )

    def _get_order(self) -> tuple[int, ...]:
        """Return the ordered displayed dimensions, but reduced to fit in the slice space."""
        order = reorder_after_dim_reduction(self.slice_input.displayed)
        if self.rgb:
            # if rgb need to keep the final axis fixed during the
            # transpose. The index of the final axis depends on how many
            # axes are displayed.
            return (*order, max(order) + 1)
        return order

    def _slice_out_of_bounds(self) -> bool:
        """Check if the data slice is out of bounds for any dimension."""
        data = self.data[0] if self.multiscale else self.data
        for d in self.slice_input.not_displayed:
            pt = self.data_slice.point[d]
            max_idx = data.shape[d] - 1
            if self.projection_mode == 'none':
                if np.round(pt) < 0 or np.round(pt) > max_idx:
                    return True
            else:
                pt = self.data_slice.point[d]
                low = np.round(pt - self.data_slice.margin_left[d])
                high = np.round(pt + self.data_slice.margin_right[d])
                if high < 0 or low > max_idx:
                    return True
        return False

    @staticmethod
    def _point_to_slices(
        point: tuple[float, ...],
    ) -> tuple[slice | int, ...]:
        # no need to check out of bounds here cause it's guaranteed

        # values in point and margins are np.nan if no slicing should happen along that dimension
        # which is always the case for displayed dims, so that becomes `slice(None)` for actually
        # indexing the layer.
        # For the rest, indices are rounded to the closest integer
        return tuple(
            slice(None) if np.isnan(p) else int(np.round(p)) for p in point
        )

    @staticmethod
    def _data_slice_to_slices(
        data_slice: _ThickNDSlice, dims_displayed: list[int]
    ) -> tuple[slice, ...]:
        slices = [slice(None) for _ in range(data_slice.ndim)]

        for dim, (point, m_left, m_right) in enumerate(data_slice):
            if dim in dims_displayed:
                # leave slice(None) for displayed dimensions
                # point and margin values here are np.nan; if np.nans pass through this check,
                # something is likely wrong with the data_slice creation at a previous step!
                continue

            # max here ensures we don't start slicing from negative values (=end of array)
            low = max(int(np.round(point - m_left)), 0)
            high = max(int(np.round(point + m_right)), 0)

            # if high is already exactly at an integer value, we need to round up
            # to next integer because slices have non-inclusive stop
            if np.isclose(high, point + m_right):
                high += 1

            # ensure we always get at least 1 slice (we're guaranteed to be
            # in bounds from a previous check)
            if low == high:
                high += 1

            slices[dim] = slice(low, high)

        return tuple(slices)
