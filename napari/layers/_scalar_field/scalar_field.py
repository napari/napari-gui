from __future__ import annotations

import types
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy import typing as npt

from napari.layers import Layer
from napari.layers._data_protocols import LayerDataProtocol
from napari.layers._multiscale_data import MultiScaleData
from napari.layers.image._image_constants import Interpolation, VolumeDepiction
from napari.layers.image._image_mouse_bindings import (
    move_plane_along_normal as plane_drag_callback,
    set_plane_position as plane_double_click_callback,
)
from napari.layers.image._image_utils import guess_multiscale
from napari.layers.image._slice import _ImageSliceRequest, _ImageSliceResponse
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice
from napari.layers.utils.plane import SlicingPlane
from napari.utils._dask_utils import DaskIndexer
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.utils.events import Event
from napari.utils.events.event import WarningEmitter
from napari.utils.events.event_utils import connect_no_arg
from napari.utils.geometry import clamp_point_to_bounding_box
from napari.utils.naming import magic_name
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.components import Dims


__all__ = ('ScalarFieldBase',)


# It is important to contain at least one abstractmethod to properly exclude this class
# in creating NAMES set inside of napari.layers.__init__
# Mixin must come before Layer
class ScalarFieldBase(Layer, ABC):
    """Base class for volumetric layers.

    Parameters
    ----------
    data : array or list of array
        Image data. Can be N >= 2 dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a
        list and arrays are decreasing in shape then the data is treated as
        a multiscale image. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    axis_labels : tuple of str, optional
        Dimension names of the layer data.
        If not provided, axis_labels will be set to (..., 'axis -2', 'axis -1').
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', 'translucent_no_depth', 'additive', and 'minimum'}.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    custom_interpolation_kernel_2d : np.ndarray
        Convolution kernel used with the 'custom' interpolation mode in 2D rendering.
    depiction : str
        3D Depiction mode. Must be one of {'volume', 'plane'}.
        The default value is 'volume'.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.
    metadata : dict
        Layer metadata.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape
        then it will be taken to be multiscale. The first image in the list
        should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    name : str
        Name of the layer. If not provided then will be guessed using heuristics.
    ndim : int
        Number of dimensions in the data.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    plane : dict or SlicingPlane
        Properties defining plane rendering in 3D. Properties are defined in
        data coordinates. Valid dictionary keys are
        {'position', 'normal', 'thickness', and 'enabled'}.
    projection_mode : str
        How data outside the viewed dimensions but inside the thick Dims slice will
        be projected onto the viewed dimensions. Must fit to cls._projectionclass.
    rendering : str
        Rendering mode used by vispy. Must be one of our supported
        modes.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    scale : tuple of float
        Scale factors for the layer.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    translate : tuple of float
        Translation values for the layer.
    units : tuple of str or pint.Unit, optional
        Units of the layer data in world coordinates.
        If not provided, the default units are assumed to be pixels.
    visible : bool
        Whether the layer visual is currently being displayed.


    Attributes
    ----------
    data : array or list of array
        Image data. Can be N dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a list
        and arrays are decreasing in shape then the data is treated as a
        multiscale image. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    axis_labels : tuple of str
        Dimension names of the layer data.
    custom_interpolation_kernel_2d : np.ndarray
        Convolution kernel used with the 'custom' interpolation mode in 2D rendering.
    depiction : str
        3D Depiction mode used by vispy. Must be one of our supported modes.
    experimental_clipping_planes : ClippingPlaneList
        Clipping planes defined in data coordinates, used to clip the volume.
    metadata : dict
        Image metadata.
    mode : str
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In TRANSFORM mode the image can be transformed interactively.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. The first image in the
        list should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    plane : SlicingPlane or dict
        Properties defining plane rendering in 3D. Valid dictionary keys are
        {'position', 'normal', 'thickness'}.
    rendering : str
        Rendering mode used by vispy. Must be one of our supported
        modes.
    units: tuple of pint.Unit
        Units of the layer data in world coordinates.

    Notes
    -----
    _data_view : array (N, M), (N, M, 3), or (N, M, 4)
        Image data for the currently viewed slice. Must be 2D image data, but
        can be multidimensional for RGB or RGBA images if multidimensional is
        `True`.
    """

    _colormaps = AVAILABLE_COLORMAPS
    _interpolation2d: Interpolation
    _interpolation3d: Interpolation

    def __init__(
        self,
        data,
        *,
        affine=None,
        axis_labels=None,
        blending='translucent',
        cache=True,
        custom_interpolation_kernel_2d=None,
        depiction='volume',
        experimental_clipping_planes=None,
        metadata=None,
        multiscale=None,
        name=None,
        ndim=None,
        opacity=1.0,
        plane=None,
        projection_mode='none',
        rendering='mip',
        rotate=None,
        scale=None,
        shear=None,
        translate=None,
        units=None,
        visible=True,
    ):
        if name is None and data is not None:
            name = magic_name(data)

        if isinstance(data, types.GeneratorType):
            data = list(data)

        if getattr(data, 'ndim', 2) < 2:
            raise ValueError(
                trans._('Image data must have at least 2 dimensions.')
            )

        # Determine if data is a multiscale
        self._data_raw = data
        if multiscale is None:
            multiscale, data = guess_multiscale(data)
        elif multiscale and not isinstance(data, MultiScaleData):
            data = MultiScaleData(data)

        # Determine dimensionality of the data
        if ndim is None:
            ndim = len(data.shape)

        super().__init__(
            data,
            ndim,
            affine=affine,
            axis_labels=axis_labels,
            blending=blending,
            cache=cache,
            experimental_clipping_planes=experimental_clipping_planes,
            metadata=metadata,
            multiscale=multiscale,
            name=name,
            opacity=opacity,
            projection_mode=projection_mode,
            scale=scale,
            shear=shear,
            rotate=rotate,
            translate=translate,
            units=units,
            visible=visible,
        )

        self.events.add(
            attenuation=Event,
            custom_interpolation_kernel_2d=Event,
            depiction=Event,
            interpolation=WarningEmitter(
                trans._(
                    "'layer.events.interpolation' is deprecated please use `interpolation2d` and `interpolation3d`",
                    deferred=True,
                ),
                type_name='select',
            ),
            interpolation2d=Event,
            interpolation3d=Event,
            iso_threshold=Event,
            plane=Event,
            rendering=Event,
        )

        self._array_like = True

        # Set data
        self._data = data
        if isinstance(data, MultiScaleData):
            self._data_level = len(data) - 1
            # Determine which level of the multiscale to use for the thumbnail.
            # Pick the smallest level with at least one axis >= 64. This is
            # done to prevent the thumbnail from being from one of the very
            # low resolution layers and therefore being very blurred.
            big_enough_levels = [
                np.any(np.greater_equal(p.shape, 64)) for p in data
            ]
            if np.any(big_enough_levels):
                self._thumbnail_level = np.where(big_enough_levels)[0][-1]
            else:
                self._thumbnail_level = 0
        else:
            self._data_level = 0
            self._thumbnail_level = 0
        displayed_axes = self._slice_input.displayed
        self.corner_pixels[1][displayed_axes] = (
            np.array(self.level_shapes)[self._data_level][displayed_axes] - 1
        )

        self._slice = _ImageSliceResponse.make_empty(
            slice_input=self._slice_input,
            rgb=len(self.data.shape) != self.ndim,
        )

        self._plane = SlicingPlane(thickness=1)
        # Whether to calculate clims on the next set_view_slice
        self._should_calc_clims = False
        # using self.colormap = colormap uses the setter in *derived* classes,
        # where the intention here is to use the base setter, so we use the
        # _set_colormap method. This is important for Labels layers, because
        # we don't want to use get_color before set_view_slice has been
        # triggered (self.refresh(), below).
        self.rendering = rendering
        self.depiction = depiction
        if plane is not None:
            self.plane = plane
        connect_no_arg(self.plane.events, self.events, 'plane')
        self.custom_interpolation_kernel_2d = custom_interpolation_kernel_2d

    def _post_init(self):
        # Trigger generation of view slice and thumbnail
        self.refresh()

    @property
    def _data_view(self) -> np.ndarray:
        """Viewable image for the current slice. (compatibility)"""
        return self._slice.image.view

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def data_raw(
        self,
    ) -> LayerDataProtocol | Sequence[LayerDataProtocol]:
        """Data, exactly as provided by the user."""
        return self._data_raw

    def _get_ndim(self) -> int:
        """Determine number of dimensions of the layer."""
        return len(self.level_shapes[0])

    @property
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        shape = self.level_shapes[0]
        return np.vstack([np.zeros(len(shape)), shape - 1])

    @property
    def _extent_data_augmented(self) -> np.ndarray:
        extent = self._extent_data
        return extent + [[-0.5], [+0.5]]

    @property
    def _extent_level_data(self) -> np.ndarray:
        """Extent of layer, accounting for current multiscale level, in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        shape = self.level_shapes[self.data_level]
        return np.vstack([np.zeros(len(shape)), shape - 1])

    @property
    def _extent_level_data_augmented(self) -> np.ndarray:
        extent = self._extent_level_data
        return extent + [[-0.5], [+0.5]]

    @property
    def data_level(self) -> int:
        """int: Current level of multiscale, or 0 if image."""
        return self._data_level

    @data_level.setter
    def data_level(self, level: int) -> None:
        if self._data_level == level:
            return
        self._data_level = level
        self.refresh(extent=False)

    def _get_level_shapes(self):
        data = self.data
        if isinstance(data, MultiScaleData):
            shapes = data.shapes
        else:
            shapes = [self.data.shape]
        return shapes

    @property
    def level_shapes(self) -> np.ndarray:
        """array: Shapes of each level of the multiscale or just of image."""
        return np.array(self._get_level_shapes())

    @property
    def downsample_factors(self) -> np.ndarray:
        """list: Downsample factors for each level of the multiscale."""
        return np.divide(self.level_shapes[0], self.level_shapes)

    @property
    def depiction(self):
        """The current 3D depiction mode.

        Selects a preset depiction mode in vispy
            * volume: images are rendered as 3D volumes.
            * plane: images are rendered as 2D planes embedded in 3D.
                plane position, normal, and thickness are attributes of
                layer.plane which can be modified directly.
        """
        return str(self._depiction)

    @depiction.setter
    def depiction(self, depiction: str | VolumeDepiction) -> None:
        """Set the current 3D depiction mode."""
        self._depiction = VolumeDepiction(depiction)
        self._update_plane_callbacks()
        self.events.depiction()

    def _reset_plane_parameters(self):
        """Set plane attributes to something valid."""
        self.plane.position = np.array(self.data.shape) / 2
        self.plane.normal = (1, 0, 0)

    def _update_plane_callbacks(self):
        """Set plane callbacks depending on depiction mode."""
        plane_drag_callback_connected = (
            plane_drag_callback in self.mouse_drag_callbacks
        )
        double_click_callback_connected = (
            plane_double_click_callback in self.mouse_double_click_callbacks
        )
        if self.depiction == VolumeDepiction.VOLUME:
            if plane_drag_callback_connected:
                self.mouse_drag_callbacks.remove(plane_drag_callback)
            if double_click_callback_connected:
                self.mouse_double_click_callbacks.remove(
                    plane_double_click_callback
                )
        elif self.depiction == VolumeDepiction.PLANE:
            if not plane_drag_callback_connected:
                self.mouse_drag_callbacks.append(plane_drag_callback)
            if not double_click_callback_connected:
                self.mouse_double_click_callbacks.append(
                    plane_double_click_callback
                )

    @property
    def plane(self):
        return self._plane

    @plane.setter
    def plane(self, value: dict | SlicingPlane) -> None:
        self._plane.update(value)
        self.events.plane()

    @property
    def custom_interpolation_kernel_2d(self):
        return self._custom_interpolation_kernel_2d

    @custom_interpolation_kernel_2d.setter
    def custom_interpolation_kernel_2d(self, value):
        if value is None:
            value = [[1]]
        self._custom_interpolation_kernel_2d = np.array(value, np.float32)
        self.events.custom_interpolation_kernel_2d()

    @abstractmethod
    def _raw_to_displayed(self, raw: np.ndarray) -> np.ndarray:
        """Determine displayed image from raw image.

        For normal image layers, just return the actual image.

        Parameters
        ----------
        raw : array
            Raw array.

        Returns
        -------
        image : array
            Displayed array.
        """
        raise NotImplementedError

    def _set_view_slice(self) -> None:
        """Set the slice output based on this layer's current state."""
        # The new slicing code makes a request from the existing state and
        # executes the request on the calling thread directly.
        # For async slicing, the calling thread will not be the main thread.
        request = self._make_slice_request_internal(
            slice_input=self._slice_input,
            data_slice=self._data_slice,
            dask_indexer=nullcontext,
        )
        response = request()
        self._update_slice_response(response)

    def _make_slice_request(self, dims: Dims) -> _ImageSliceRequest:
        """Make an image slice request based on the given dims and this image."""
        slice_input = self._make_slice_input(dims)
        # For the existing sync slicing, indices is passed through
        # to avoid some performance issues related to the evaluation of the
        # data-to-world transform and its inverse. Async slicing currently
        # absorbs these performance issues here, but we can likely improve
        # things either by caching the world-to-data transform on the layer
        # or by lazily evaluating it in the slice task itself.
        indices = slice_input.data_slice(self._data_to_world.inverse)
        return self._make_slice_request_internal(
            slice_input=slice_input,
            data_slice=indices,
            dask_indexer=self.dask_optimized_slicing,
        )

    def _make_slice_request_internal(
        self,
        *,
        slice_input: _SliceInput,
        data_slice: _ThickNDSlice,
        dask_indexer: DaskIndexer,
    ) -> _ImageSliceRequest:
        """Needed to support old-style sync slicing through _slice_dims and
        _set_view_slice.

        This is temporary scaffolding that should go away once we have completed
        the async slicing project: https://github.com/napari/napari/issues/4795
        """
        return _ImageSliceRequest(
            slice_input=slice_input,
            data=self.data,
            dask_indexer=dask_indexer,
            data_slice=data_slice,
            projection_mode=self.projection_mode,
            multiscale=self.multiscale,
            corner_pixels=self.corner_pixels,
            rgb=len(self.data.shape) != self.ndim,
            data_level=self.data_level,
            thumbnail_level=self._thumbnail_level,
            level_shapes=self.level_shapes,
            downsample_factors=self.downsample_factors,
        )

    def _update_slice_response(self, response: _ImageSliceResponse) -> None:
        """Update the slice output state currently on the layer. Currently used
        for both sync and async slicing.
        """
        response = response.to_displayed(self._raw_to_displayed)
        # We call to_displayed here to ensure that if the contrast limits
        # are outside the range of supported by vispy, then data view is
        # rescaled to fit within the range.
        self._slice_input = response.slice_input
        self._transforms[0] = response.tile_to_data
        self._slice = response

    def _get_value(self, position):
        """Value of the data at a position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : tuple
            Value of the data.
        """
        if self.multiscale:
            # for multiscale data map the coordinate from the data back to
            # the tile
            coord = self._transforms['tile2data'].inverse(position)
        else:
            coord = position

        coord = np.round(coord).astype(int)

        raw = self._slice.image.raw
        shape = (
            raw.shape[:-1] if self.ndim != len(self._data.shape) else raw.shape
        )

        if self.ndim < len(coord):
            # handle 3D views of 2D data by omitting extra coordinate
            offset = len(coord) - len(shape)
            coord = coord[[d + offset for d in self._slice_input.displayed]]
        else:
            coord = coord[self._slice_input.displayed]

        if all(0 <= c < s for c, s in zip(coord, shape, strict=False)):
            value = raw[tuple(coord)]
        else:
            value = None

        if self.multiscale:
            value = (self.data_level, value)

        return value

    def _get_value_ray(
        self,
        start_point: np.ndarray | None,
        end_point: np.ndarray | None,
        dims_displayed: list[int],
    ) -> int | None:
        """Get the first non-background value encountered along a ray.

        Parameters
        ----------
        start_point : np.ndarray
            (n,) array containing the start point of the ray in data coordinates.
        end_point : np.ndarray
            (n,) array containing the end point of the ray in data coordinates.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the viewer.

        Returns
        -------
        value : Optional[int]
            The first non-background value encountered along the ray. If none
            was encountered or the viewer is in 2D mode, returns None.
        """
        if start_point is None or end_point is None:
            return None
        if len(dims_displayed) == 3:
            # only use get_value_ray on 3D for now
            # we use dims_displayed because the image slice
            # has its dimensions  in th same order as the vispy
            # Volume
            # Account for downsampling in the case of multiscale
            # -1 means lowest resolution here.
            start_point = (
                start_point[dims_displayed]
                / self.downsample_factors[-1][dims_displayed]
            )
            end_point = (
                end_point[dims_displayed]
                / self.downsample_factors[-1][dims_displayed]
            )
            start_point = cast(np.ndarray, start_point)
            end_point = cast(np.ndarray, end_point)
            sample_ray = end_point - start_point
            length_sample_vector = np.linalg.norm(sample_ray)
            n_points = int(2 * length_sample_vector)
            sample_points = np.linspace(
                start_point, end_point, n_points, endpoint=True
            )
            im_slice = self._slice.image.raw
            # ensure the bounding box is for the proper multiscale level
            bounding_box = self._display_bounding_box_at_level(
                dims_displayed, self.data_level
            )
            # the display bounding box is returned as a closed interval
            # (i.e. the endpoint is included) by the method, but we need
            # open intervals in the code that follows, so we add 1.
            bounding_box[:, 1] += 1

            clamped = clamp_point_to_bounding_box(
                sample_points,
                bounding_box,
            ).astype(int)
            values = im_slice[tuple(clamped.T)]
            return self._calculate_value_from_ray(values)

        return None

    @abstractmethod
    def _calculate_value_from_ray(self, values):
        raise NotImplementedError

    def _get_value_3d(
        self,
        start_point: np.ndarray | None,
        end_point: np.ndarray | None,
        dims_displayed: list[int],
    ) -> int | None:
        """Get the first non-background value encountered along a ray.

        Parameters
        ----------
        start_point : np.ndarray
            (n,) array containing the start point of the ray in data coordinates.
        end_point : np.ndarray
            (n,) array containing the end point of the ray in data coordinates.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the viewer.

        Returns
        -------
        value : int
            The first non-zero value encountered along the ray. If a
            non-zero value is not encountered, returns None.
        """
        return self._get_value_ray(
            start_point=start_point,
            end_point=end_point,
            dims_displayed=dims_displayed,
        )

    def _get_offset_data_position(self, position: npt.NDArray) -> npt.NDArray:
        """Adjust position for offset between viewer and data coordinates.

        VisPy considers the coordinate system origin to be the canvas corner,
        while napari considers the origin to be the **center** of the corner
        pixel. To get the correct value under the mouse cursor, we need to
        shift the position by 0.5 pixels on each axis.
        """
        return position + 0.5

    def _display_bounding_box_at_level(
        self, dims_displayed: list[int], data_level: int
    ) -> npt.NDArray:
        """An axis aligned (ndisplay, 2) bounding box around the data at a given level"""
        shape = self.level_shapes[data_level]
        extent_at_level = np.vstack([np.zeros(len(shape)), shape - 1])
        return extent_at_level[:, dims_displayed].T

    def _display_bounding_box_augmented_data_level(
        self, dims_displayed: list[int]
    ) -> npt.NDArray:
        """An augmented, axis-aligned (ndisplay, 2) bounding box.
        If the layer is multiscale layer, then returns the
        bounding box of the data at the current level
        """
        return self._extent_level_data_augmented[:, dims_displayed].T
