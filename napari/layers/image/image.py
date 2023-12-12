"""Image class.
"""
from __future__ import annotations

import types
import warnings
from contextlib import nullcontext
from typing import TYPE_CHECKING, List, Sequence, Tuple, Union, cast

import numpy as np
from scipy import ndimage as ndi

from napari.layers._data_protocols import LayerDataProtocol
from napari.layers._multiscale_data import MultiScaleData
from napari.layers.base import Layer
from napari.layers.image._image_constants import (
    ImageProjectionMode,
    ImageRendering,
    Interpolation,
    InterpolationStr,
    VolumeDepiction,
)
from napari.layers.image._image_mouse_bindings import (
    move_plane_along_normal as plane_drag_callback,
    set_plane_position as plane_double_click_callback,
)
from napari.layers.image._image_utils import guess_multiscale, guess_rgb
from napari.layers.image._slice import _ImageSliceRequest, _ImageSliceResponse
from napari.layers.intensity_mixin import IntensityVisualizationMixin
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice
from napari.layers.utils.layer_utils import calc_data_range
from napari.layers.utils.plane import SlicingPlane
from napari.utils._dask_utils import DaskIndexer
from napari.utils._dtype import get_dtype_limits, normalize_dtype
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.utils.events import Event
from napari.utils.events.event import WarningEmitter
from napari.utils.events.event_utils import connect_no_arg
from napari.utils.migrations import rename_argument
from napari.utils.naming import magic_name
from napari.utils.translations import trans

if TYPE_CHECKING:
    import numpy.typing as npt

    from napari.components import Dims


# It is important to contain at least one abstractmethod to properly exclude this class
# in creating NAMES set inside of napari.layers.__init__
# Mixin must come before Layer
class _ImageBase(IntensityVisualizationMixin, Layer):
    """Image layer.

    Parameters
    ----------
    data : array or list of array
        Image data. Can be N >= 2 dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a
        list and arrays are decreasing in shape then the data is treated as
        a multiscale image. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    rgb : bool
        Whether the image is rgb RGB or RGBA. If not specified by user and
        the last dimension of the data has length 3 or 4 it will be set as
        `True`. If `False` the image is interpreted as a luminance image.
    colormap : str, napari.utils.Colormap, tuple, dict
        Colormap to use for luminance images. If a string must be the name
        of a supported colormap from vispy or matplotlib. If a tuple the
        first value must be a string to assign as a name to a colormap and
        the second item must be a Colormap. If a dict the key must be a
        string to assign as a name to a colormap and the value must be a
        Colormap.
    contrast_limits : list (2,)
        Color limits to be used for determining the colormap bounds for
        luminance images. If not passed is calculated as the min and max of
        the image.
    gamma : float
        Gamma correction for determining colormap linearity. Defaults to 1.
    interpolation : str
        Interpolation mode used by vispy. Must be one of our supported modes.
        'custom' is a special mode for 2D interpolation in which a regular grid
        of samples are taken from the texture around a position using 'linear'
        interpolation before being multiplied with a custom interpolation kernel
        (provided with 'custom_interpolation_kernel_2d').
    rendering : str
        Rendering mode used by vispy. Must be one of our supported
        modes.
    depiction : str
        3D Depiction mode. Must be one of {'volume', 'plane'}.
        The default value is 'volume'.
    iso_threshold : float
        Threshold for isosurface.
    attenuation : float
        Attenuation rate for attenuated maximum intensity projection.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape
        then it will be taken to be multiscale. The first image in the list
        should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    plane : dict or SlicingPlane
        Properties defining plane rendering in 3D. Properties are defined in
        data coordinates. Valid dictionary keys are
        {'position', 'normal', 'thickness', and 'enabled'}.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.
    custom_interpolation_kernel_2d : np.ndarray
        Convolution kernel used with the 'custom' interpolation mode in 2D rendering.

    Attributes
    ----------
    data : array or list of array
        Image data. Can be N dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a list
        and arrays are decreasing in shape then the data is treated as a
        multiscale image. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    metadata : dict
        Image metadata.
    rgb : bool
        Whether the image is rgb RGB or RGBA if rgb. If not
        specified by user and the last dimension of the data has length 3 or 4
        it will be set as `True`. If `False` the image is interpreted as a
        luminance image.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. The first image in the
        list should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    mode : str
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In TRANSFORM mode the image can be transformed interactively.
    colormap : 2-tuple of str, napari.utils.Colormap
        The first is the name of the current colormap, and the second value is
        the colormap. Colormaps are used for luminance images, if the image is
        rgb the colormap is ignored.
    colormaps : tuple of str
        Names of the available colormaps.
    contrast_limits : list (2,) of float
        Color limits to be used for determining the colormap bounds for
        luminance images. If the image is rgb the contrast_limits is ignored.
    contrast_limits_range : list (2,) of float
        Range for the color limits for luminance images. If the image is
        rgb the contrast_limits_range is ignored.
    gamma : float
        Gamma correction for determining colormap linearity.
    interpolation : str
        Interpolation mode used by vispy. Must be one of our supported modes.
        'custom' is a special mode for 2D interpolation in which a regular grid
        of samples are taken from the texture around a position using 'linear'
        interpolation before being multiplied with a custom interpolation kernel
        (provided with 'custom_interpolation_kernel_2d').
    rendering : str
        Rendering mode used by vispy. Must be one of our supported
        modes.
    depiction : str
        3D Depiction mode used by vispy. Must be one of our supported modes.
    iso_threshold : float
        Threshold for isosurface.
    attenuation : float
        Attenuation rate for attenuated maximum intensity projection.
    plane : SlicingPlane or dict
        Properties defining plane rendering in 3D. Valid dictionary keys are
        {'position', 'normal', 'thickness'}.
    experimental_clipping_planes : ClippingPlaneList
        Clipping planes defined in data coordinates, used to clip the volume.
    custom_interpolation_kernel_2d : np.ndarray
        Convolution kernel used with the 'custom' interpolation mode in 2D rendering.

    Notes
    -----
    _data_view : array (N, M), (N, M, 3), or (N, M, 4)
        Image data for the currently viewed slice. Must be 2D image data, but
        can be multidimensional for RGB or RGBA images if multidimensional is
        `True`.
    _colorbar : array
        Colorbar for current colormap.
    """

    _colormaps = AVAILABLE_COLORMAPS
    _interpolation2d: Interpolation
    _interpolation3d: Interpolation

    @rename_argument(
        from_name="interpolation",
        to_name="interpolation2d",
        version="0.6.0",
        since_version="0.4.17",
    )
    def __init__(
        self,
        data,
        *,
        rgb=None,
        colormap='gray',
        contrast_limits=None,
        gamma=1.0,
        interpolation2d='nearest',
        interpolation3d='linear',
        rendering='mip',
        iso_threshold=None,
        attenuation=0.05,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=1.0,
        blending='translucent',
        visible=True,
        multiscale=None,
        cache=True,
        depiction='volume',
        plane=None,
        experimental_clipping_planes=None,
        custom_interpolation_kernel_2d=None,
        projection_mode='none',
    ) -> None:
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

        # Determine if rgb
        rgb_guess = guess_rgb(data.shape)
        if rgb and not rgb_guess:
            raise ValueError(
                trans._(
                    "'rgb' was set to True but data does not have suitable dimensions."
                )
            )
        if rgb is None:
            rgb = rgb_guess
        self.rgb = rgb

        # Determine dimensionality of the data
        ndim = len(data.shape)
        if rgb:
            ndim -= 1

        super().__init__(
            data,
            ndim,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            rotate=rotate,
            shear=shear,
            affine=affine,
            opacity=opacity,
            blending=blending,
            visible=visible,
            multiscale=multiscale,
            cache=cache,
            experimental_clipping_planes=experimental_clipping_planes,
            projection_mode=projection_mode,
        )

        self.events.add(
            interpolation=WarningEmitter(
                trans._(
                    "'layer.events.interpolation' is deprecated please use `interpolation2d` and `interpolation3d`",
                    deferred=True,
                ),
                type_name='select',
            ),
            interpolation2d=Event,
            interpolation3d=Event,
            rendering=Event,
            plane=Event,
            depiction=Event,
            iso_threshold=Event,
            attenuation=Event,
            custom_interpolation_kernel_2d=Event,
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
            slice_input=self._slice_input, rgb=self.rgb
        )

        # Set contrast limits, colormaps and plane parameters
        self._gamma = gamma
        self._attenuation = attenuation
        self._plane = SlicingPlane(thickness=1, enabled=False, draggable=True)
        # Whether to calculate clims on the next set_view_slice
        self._should_calc_clims = False
        if contrast_limits is None:
            if not isinstance(data, np.ndarray):
                dtype = normalize_dtype(getattr(data, 'dtype', None))
                if np.issubdtype(dtype, np.integer):
                    self.contrast_limits_range = get_dtype_limits(dtype)
                else:
                    self.contrast_limits_range = (0, 1)
                self._should_calc_clims = dtype != np.uint8
            else:
                # Delay contrast limit calculation
                self.contrast_limits_range = (0, 1)
                self._should_calc_clims = True
                # self.contrast_limits_range = self._calc_data_range()
        else:
            self.contrast_limits_range = contrast_limits
        self._contrast_limits: Tuple[float, float] = self.contrast_limits_range
        if iso_threshold is None:
            cmin, cmax = self.contrast_limits_range
            self._iso_threshold = cmin + (cmax - cmin) / 2
        else:
            self._iso_threshold = iso_threshold
        # using self.colormap = colormap uses the setter in *derived* classes,
        # where the intention here is to use the base setter, so we use the
        # _set_colormap method. This is important for Labels layers, because
        # we don't want to use get_color before set_view_slice has been
        # triggered (self.refresh(), below).
        self._set_colormap(colormap)
        self.contrast_limits = self._contrast_limits
        self._interpolation2d = Interpolation.NEAREST
        self._interpolation3d = Interpolation.NEAREST
        self.interpolation2d = interpolation2d
        self.interpolation3d = interpolation3d
        self.rendering = rendering
        self.depiction = depiction
        if plane is not None:
            self.plane = plane
        connect_no_arg(self.plane.events, self.events, 'plane')
        self.custom_interpolation_kernel_2d = custom_interpolation_kernel_2d

        # Trigger generation of view slice and thumbnail
        self.refresh()

    @property
    def _data_view(self) -> np.ndarray:
        """Viewable image for the current slice. (compatibility)"""
        return self._slice.image.view

    def _calc_data_range(self, mode='data') -> Tuple[float, float]:
        """
        Calculate the range of the data values in the currently viewed slice
        or full data array
        """
        if mode == 'data':
            input_data = self.data[-1] if self.multiscale else self.data
        elif mode == 'slice':
            data = self._slice.image.view  # ugh
            input_data = data[-1] if self.multiscale else data
        else:
            raise ValueError(
                trans._(
                    "mode must be either 'data' or 'slice', got {mode!r}",
                    deferred=True,
                    mode=mode,
                )
            )
        return calc_data_range(input_data, rgb=self.rgb)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def data_raw(
        self,
    ) -> Union[LayerDataProtocol, Sequence[LayerDataProtocol]]:
        """Data, exactly as provided by the user."""
        return self._data_raw

    @property
    def data(self) -> LayerDataProtocol:
        """Data, possibly in multiscale wrapper. Obeys LayerDataProtocol."""
        return self._data

    @data.setter
    def data(self, data: Union[LayerDataProtocol, MultiScaleData]):
        self._data_raw = data
        # note, we don't support changing multiscale in an Image instance
        self._data = MultiScaleData(data) if self.multiscale else data  # type: ignore
        self._update_dims()
        self.events.data(value=self.data)
        if self._keep_auto_contrast:
            self.reset_contrast_limits()
        self._reset_editable()

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
    def data_level(self, level: int):
        if self._data_level == level:
            return
        self._data_level = level
        self.refresh()

    @property
    def level_shapes(self) -> np.ndarray:
        """array: Shapes of each level of the multiscale or just of image."""
        data = self.data
        if isinstance(data, MultiScaleData):
            shapes = data.shapes
        else:
            shapes = [self.data.shape]
        if self.rgb:
            shapes = [s[:-1] for s in shapes]
        return np.array(shapes)

    @property
    def downsample_factors(self) -> np.ndarray:
        """list: Downsample factors for each level of the multiscale."""
        return np.divide(self.level_shapes[0], self.level_shapes)

    @property
    def iso_threshold(self):
        """float: threshold for isosurface."""
        return self._iso_threshold

    @iso_threshold.setter
    def iso_threshold(self, value):
        self._iso_threshold = value
        self._update_thumbnail()
        self.events.iso_threshold()

    @property
    def attenuation(self):
        """float: attenuation rate for attenuated_mip rendering."""
        return self._attenuation

    @attenuation.setter
    def attenuation(self, value):
        self._attenuation = value
        self._update_thumbnail()
        self.events.attenuation()

    @property
    def interpolation(self):
        """Return current interpolation mode.

        Selects a preset interpolation mode in vispy that determines how volume
        is displayed.  Makes use of the two Texture2D interpolation methods and
        the available interpolation methods defined in
        vispy/gloo/glsl/misc/spatial_filters.frag

        Options include:
        'bessel', 'cubic', 'linear', 'blackman', 'catrom', 'gaussian',
        'hamming', 'hanning', 'hermite', 'kaiser', 'lanczos', 'mitchell',
        'nearest', 'spline16', 'spline36'

        Returns
        -------
        str
            The current interpolation mode
        """
        warnings.warn(
            trans._(
                "Interpolation attribute is deprecated since 0.4.17. Please use interpolation2d or interpolation3d",
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
        return str(
            self._interpolation2d
            if self._slice_input.ndisplay == 2
            else self._interpolation3d
        )

    @interpolation.setter
    def interpolation(self, interpolation):
        """Set current interpolation mode."""
        warnings.warn(
            trans._(
                "Interpolation setting is deprecated since 0.4.17. Please use interpolation2d or interpolation3d",
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
        if self._slice_input.ndisplay == 3:
            self.interpolation3d = interpolation
        else:
            if interpolation == 'bilinear':
                interpolation = 'linear'
                warnings.warn(
                    trans._(
                        "'bilinear' is invalid for interpolation2d (introduced in napari 0.4.17). "
                        "Please use 'linear' instead, and please set directly the 'interpolation2d' attribute'.",
                    ),
                    category=DeprecationWarning,
                    stacklevel=2,
                )
            self.interpolation2d = interpolation

    @property
    def interpolation2d(self) -> InterpolationStr:
        return cast(InterpolationStr, str(self._interpolation2d))

    @interpolation2d.setter
    def interpolation2d(self, value: Union[InterpolationStr, Interpolation]):
        if value == 'bilinear':
            raise ValueError(
                trans._(
                    "'bilinear' interpolation is not valid for interpolation2d. Did you mean 'linear' instead ?",
                ),
            )
        if value == 'bicubic':
            value = 'cubic'
            warnings.warn(
                trans._("'bicubic' is deprecated. Please use 'cubic' instead"),
                category=DeprecationWarning,
                stacklevel=2,
            )
        self._interpolation2d = Interpolation(value)
        self.events.interpolation2d(value=self._interpolation2d)
        self.events.interpolation(value=self._interpolation2d)

    @property
    def interpolation3d(self) -> InterpolationStr:
        return cast(InterpolationStr, str(self._interpolation3d))

    @interpolation3d.setter
    def interpolation3d(self, value: Union[InterpolationStr, Interpolation]):
        if value == 'custom':
            raise NotImplementedError(
                'custom interpolation is not implemented yet for 3D rendering'
            )
        if value == 'bicubic':
            value = 'cubic'
            warnings.warn(
                trans._("'bicubic' is deprecated. Please use 'cubic' instead"),
                category=DeprecationWarning,
                stacklevel=2,
            )
        self._interpolation3d = Interpolation(value)
        self.events.interpolation3d(value=self._interpolation3d)
        self.events.interpolation(value=self._interpolation3d)

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
    def depiction(self, depiction: Union[str, VolumeDepiction]):
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
    def plane(self, value: Union[dict, SlicingPlane]):
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

    def _raw_to_displayed(self, raw):
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
        image = raw
        return image

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
            rgb=self.rgb,
            data_level=self.data_level,
            thumbnail_level=self._thumbnail_level,
            level_shapes=self.level_shapes,
            downsample_factors=self.downsample_factors,
        )

    def _update_slice_response(self, response: _ImageSliceResponse) -> None:
        """Update the slice output state currently on the layer. Currently used
        for both sync and async slicing.
        """
        self._slice_input = response.slice_input
        self._transforms[0] = response.tile_to_data
        self._slice = response

        # Maybe reset the contrast limits based on the new slice.
        if self._should_calc_clims:
            self.reset_contrast_limits_range()
            self.reset_contrast_limits()
            self._should_calc_clims = False
        elif self._keep_auto_contrast:
            self.reset_contrast_limits()

    def _update_thumbnail(self):
        """Update thumbnail with current image data and colormap."""
        image = self._slice.thumbnail.view

        if self._slice_input.ndisplay == 3 and self.ndim > 2:
            image = np.max(image, axis=0)

        # float16 not supported by ndi.zoom
        dtype = np.dtype(image.dtype)
        if dtype in [np.dtype(np.float16)]:
            image = image.astype(np.float32)

        raw_zoom_factor = np.divide(
            self._thumbnail_shape[:2], image.shape[:2]
        ).min()
        new_shape = np.clip(
            raw_zoom_factor * np.array(image.shape[:2]),
            1,  # smallest side should be 1 pixel wide
            self._thumbnail_shape[:2],
        )
        zoom_factor = tuple(new_shape / image.shape[:2])
        if self.rgb:
            downsampled = ndi.zoom(
                image, zoom_factor + (1,), prefilter=False, order=0
            )
            if image.shape[2] == 4:  # image is RGBA
                colormapped = np.copy(downsampled)
                colormapped[..., 3] = downsampled[..., 3] * self.opacity
                if downsampled.dtype == np.uint8:
                    colormapped = colormapped.astype(np.uint8)
            else:  # image is RGB
                if downsampled.dtype == np.uint8:
                    alpha = np.full(
                        downsampled.shape[:2] + (1,),
                        int(255 * self.opacity),
                        dtype=np.uint8,
                    )
                else:
                    alpha = np.full(downsampled.shape[:2] + (1,), self.opacity)
                colormapped = np.concatenate([downsampled, alpha], axis=2)
        else:
            downsampled = ndi.zoom(
                image, zoom_factor, prefilter=False, order=0
            )
            low, high = self.contrast_limits
            downsampled = np.clip(downsampled, low, high)
            color_range = high - low
            if color_range != 0:
                downsampled = (downsampled - low) / color_range
            downsampled = downsampled**self.gamma
            color_array = self.colormap.map(downsampled.ravel())
            colormapped = color_array.reshape((*downsampled.shape, 4))
            colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

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
        shape = raw.shape[:-1] if self.rgb else raw.shape

        if self.ndim < len(coord):
            # handle 3D views of 2D data by omitting extra coordinate
            offset = len(coord) - len(shape)
            coord = coord[[d + offset for d in self._slice_input.displayed]]
        else:
            coord = coord[self._slice_input.displayed]

        if all(0 <= c < s for c, s in zip(coord, shape)):
            value = raw[tuple(coord)]
        else:
            value = None

        if self.multiscale:
            value = (self.data_level, value)

        return value

    def _get_offset_data_position(self, position: npt.NDArray) -> npt.NDArray:
        """Adjust position for offset between viewer and data coordinates.

        VisPy considers the coordinate system origin to be the canvas corner,
        while napari considers the origin to be the **center** of the corner
        pixel. To get the correct value under the mouse cursor, we need to
        shift the position by 0.5 pixels on each axis.
        """
        return position + 0.5

    def _display_bounding_box_at_level(
        self, dims_displayed: List[int], data_level: int
    ) -> npt.NDArray:
        """An axis aligned (ndisplay, 2) bounding box around the data at a given level"""
        shape = self.level_shapes[data_level]
        extent_at_level = np.vstack([np.zeros(len(shape)), shape - 1])
        return extent_at_level[:, dims_displayed].T

    def _display_bounding_box_augmented_data_level(
        self, dims_displayed: List[int]
    ) -> npt.NDArray:
        """An augmented, axis-aligned (ndisplay, 2) bounding box.
        If the layer is multiscale layer, then returns the
        bounding box of the data at the current level
        """
        return self._extent_level_data_augmented[:, dims_displayed].T


class Image(_ImageBase):
    _projectionclass = ImageProjectionMode

    @property
    def rendering(self):
        """Return current rendering mode.

        Selects a preset rendering mode in vispy that determines how
        volume is displayed.  Options include:

        * ``translucent``: voxel colors are blended along the view ray until
            the result is opaque.
        * ``mip``: maximum intensity projection. Cast a ray and display the
            maximum value that was encountered.
        * ``minip``: minimum intensity projection. Cast a ray and display the
            minimum value that was encountered.
        * ``attenuated_mip``: attenuated maximum intensity projection. Cast a
            ray and attenuate values based on integral of encountered values,
            display the maximum value that was encountered after attenuation.
            This will make nearer objects appear more prominent.
        * ``additive``: voxel colors are added along the view ray until
            the result is saturated.
        * ``iso``: isosurface. Cast a ray until a certain threshold is
            encountered. At that location, lighning calculations are
            performed to give the visual appearance of a surface.
        * ``average``: average intensity projection. Cast a ray and display the
            average of values that were encountered.

        Returns
        -------
        str
            The current rendering mode
        """
        return str(self._rendering)

    @rendering.setter
    def rendering(self, rendering):
        self._rendering = ImageRendering(rendering)
        self.events.rendering()

    def _get_state(self):
        """Get dictionary of layer state.

        Returns
        -------
        state : dict
            Dictionary of layer state.
        """
        state = self._get_base_state()
        state.update(
            {
                'rgb': self.rgb,
                'multiscale': self.multiscale,
                'colormap': self.colormap.dict(),
                'contrast_limits': self.contrast_limits,
                'interpolation2d': self.interpolation2d,
                'interpolation3d': self.interpolation3d,
                'rendering': self.rendering,
                'depiction': self.depiction,
                'plane': self.plane.dict(),
                'iso_threshold': self.iso_threshold,
                'attenuation': self.attenuation,
                'gamma': self.gamma,
                'data': self.data,
                'custom_interpolation_kernel_2d': self.custom_interpolation_kernel_2d,
            }
        )
        return state


Image.__doc__ = _ImageBase.__doc__
