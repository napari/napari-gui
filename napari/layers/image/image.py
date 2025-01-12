"""Image class."""

from __future__ import annotations

import typing
import warnings
from typing import Any, Literal, cast

import numpy as np
from scipy import ndimage as ndi

from napari.layers._data_protocols import LayerDataProtocol
from napari.layers._multiscale_data import MultiScaleData
from napari.layers._scalar_field.scalar_field import ScalarFieldBase
from napari.layers.image._image_constants import (
    ImageProjectionMode,
    ImageRendering,
    Interpolation,
    InterpolationStr,
)
from napari.layers.image._image_utils import guess_rgb
from napari.layers.image._slice import _ImageSliceResponse
from napari.layers.intensity_mixin import IntensityVisualizationMixin
from napari.layers.utils.layer_utils import calc_data_range
from napari.utils._dtype import get_dtype_limits, normalize_dtype
from napari.utils.colormaps import ensure_colormap
from napari.utils.colormaps.colormap_utils import _coerce_contrast_limits
from napari.utils.migrations import rename_argument
from napari.utils.translations import trans

__all__ = ('Image',)


class Image(IntensityVisualizationMixin, ScalarFieldBase):
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
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    attenuation : float
        Attenuation rate for attenuated maximum intensity projection.
    axis_labels : tuple of str
        Dimension names of the layer data.
        If not provided, axis_labels will be set to (..., 'axis -2', 'axis -1').
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'translucent', 'translucent_no_depth', 'additive', 'minimum', 'opaque'}.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    colormap : str, napari.utils.Colormap, tuple, dict
        Colormaps to use for luminance images. If a string, it can be the name
        of a supported colormap from vispy or matplotlib or the name of
        a vispy color or a hexadecimal RGB color representation.
        If a tuple, the first value must be a string to assign as a name to a
        colormap and the second item must be a Colormap. If a dict, the key must
        be a string to assign as a name to a colormap and the value must be a
        Colormap.
    contrast_limits : list (2,)
        Intensity value limits to be used for determining the minimum and maximum colormap bounds for
        luminance images. If not passed, they will be calculated as the min and max intensity value of
        the image.
    custom_interpolation_kernel_2d : np.ndarray
        Convolution kernel used with the 'custom' interpolation mode in 2D rendering.
    depiction : str
        3D Depiction mode. Must be one of {'volume', 'plane'}.
        The default value is 'volume'.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.
    gamma : float
        Gamma correction for determining colormap linearity; defaults to 1.
    interpolation2d : str
        Interpolation mode used by vispy for rendering 2d data.
        Must be one of our supported modes.
        (for list of supported modes see Interpolation enum)
        'custom' is a special mode for 2D interpolation in which a regular grid
        of samples is taken from the texture around a position using 'linear'
        interpolation before being multiplied with a custom interpolation kernel
        (provided with 'custom_interpolation_kernel_2d').
    interpolation3d : str
        Same as 'interpolation2d' but for 3D rendering.
    iso_threshold : float
        Threshold for isosurface.
    metadata : dict
        Layer metadata.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array-like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape,
        then it will be taken to be multiscale. The first image in the list
        should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    name : str
        Name of the layer.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    plane : dict or SlicingPlane
        Properties defining plane rendering in 3D. Properties are defined in
        data coordinates. Valid dictionary keys are
        {'position', 'normal', 'thickness', and 'enabled'}.
    projection_mode : str
        How data outside the viewed dimensions, but inside the thick Dims slice will
        be projected onto the viewed dimensions. Must fit to ImageProjectionMode
    rendering : str
        Rendering mode used by vispy. Must be one of our supported
        modes.
    rgb : bool, optional
        Whether the image is RGB or RGBA if rgb. If not
        specified by user, but the last dimension of the data has length 3 or 4,
        it will be set as `True`. If `False`, the image is interpreted as a
        luminance image.
    rotate : float, 3-tuple of float, or n-D array.
        If a float, convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple, convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise, assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        'np.degrees' if needed.
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
    interpolation2d : str
        Interpolation mode used by vispy. Must be one of our supported modes.
        'custom' is a special mode for 2D interpolation in which a regular grid
        of samples are taken from the texture around a position using 'linear'
        interpolation before being multiplied with a custom interpolation kernel
        (provided with 'custom_interpolation_kernel_2d').
    interpolation3d : str
        Same as 'interpolation2d' but for 3D rendering.
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
    units: tuple of pint.Unit
        Units of the layer data in world coordinates.
    Notes
    -----
    _data_view : array (N, M), (N, M, 3), or (N, M, 4)
        Image data for the currently viewed slice. Must be 2D image data, but
        can be multidimensional for RGB or RGBA images if multidimensional is
        `True`.
    """

    _projectionclass = ImageProjectionMode

    @rename_argument(
        from_name='interpolation',
        to_name='interpolation2d',
        version='0.6.0',
        since_version='0.4.17',
    )
    def __init__(
        self,
        data,
        *,
        affine=None,
        attenuation=0.05,
        axis_labels=None,
        blending='translucent',
        cache=True,
        colormap='gray',
        contrast_limits=None,
        custom_interpolation_kernel_2d=None,
        depiction='volume',
        experimental_clipping_planes=None,
        gamma=1.0,
        interpolation2d='nearest',
        interpolation3d='linear',
        iso_threshold=None,
        metadata=None,
        multiscale=None,
        name=None,
        opacity=1.0,
        plane=None,
        projection_mode='none',
        rendering='mip',
        rgb=None,
        rotate=None,
        scale=None,
        shear=None,
        translate=None,
        units=None,
        visible=True,
    ):
        # Determine if rgb
        data_shape = data.shape if hasattr(data, 'shape') else data[0].shape
        if rgb and not guess_rgb(data_shape, min_side_len=0):
            raise ValueError(
                trans._(
                    "'rgb' was set to True but data does not have suitable dimensions."
                )
            )
        if rgb is None:
            rgb = guess_rgb(data_shape)

        self.rgb = rgb
        super().__init__(
            data,
            affine=affine,
            axis_labels=axis_labels,
            blending=blending,
            cache=cache,
            custom_interpolation_kernel_2d=custom_interpolation_kernel_2d,
            depiction=depiction,
            experimental_clipping_planes=experimental_clipping_planes,
            metadata=metadata,
            multiscale=multiscale,
            name=name,
            ndim=len(data_shape) - 1 if rgb else len(data_shape),
            opacity=opacity,
            plane=plane,
            projection_mode=projection_mode,
            rendering=rendering,
            rotate=rotate,
            scale=scale,
            shear=shear,
            translate=translate,
            units=units,
            visible=visible,
        )

        self.rgb = rgb
        self._colormap = ensure_colormap(colormap)
        self._gamma = gamma
        self._interpolation2d = Interpolation.NEAREST
        self._interpolation3d = Interpolation.NEAREST
        self.interpolation2d = interpolation2d
        self.interpolation3d = interpolation3d
        self._attenuation = attenuation

        # Set contrast limits, colormaps and plane parameters
        if contrast_limits is None:
            if not isinstance(data, np.ndarray):
                dtype = normalize_dtype(getattr(data, 'dtype', None))
                if np.issubdtype(dtype, np.integer):
                    self.contrast_limits_range = get_dtype_limits(dtype)
                else:
                    self.contrast_limits_range = (0, 1)
                self._should_calc_clims = dtype != np.uint8
            else:
                self.contrast_limits_range = self._calc_data_range()
        else:
            self.contrast_limits_range = contrast_limits
        self._contrast_limits: tuple[float, float] = self.contrast_limits_range
        self.contrast_limits = self._contrast_limits

        if iso_threshold is None:
            cmin, cmax = self.contrast_limits_range
            self._iso_threshold = cmin + (cmax - cmin) / 2
        else:
            self._iso_threshold = iso_threshold

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

    def _get_state(self) -> dict[str, Any]:
        """Get dictionary of layer state.

        Returns
        -------
        state : dict of str to Any
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

    def _update_slice_response(self, response: _ImageSliceResponse) -> None:
        if self._keep_auto_contrast:
            data = response.image.raw
            input_data = data[-1] if self.multiscale else data
            self.contrast_limits = calc_data_range(
                typing.cast(LayerDataProtocol, input_data), rgb=self.rgb
            )

        super()._update_slice_response(response)

        # Maybe reset the contrast limits based on the new slice.
        if self._should_calc_clims:
            self.reset_contrast_limits_range()
            self.reset_contrast_limits()
            self._should_calc_clims = False
        elif self._keep_auto_contrast:
            self.reset_contrast_limits()

    @property
    def attenuation(self) -> float:
        """float: attenuation rate for attenuated_mip rendering."""
        return self._attenuation

    @attenuation.setter
    def attenuation(self, value: float) -> None:
        self._attenuation = value
        self._update_thumbnail()
        self.events.attenuation()

    @property
    def data(self) -> LayerDataProtocol | MultiScaleData:
        """Data, possibly in multiscale wrapper. Obeys LayerDataProtocol."""
        return self._data

    @data.setter
    def data(self, data: LayerDataProtocol | MultiScaleData) -> None:
        self._data_raw = data
        # note, we don't support changing multiscale in an Image instance
        self._data = MultiScaleData(data) if self.multiscale else data  # type: ignore
        self._update_dims()
        if self._keep_auto_contrast:
            self.reset_contrast_limits()
        self.events.data(value=self.data)
        self._reset_editable()

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
                'Interpolation attribute is deprecated since 0.4.17. Please use interpolation2d or interpolation3d',
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
                'Interpolation setting is deprecated since 0.4.17. Please use interpolation2d or interpolation3d',
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
    def interpolation2d(self, value: InterpolationStr | Interpolation) -> None:
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
    def interpolation3d(self, value: InterpolationStr | Interpolation) -> None:
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
    def iso_threshold(self) -> float:
        """float: threshold for isosurface."""
        return self._iso_threshold

    @iso_threshold.setter
    def iso_threshold(self, value: float) -> None:
        self._iso_threshold = value
        self._update_thumbnail()
        self.events.iso_threshold()

    def _get_level_shapes(self):
        shapes = super()._get_level_shapes()
        if self.rgb:
            shapes = [s[:-1] for s in shapes]
        return shapes

    def _update_thumbnail(self):
        """Update thumbnail with current image data and colormap."""
        # don't bother updating thumbnail if we don't have any data
        # this also avoids possible dtype mismatch issues below
        # for example np.clip may raise an OverflowError (in numpy 2.0)
        if self._slice.empty:
            return

        image = self._slice.thumbnail.raw

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
            if np.issubdtype(downsampled.dtype, np.integer):
                low = max(low, np.iinfo(downsampled.dtype).min)
                high = min(high, np.iinfo(downsampled.dtype).max)
            downsampled = np.clip(downsampled, low, high)
            color_range = high - low
            if color_range != 0:
                downsampled = (downsampled - low) / color_range
            downsampled = downsampled**self.gamma
            color_array = self.colormap.map(downsampled.ravel())
            colormapped = color_array.reshape((*downsampled.shape, 4))
            colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    def _calc_data_range(
        self, mode: Literal['data', 'slice'] = 'data'
    ) -> tuple[float, float]:
        """
        Calculate the range of the data values in the currently viewed slice
        or full data array
        """
        if mode == 'data':
            input_data = self.data[-1] if self.multiscale else self.data
        elif mode == 'slice':
            data = self._slice.image.raw  # ugh
            input_data = data[-1] if self.multiscale else data
        else:
            raise ValueError(
                trans._(
                    "mode must be either 'data' or 'slice', got {mode!r}",
                    deferred=True,
                    mode=mode,
                )
            )
        return calc_data_range(
            cast(LayerDataProtocol, input_data), rgb=self.rgb
        )

    def _raw_to_displayed(self, raw: np.ndarray) -> np.ndarray:
        """Determine displayed image from raw image.

        This function checks if current contrast_limits are within the range
        supported by vispy.
        If yes, it returns the raw image.
        If not, it rescales the raw image to fit within
        the range supported by vispy.

        Parameters
        ----------
        raw : array
            Raw array.

        Returns
        -------
        image : array
            Displayed array.
        """
        fixed_contrast_info = _coerce_contrast_limits(self.contrast_limits)
        if np.allclose(
            fixed_contrast_info.contrast_limits, self.contrast_limits
        ):
            return raw

        return fixed_contrast_info.coerce_data(raw)

    @IntensityVisualizationMixin.contrast_limits.setter  # type: ignore [attr-defined]
    def contrast_limits(self, contrast_limits):
        IntensityVisualizationMixin.contrast_limits.fset(self, contrast_limits)
        if not np.allclose(
            _coerce_contrast_limits(self.contrast_limits).contrast_limits,
            self.contrast_limits,
        ):
            prev = self._keep_auto_contrast
            self._keep_auto_contrast = False
            try:
                self.refresh(highlight=False, extent=False)
            finally:
                self._keep_auto_contrast = prev

    def _calculate_value_from_ray(self, values):
        # translucent is special: just return the first value, no matter what
        if self.rendering == ImageRendering.TRANSLUCENT:
            return np.ravel(values)[0]
        # iso is weird too: just return None always
        if self.rendering == ImageRendering.ISO:
            return None

        # if the whole ray is NaN, we should see nothing, so return None
        # this check saves us some warnings later as well, so better do it now
        if np.all(np.isnan(values)):
            return None

        # "summary" renderings; they do not represent a specific pixel, so we just
        # return the summary value. We should probably differentiate these somehow.
        # these are also probably not the same as how the gpu does it...
        if self.rendering == ImageRendering.AVERAGE:
            return np.nanmean(values)
        if self.rendering == ImageRendering.ADDITIVE:
            # TODO: this is "broken" cause same pixel gets multisampled...
            #       but it looks like it's also overdoing it in vispy vis too?
            #       I don't know if there's a way to *not* do it...
            return np.nansum(values)

        # all the following cases are returning the *actual* value of the image at the
        # "selected" pixel, whose position changes depending on the rendering mode.
        if self.rendering == ImageRendering.MIP:
            return np.nanmax(values)
        if self.rendering == ImageRendering.MINIP:
            return np.nanmin(values)
        if self.rendering == ImageRendering.ATTENUATED_MIP:
            # normalize values so attenuation applies from 0 to 1
            values_attenuated = (
                values - self.contrast_limits[0]
            ) / self.contrast_limits[1]
            # approx, step size is actually calculated with int(lenght(ray) * 2)
            step_size = 0.5
            sumval = (
                step_size
                * np.cumsum(np.clip(values_attenuated, 0, 1))
                * len(values_attenuated)
            )
            scale = np.exp(-self.attenuation * (sumval - 1))
            return values[np.nanargmin(values_attenuated * scale)]

        raise RuntimeError(  # pragma: no cover
            f'ray value calculation not implemented for {self.rendering}'
        )
