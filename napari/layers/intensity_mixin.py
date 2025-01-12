from typing import TYPE_CHECKING

import numpy as np

from napari.utils._dtype import normalize_dtype
from napari.utils.colormaps import ensure_colormap
from napari.utils.events import Event
from napari.utils.status_messages import format_float
from napari.utils.validators import _validate_increasing, validate_n_seq

validate_2_tuple = validate_n_seq(2)

if TYPE_CHECKING:
    from napari.layers._scalar_field.scalar_field import ScalarFieldBase


class IntensityVisualizationMixin:
    """A mixin that adds gamma, colormap, and contrast limits logic to Layers.

    When used, this should come before the Layer in the inheritance, e.g.:

        class Image(IntensityVisualizationMixin, Layer):
            def __init__(self):
                ...

    Note: `contrast_limits_range` is range extent available on the widget,
    and `contrast_limits` is the visible range (the set values on the widget)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.events.add(
            contrast_limits=Event,
            contrast_limits_range=Event,
            gamma=Event,
            colormap=Event,
        )
        self._gamma = 1
        self._colormap_name = ''
        self._contrast_limits_msg = ''
        self._contrast_limits: tuple[float | None, float | None] = (
            None,
            None,
        )
        self._contrast_limits_range: tuple[float | None, float | None] = (
            None,
            None,
        )
        self._auto_contrast_source = 'slice'
        self._keep_auto_contrast = False

    def reset_contrast_limits(self: 'ScalarFieldBase', mode=None):
        """Scale contrast limits to data range"""
        mode = mode or self._auto_contrast_source
        self.contrast_limits = self._calc_data_range(mode)

    def _calc_data_range(self, mode):
        raise NotImplementedError

    def reset_contrast_limits_range(self, mode=None):
        """Scale contrast limits range to data type if dtype is an integer,
        or use the current maximum data range otherwise.
        """
        dtype = normalize_dtype(self.dtype)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            self.contrast_limits_range = (info.min, info.max)
        else:
            mode = mode or self._auto_contrast_source
            self.contrast_limits_range = self._calc_data_range(mode)

    @property
    def colormap(self):
        """napari.utils.Colormap: colormap for luminance images."""
        return self._colormap

    @colormap.setter
    def colormap(self, colormap):
        self._set_colormap(colormap)

    def _set_colormap(self, colormap):
        self._colormap = ensure_colormap(colormap)
        self._update_thumbnail()
        self.events.colormap()

    @property
    def colormaps(self):
        """tuple of str: names of available colormaps."""
        return tuple(self._colormaps.keys())

    @property
    def contrast_limits(self):
        """list of float: Limits to use for the colormap."""
        return list(self._contrast_limits)

    @contrast_limits.setter
    def contrast_limits(self, contrast_limits):
        validate_2_tuple(contrast_limits)
        _validate_increasing(contrast_limits)
        self._contrast_limits_msg = (
            format_float(contrast_limits[0])
            + ', '
            + format_float(contrast_limits[1])
        )
        self._contrast_limits = contrast_limits
        # make sure range slider is big enough to fit range
        newrange = list(self.contrast_limits_range)
        newrange[0] = min(newrange[0], contrast_limits[0])
        newrange[1] = max(newrange[1], contrast_limits[1])
        self.contrast_limits_range = newrange
        self._update_thumbnail()
        self.events.contrast_limits()

    @property
    def contrast_limits_range(self):
        """The current valid range of the contrast limits."""
        return list(self._contrast_limits_range)

    @contrast_limits_range.setter
    def contrast_limits_range(self, value):
        """Set the valid range of the contrast limits.
        If either value is "None", the current range will be preserved.
        If the range overlaps the current contrast limits, the range will be set
        requested and there will be no change the contrast limits.
        If the requested contrast range limits are completely outside the
        current contrast limits, the range will be set as requested and the
        contrast limits will be reset to the new range.
        """
        validate_2_tuple(value)
        _validate_increasing(value)
        if list(value) == self.contrast_limits_range:
            return

        # if either value is "None", it just preserves the current range
        current_range = self.contrast_limits_range
        value = list(value)  # make sure it is mutable
        for i in range(2):
            value[i] = current_range[i] if value[i] is None else value[i]
        self._contrast_limits_range = value
        self.events.contrast_limits_range()

        # make sure that the contrast limits fit within the new range
        # this also serves the purpose of emitting events.contrast_limits()
        # and updating the views/controllers
        if hasattr(self, '_contrast_limits') and any(self._contrast_limits):
            clipped_limits = np.clip(self.contrast_limits, *value)
            if clipped_limits[0] < clipped_limits[1]:
                self.contrast_limits = tuple(clipped_limits)
            else:
                self.contrast_limits = tuple(value)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = float(value)
        self._update_thumbnail()
        self.events.gamma()
