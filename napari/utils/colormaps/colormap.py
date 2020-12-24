from enum import Enum
from functools import partial

import numpy as np
from pydantic import BaseModel, Field, root_validator, validator

from ..pydantic import Array, PydanticConfig, evented_model
from .colorbars import make_colorbar
from .standardize_color import transform_color


class ColormapInterpolationMode(str, Enum):
    """INTERPOLATION: Interpolation mode for colormaps.

    Selects an interpolation mode for the colormap.
            * linear: colors are defined by linear interpolation between
              colors of neighboring controls points.
            * zero: colors are defined by the value of the color in the
              bin between by neighboring controls points.
    """

    LINEAR = 'linear'
    ZERO = 'zero'


@evented_model
class Colormap(BaseModel):
    """Colormap that relates intensity values to colors.

    Attributes
    ----------
    colors : array, shape (N, 4)
        Data used in the colormap.
    name : str
        Name of the colormap.
    controls : array, shape (N,) or (N+1,)
        Control points of the colormap.
    interpolation : str
        Colormap interpolation mode, either 'linear' or
        'zero'. If 'linear', ncontrols = ncolors (one
        color per control point). If 'zero', ncontrols
        = ncolors+1 (one color per bin).
    """

    # fields
    colors: Array[float, (-1, 4)]
    name: str = 'custom'
    controls: Array[float, (-1,)] = Field(
        default_factory=partial(np.zeros, (0,))
    )
    interpolation: ColormapInterpolationMode = ColormapInterpolationMode.LINEAR

    # Config
    Config = PydanticConfig

    # validators
    _ensure_color_array = validator('colors', pre=True, allow_reuse=True)(
        transform_color
    )

    @root_validator
    def _check_controls(cls, values):
        if len(values['controls']) == 0:
            n_controls = len(values['colors']) + int(
                values['interpolation'] == ColormapInterpolationMode.ZERO
            )
            values['controls'] = np.linspace(0, 1, n_controls)
        return values

    def __iter__(self):
        yield from (self.colors, self.controls, self.interpolation)

    def map(self, values):
        values = np.atleast_1d(values)
        if self.interpolation == ColormapInterpolationMode.LINEAR:
            # One color per control point
            cols = [
                np.interp(values, self.controls, self.colors[:, i])
                for i in range(4)
            ]
            cols = np.stack(cols, axis=1)
        elif self.interpolation == ColormapInterpolationMode.ZERO:
            # One color per bin
            indices = np.clip(
                np.searchsorted(self.controls, values) - 1, 0, len(self.colors)
            )
            cols = self.colors[indices.astype(np.int32)]
        else:
            raise ValueError('Unrecognized Colormap Interpolation Mode')

        return cols

    @property
    def colorbar(self):
        return make_colorbar(self)
