from typing import (
    Any,
    Literal,
    Protocol,
    Union,
    runtime_checkable,
)

import numpy as np

from napari._pydantic_compat import Field, parse_obj_as, validator
from napari.layers.utils.color_transformations import ColorType
from napari.layers.utils.style_encoding import (
    StyleEncoding,
    _ConstantStyleEncoding,
    _DerivedStyleEncoding,
    _ManualStyleEncoding,
)
from napari.utils import Colormap
from napari.utils.color import ColorArray, ColorValue
from napari.utils.colormaps import ValidColormapArg, ensure_colormap
from napari.utils.colormaps.categorical_colormap import CategoricalColormap
from napari.utils.translations import trans

"""The default color to use, which may also be used a safe fallback color."""
DEFAULT_COLOR = ColorValue.validate('cyan')


@runtime_checkable
class ColorEncoding(StyleEncoding[ColorValue, ColorArray], Protocol):
    """Encodes colors from features."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(
        cls, value: Union['ColorEncoding', dict, str, ColorType]
    ) -> 'ColorEncoding':
        """Validates and coerces a value to a ColorEncoding.

        Parameters
        ----------
        value : ColorEncodingArgument
            The value to validate and coerce.
            If this is already a ColorEncoding, it is returned as is.
            If this is a dict, then it should represent one of the built-in color encodings.
            If this a string, then a DirectColorEncoding is returned.
            If this a single color, a ConstantColorEncoding is returned.
            If this is a sequence of colors, a ManualColorEncoding is returned.

        Returns
        -------
        ColorEncoding

        Raises
        ------
        TypeError
            If the value is not a supported type.
        ValidationError
            If the value cannot be parsed into a ColorEncoding.
        """
        if isinstance(value, ColorEncoding):
            return value
        if isinstance(value, dict):
            return parse_obj_as(
                Union[
                    ConstantColorEncoding,
                    ManualColorEncoding,
                    DirectColorEncoding,
                    NominalColorEncoding,
                    QuantitativeColorEncoding,
                ],
                value,
            )
        try:
            color_array = ColorArray.validate(value)
        except (ValueError, AttributeError, KeyError) as e:
            raise TypeError(
                trans._(
                    'value should be a ColorEncoding, a dict, a color, or a sequence of colors',
                    deferred=True,
                )
            ) from e
        if color_array.shape[0] == 1:
            return ConstantColorEncoding(constant=value)
        return ManualColorEncoding(array=color_array, default=DEFAULT_COLOR)


class ConstantColorEncoding(_ConstantStyleEncoding[ColorValue, ColorArray]):
    """Encodes color values from a single constant color.

    Attributes
    ----------
    constant : ColorValue
        The constant color RGBA value.
    """

    encoding_type: Literal['ConstantColorEncoding'] = 'ConstantColorEncoding'
    constant: ColorValue


class ManualColorEncoding(_ManualStyleEncoding[ColorValue, ColorArray]):
    """Encodes color values manually in an array attribute.

    Attributes
    ----------
    array : ColorArray
        The array of color values. Can be written to directly to make
        persistent updates.
    default : ColorValue
        The default color value.
    """

    encoding_type: Literal['ManualColorEncoding'] = 'ManualColorEncoding'
    array: ColorArray
    default: ColorValue = Field(default_factory=lambda: DEFAULT_COLOR)


class DirectColorEncoding(_DerivedStyleEncoding[ColorValue, ColorArray]):
    """Encodes color values directly from a feature column.

    Attributes
    ----------
    feature : str
        The name of the feature that contains the desired color values.
    fallback : ColorArray
        The safe constant fallback color to use if the feature column
        does not contain valid color values.
    """

    encoding_type: Literal['DirectColorEncoding'] = 'DirectColorEncoding'
    feature: str
    fallback: ColorValue = Field(default_factory=lambda: DEFAULT_COLOR)

    def __call__(self, features: Any) -> ColorArray:
        # A column-like may be a series or have an object dtype (e.g. color names),
        # neither of which transform_color handles, so convert to a list.
        return ColorArray.validate(list(features[self.feature]))


class NominalColorEncoding(_DerivedStyleEncoding[ColorValue, ColorArray]):
    """Encodes color values from a nominal feature whose values are mapped to colors.

    Attributes
    ----------
    feature : str
        The name of the feature that contains the nominal values to be mapped to colors.
    colormap : CategoricalColormap
        Maps the feature values to colors.
    fallback : ColorValue
        The safe constant fallback color to use if mapping the feature values to
        colors fails.
    """

    encoding_type: Literal['NominalColorEncoding'] = 'NominalColorEncoding'
    feature: str
    colormap: CategoricalColormap
    fallback: ColorValue = Field(default_factory=lambda: DEFAULT_COLOR)

    def __call__(self, features: Any) -> ColorArray:
        # map is not expecting some column-likes (e.g. pandas.Series), so ensure
        # this is a numpy array first.
        values = np.asarray(features[self.feature])
        return self.colormap.map(values)


class QuantitativeColorEncoding(_DerivedStyleEncoding[ColorValue, ColorArray]):
    """Encodes color values from a quantitative feature whose values are mapped to colors.

    Attributes
    ----------
    feature : str
        The name of the feature that contains the nominal values to be mapped to colors.
    colormap : Colormap
        Maps feature values to colors.
    contrast_limits : Optional[Tuple[float, float]]
        The (min, max) feature values that should respectively map to the first and last
        colors in the colormap. If None, then this will attempt to calculate these values
        from the feature values each time this generates color values. If that attempt
        fails, these are effectively (0, 1).
    fallback : ColorValue
        The safe constant fallback color to use if mapping the feature values to
        colors fails.
    """

    encoding_type: Literal['QuantitativeColorEncoding'] = (
        'QuantitativeColorEncoding'
    )
    feature: str
    colormap: Colormap
    contrast_limits: tuple[float, float] | None = None
    fallback: ColorValue = Field(default_factory=lambda: DEFAULT_COLOR)

    def __call__(self, features: Any) -> ColorArray:
        values = features[self.feature]
        contrast_limits = self.contrast_limits or _calculate_contrast_limits(
            values
        )
        if contrast_limits is not None:
            values = np.interp(values, contrast_limits, (0, 1))
        return self.colormap.map(values)

    @validator('colormap', pre=True, always=True, allow_reuse=True)
    def _check_colormap(cls, colormap: ValidColormapArg) -> Colormap:
        return ensure_colormap(colormap)

    @validator('contrast_limits', pre=True, always=True, allow_reuse=True)
    def _check_contrast_limits(
        cls, contrast_limits
    ) -> tuple[float, float] | None:
        if (contrast_limits is not None) and (
            contrast_limits[0] >= contrast_limits[1]
        ):
            raise ValueError(
                trans._(
                    'contrast_limits must be a strictly increasing pair of values',
                    deferred=True,
                )
            )
        return contrast_limits


def _calculate_contrast_limits(
    values: np.ndarray,
) -> tuple[float, float] | None:
    contrast_limits = None
    if values.size > 0:
        min_value = np.min(values)
        max_value = np.max(values)
        # Use < instead of != to handle nans.
        if min_value < max_value:
            contrast_limits = (min_value, max_value)
    return contrast_limits
