"""Contains napari color constants and utilities."""

from collections.abc import Callable, Iterator
from typing import Union

import numpy as np

from napari.utils.colormaps.standardize_color import transform_color

ColorValueParam = Union[np.ndarray, list, tuple, str, None]
ColorArrayParam = Union[np.ndarray, list, tuple, None]


class ColorValue(np.ndarray):
    """A custom pydantic field type for storing one color value.

    Using this as a field type in a pydantic model means that validation
    of that field (e.g. on initialization or setting) will automatically
    use the ``validate`` method to coerce a value to a single color.
    """

    def __new__(cls, value: ColorValueParam) -> 'ColorValue':
        return cls.validate(value)

    @classmethod
    def __get_validators__(
        cls,
    ) -> Iterator[Callable[[ColorValueParam], 'ColorValue']]:
        yield cls.validate

    @classmethod
    def validate(cls, value: ColorValueParam) -> 'ColorValue':
        """Validates and coerces the given value into an array storing one color.

        Parameters
        ----------
        value : Union[np.ndarray, list, tuple, str, None]
            A supported single color value, which must be one of the following.

            - A supported RGB(A) sequence of floating point values in [0, 1].
            - A CSS3 color name: https://www.w3.org/TR/css-color-3/#svg-color
            - A single character matplotlib color name: https://matplotlib.org/stable/tutorials/colors/colors.html#specifying-colors
            - An RGB(A) hex code string.

        Returns
        -------
        np.ndarray
            An RGBA color vector of floating point values in [0, 1].

        Raises
        ------
        ValueError, AttributeError, KeyError
            If the value is not recognized as a color.

        Examples
        --------
        Coerce an RGBA array-like.

        >>> ColorValue.validate([1, 0, 0, 1])
        array([1., 0., 0., 1.], dtype=float32)

        Coerce a CSS3 color name.

        >>> ColorValue.validate('red')
        array([1., 0., 0., 1.], dtype=float32)

        Coerce a matplotlib single character color name.

        >>> ColorValue.validate('r')
        array([1., 0., 0., 1.], dtype=float32)

        Coerce an RGB hex-code.

        >>> ColorValue.validate('#ff0000')
        array([1., 0., 0., 1.], dtype=float32)
        """
        return transform_color(value)[0].view(cls)


class ColorArray(np.ndarray):
    """A custom pydantic field type for storing an array of color values.

    Using this as a field type in a pydantic model means that validation
    of that field (e.g. on initialization or setting) will automatically
    use the ``validate`` method to coerce a value to an array of colors.
    """

    def __new__(cls, value: ColorArrayParam) -> 'ColorArray':
        return cls.validate(value)

    @classmethod
    def __get_validators__(
        cls,
    ) -> Iterator[Callable[[ColorArrayParam], 'ColorArray']]:
        yield cls.validate

    def __sizeof__(self) -> int:
        return super().__sizeof__() + self.nbytes

    @classmethod
    def validate(cls, value: ColorArrayParam) -> 'ColorArray':
        """Validates and coerces the given value into an array storing many colors.

        Parameters
        ----------
        value : Union[np.ndarray, list, tuple, None]
            A supported sequence of single color values.
            See ``ColorValue.validate`` for valid single color values.
            In general each value should be of the same type, so avoid
            passing values like ``['red', [0, 0, 1]]``.

        Returns
        -------
        np.ndarray
            An array of N colors where each row is an RGBA color vector with
            floating point values in [0, 1].

        Raises
        ------
        ValueError, AttributeError, KeyError
            If the value is not recognized as an array of colors.

        Examples
        --------
        Coerce a list of CSS3 color names.

        >>> ColorArray.validate(['red', 'blue'])
        array([[1., 0., 0., 1.],
               [0., 0., 1., 1.]], dtype=float32)

        Coerce a tuple of matplotlib single character color names.

        >>> ColorArray.validate(('r', 'b'))
        array([[1., 0., 0., 1.],
               [0., 0., 1., 1.]], dtype=float32)
        """
        # Special case an empty supported sequence because transform_color
        # warns and returns an array containing a default color in that case.
        if isinstance(value, (np.ndarray, list, tuple)) and len(value) == 0:
            return np.empty((0, 4), np.float32).view(cls)
        return transform_color(value).view(cls)
