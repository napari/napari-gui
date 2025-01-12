"""This module contains functions that 'standardize' the color handling
of napari layers by supplying functions that are able to convert most
color representation the user had in mind into a single representation -
a numpy Nx4 array of float32 values between 0 and 1 - that is used across
the codebase. The color is always in an RGBA format. To handle colors in
HSV, for example, we should point users to skimage, matplotlib and others.

The main function of the module is "transform_color", which might call
a cascade of other, private, function in the module to do the hard work
of converting the input. This function will either be called directly, or
used by the function "transform_color_with_defaults", which is a helper
function for the layer objects located in
``layers.utils.color_transformations.py``.

In general, when handling colors we try to catch invalid color
representations, warn the users of their misbehaving and return a default
white color array, since it seems unreasonable to crash the entire napari
session due to mis-represented colors.
"""

import functools
import types
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from vispy.color import ColorArray, get_color_dict, get_color_names
from vispy.color.color_array import _string_to_rgb

from napari.utils.translations import trans


def transform_color(colors: Any) -> np.ndarray:
    """Transforms provided color(s) to an Nx4 array of RGBA np.float32
    values.

    N is the number of given colors. The function is designed to parse all
    valid color representations a user might have and convert them properly.
    That being said, combinations of different color representation in the
    same list of colors is prohibited, and will error. This means that a list
    of ['red', np.array([1, 0, 0])] cannot be parsed and has to be manually
    pre-processed by the user before sent to this function. In addition, the
    provided colors - if numeric - should already be in an RGB(A) format. To
    convert an existing numeric color array to RGBA format use skimage before
    calling this function.

    Parameters
    ----------
    colors : string and array-like.
        The color(s) to interpret and convert

    Returns
    -------
    colors : np.ndarray
        An instance of np.ndarray with a data type of float32, 4 columns in
        RGBA order and N rows, with N being the number of colors. The array
        will always be 2D even if a single color is passed.

    Raises
    ------
    ValueError, AttributeError, KeyError
        invalid inputs
    """
    colortype = type(colors)
    for typ, handler in _color_switch.items():
        if issubclass(colortype, typ):
            return handler(colors)
    raise ValueError(f"cannot convert type '{colortype}' to a color array.")


@functools.lru_cache(maxsize=1024)
def _handle_str(color: str) -> np.ndarray:
    """Creates an array from a color of type string.

    The function uses an LRU cache to enhance performance.

    Parameters
    ----------
    color : str
        A single string as an input color. Can be a color name or a
        hex representation of a color, with either 6 or 8 hex digits.

    Returns
    -------
    colorarray : np.ndarray
        1x4 array

    """
    if len(color) == 0:
        warnings.warn(
            trans._(
                'Empty string detected. Returning black instead.',
                deferred=True,
            )
        )
        return np.zeros((1, 4), dtype=np.float32)

    colorarray = np.atleast_2d(_string_to_rgb(color)).astype(np.float32)
    if colorarray.shape[1] == 3:
        colorarray = np.column_stack([colorarray, np.float32(1.0)])
    return colorarray


def _handle_list_like(colors: Sequence) -> np.ndarray | None:
    """Parse a list-like container of colors into a numpy Nx4 array.

    Handles all list-like containers of colors using recursion (if necessary).
    The colors inside the container should all be represented in the same
    manner. This means that a list containing ['r', (1., 1., 1.)] will raise
    an error. Note that numpy arrays are handled in _handle_array. Lists which
    are known to contain strings will be parsed with _handle_str_list_like.
    Generators should first visit _handle_generator before arriving as input.

    Parameters
    ----------
    colors : Sequence
        A list-like container of colors. The colors inside should be homogeneous
        in their representation.

    Returns
    -------
    color_array : np.ndarray
        Nx4 numpy array, with N being the length of ``colors``.
    """

    try:
        # The following conversion works for most cases, and so it's expected
        # that most valid inputs will pass this .asarray() call
        # with ease. Those who don't are usually too cryptic to decipher.
        # If only some of the colors are strings, explicitly provide an object
        # dtype to avoid the deprecated behavior described in:
        # https://github.com/napari/napari/issues/2791
        num_str = len([c for c in colors if isinstance(c, str)])
        dtype = 'O' if 0 < num_str < len(colors) else None
        color_array = np.atleast_2d(np.asarray(colors, dtype=dtype))
    except ValueError:
        warnings.warn(
            trans._(
                "Couldn't convert input color array to a proper numpy array. Please make sure that your input data is in a parsable format. Converting input to a white color array.",
                deferred=True,
            )
        )
        return np.ones((max(len(colors), 1), 4), dtype=np.float32)

    # Happy path - converted to a float\integer array
    if color_array.dtype.kind in ['f', 'i']:
        return _handle_array(color_array)

    # User input was an iterable with strings
    if color_array.dtype.kind in ['U', 'O']:
        return _handle_str_list_like(color_array.ravel())
    return None


def _handle_generator(colors) -> np.ndarray | None:
    """Generators are converted to lists since we need to know their
    length to instantiate a proper array.
    """
    return _handle_list_like(list(colors))


def handle_nested_colors(colors) -> ColorArray:
    """In case of an array-like container holding colors, unpack it."""
    colors_as_rbga = np.ones((len(colors), 4), dtype=np.float32)
    for idx, color in enumerate(colors):
        colors_as_rbga[idx] = _color_switch[type(color)](color)
    return ColorArray(colors_as_rbga)


def _handle_array(colors: np.ndarray) -> np.ndarray:
    """Converts the given array into an array in the right format."""
    kind = colors.dtype.kind

    # Object arrays aren't handled by napari
    if kind == 'O':
        warnings.warn(
            trans._(
                'An object array was passed as the color input. Please convert its datatype before sending it to napari. Converting input to a white color array.',
                deferred=True,
            )
        )
        return np.ones((max(len(colors), 1), 4), dtype=np.float32)

    # An array of strings will be treated as a list if compatible
    if kind == 'U':
        if colors.ndim == 1:
            return _handle_str_list_like(colors)

        warnings.warn(
            trans._(
                'String color arrays should be one-dimensional. Converting input to a white color array.',
                deferred=True,
            )
        )
        return np.ones((len(colors), 4), dtype=np.float32)

    # Test the dimensionality of the input array

    # Empty color array can be a way for the user to signal
    # that it wants the "default" colors of napari. We return
    # a single white color.
    if colors.shape[-1] == 0:
        warnings.warn(
            trans._(
                'Given color input is empty. Converting input to a white color array.',
                deferred=True,
            )
        )
        return np.ones((1, 4), dtype=np.float32)

    colors = np.atleast_2d(colors)

    # Arrays with more than two dimensions don't have a clear
    # conversion method to a color array and thus raise an error.
    if colors.ndim > 2:
        raise ValueError(
            trans._(
                'Given colors input should contain one or two dimensions. Received array with {ndim} dimensions.',
                deferred=True,
                ndim=colors.ndim,
            )
        )

    # User provided a list of numbers as color input. This input
    # cannot be coerced into something understandable and thus
    # will return an error.
    if colors.shape[0] == 1 and colors.shape[1] not in {3, 4}:
        raise ValueError(
            trans._(
                'Given color array has an unsupported format. Received the following array:\n{colors}\nA proper color array should have 3-4 columns with a row per data entry.',
                deferred=True,
                colors=colors,
            )
        )

    # The user gave a list of colors, but it contains a wrong number
    # of columns. This check will also drop Nx1 (2D) arrays, since
    # numpy has vectors, and representing colors in this way
    # (column vector-like) is redundant. However, this results in a
    # warning and not a ValueError since we know the number of colors
    # in this dataset, meaning we can save the napari session by
    # rendering the data in white, which better than crashing.
    if not 3 <= colors.shape[1] <= 4:
        warnings.warn(
            trans._(
                'Given colors input should contain three or four columns. Received array with {shape} columns. Converting input to a white color array.',
                deferred=True,
                shape=colors.shape[1],
            )
        )
        return np.ones((len(colors), 4), dtype=np.float32)

    # Arrays with floats and ints can be safely converted to the proper format
    if kind in ['f', 'i', 'u']:
        return _convert_array_to_correct_format(colors)

    raise ValueError(
        trans._(
            'Data type of array ({color_dtype}) not supported.',
            deferred=True,
            color_dtype=colors.dtype,
        )
    )


def _convert_array_to_correct_format(colors: np.ndarray) -> np.ndarray:
    """Asserts shape, dtype and normalization of given color array.

    This function deals with arrays which are already 'well-behaved',
    i.e. have (almost) the correct number of columns and are able to represent
    colors correctly. It then it makes sure that the array indeed has exactly
    four columns and that its values are normalized between 0 and 1, with a
    data type of float32.

    Parameters
    ----------
    colors : np.ndarray
        Input color array, perhaps un-normalized and without the alpha channel.

    Returns
    -------
    colors : np.ndarray
        Nx4, float32 color array with values in the range [0, 1]
    """
    if colors.shape[1] == 3:
        colors = np.column_stack(
            [colors, np.ones(len(colors), dtype=np.float32)]
        )

    if colors.min() < 0:
        raise ValueError(
            trans._(
                'Colors input had negative values.',
                deferred=True,
            )
        )

    if colors.max() > 1:
        warnings.warn(
            trans._(
                "Colors with values larger than one detected. napari will normalize these colors for you. If you'd like to convert these yourself, please use the proper method from skimage.color.",
                deferred=True,
            )
        )
        colors = _normalize_color_array(colors)
    return np.atleast_2d(np.asarray(colors, dtype=np.float32))


def _handle_str_list_like(colors: Sequence | np.ndarray) -> np.ndarray:
    """Converts lists or arrays filled with strings to the proper color array
    format.

    Parameters
    ----------
    colors : list-like
        A sequence of string colors

    Returns
    -------
    color_array : np.ndarray
        Nx4, float32 color array
    """
    color_array = np.empty((len(colors), 4), dtype=np.float32)
    for idx, c in enumerate(colors):
        try:
            color_array[idx, :] = _color_switch[type(c)](c)
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(
                trans._(
                    'Invalid color found: {color} at index {idx}.',
                    deferred=True,
                    color=c,
                    idx=idx,
                )
            ) from e
    return color_array


def _handle_none(color) -> np.ndarray:
    """Converts color given as None to black.

    Parameters
    ----------
    color : NoneType
        None value given as a color

    Returns
    -------
    arr : np.ndarray
        1x4 numpy array of float32 zeros

    """
    return np.zeros((1, 4), dtype=np.float32)


def _normalize_color_array(colors: np.ndarray) -> np.ndarray:
    """Normalize all array values to the range [0, 1].

    The added complexity here stems from the fact that if a row in the given
    array contains four identical value a simple normalization might raise a
    division by zero exception.

    Parameters
    ----------
    colors : np.ndarray
        A numpy array with values possibly outside the range of [0, 1]

    Returns
    -------
    colors : np.ndarray
        Copy of input array with normalized values
    """
    colors = colors.astype(np.float32, copy=True)
    out_of_bounds_idx = np.unique(np.where((colors > 1) | (colors < 0))[0])
    out_of_bounds = colors[out_of_bounds_idx]
    norm = np.linalg.norm(out_of_bounds, np.inf, axis=1)
    out_of_bounds = out_of_bounds / norm[:, np.newaxis]
    colors[out_of_bounds_idx] = out_of_bounds
    return colors.astype(np.float32)


_color_switch: dict[Any, Callable] = {
    str: _handle_str,
    np.str_: _handle_str,
    list: _handle_list_like,
    tuple: _handle_list_like,
    types.GeneratorType: _handle_generator,
    np.ndarray: _handle_array,
    type(None): _handle_none,
}


def _create_hex_to_name_dict():
    """Create a dictionary mapping hexadecimal RGB colors into their
    'official' name.

    Returns
    -------
    hex_to_rgb : dict
        Mapping from hexadecimal RGB ('#ff0000') to name ('red').
    """
    colordict = get_color_dict()
    hex_to_name = {f'{v.lower()}ff': k for k, v in colordict.items()}
    return hex_to_name


def get_color_namelist():
    """Gets all the color names supported by napari.

    Returns
    -------
    list[str]
        All the color names supported by napari.
    """
    return get_color_names()


hex_to_name = _create_hex_to_name_dict()


def _check_color_dim(val):
    """Ensures input is Nx4.

    Parameters
    ----------
    val : np.ndarray
        A color array of possibly less than 4 columns

    Returns
    -------
    val : np.ndarray
        A four columns version of the input array. If the original array
        was a missing the fourth channel, it's added as 1.0 values.
    """
    val = np.atleast_2d(val)
    if val.shape[1] not in (3, 4):
        strval = str(val)
        if len(strval) > 100:
            strval = strval[:97] + '...'
        raise RuntimeError(
            trans._(
                'Value must have second dimension of size 3 or 4. Got `{val}`, shape={shape}',
                deferred=True,
                shape=val.shape,
                val=strval,
            )
        )

    if val.shape[1] == 3:
        val = np.column_stack([val, np.float32(1.0)])
    return val


def rgb_to_hex(rgbs: Sequence) -> np.ndarray:
    """Convert RGB to hex quadruplet.

    Taken from vispy with slight modifications.

    Parameters
    ----------
    rgbs : Sequence
        A list-like container of colors in RGBA format with values
        between [0, 1]

    Returns
    -------
    arr : np.ndarray
        An array of the hex representation of the input colors

    """
    rgbs = _check_color_dim(rgbs)
    return np.array(
        [
            f'#{"%02x" * 4}' % tuple((255 * rgb).astype(np.uint8))
            for rgb in rgbs
        ],
        '|U9',
    )
