import pytest
import numpy as np
from vispy.color import ColorArray

from napari.layers.util.color_transformations import (
    transform_color_with_defaults,
    normalize_and_broadcast_colors,
)


def test_transform_color_basic():
    """Test inner method with the same name."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    colorarray = transform_color_with_defaults(
        data=data, colors='r', elem_name='edge_color', default='black'
    )
    np.testing.assert_array_equal(colorarray, ColorArray('r').rgba)


def test_transform_color_wrong_colorname():
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    with pytest.warns(UserWarning):
        colorarray = transform_color_with_defaults(
            data=data, colors='rr', elem_name='edge_color', default='black'
        )
    np.testing.assert_array_equal(colorarray, ColorArray('black').rgba)


def test_transform_color_wrong_colorlen():
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    with pytest.warns(UserWarning):
        colorarray = transform_color_with_defaults(
            data=data,
            colors=['r', 'r'],
            elem_name='face_color',
            default='black',
        )
    np.testing.assert_array_equal(colorarray, ColorArray('black').rgba)


def test_normalize_colors_basic():
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    colors = ColorArray(['w'] * shape[0]).rgba
    colorarray = normalize_and_broadcast_colors(data, colors)
    np.testing.assert_array_equal(colorarray, colors)


def test_normalize_colors_wrong_num():
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    colors = ColorArray(['w'] * shape[0]).rgba
    with pytest.warns(UserWarning):
        colorarray = normalize_and_broadcast_colors(data, colors[:-1])
    np.testing.assert_array_equal(colorarray, colors)


def test_normalize_colors_zero_colors():
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    real = np.ones((shape[0], 4), dtype=np.float32)
    with pytest.warns(UserWarning):
        colorarray = normalize_and_broadcast_colors(data, [])
    np.testing.assert_array_equal(colorarray, real)
