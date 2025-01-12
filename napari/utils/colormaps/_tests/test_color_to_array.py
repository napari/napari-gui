import numpy as np
import pytest

from napari.utils.colormaps._tests.colors_data import (
    invalid_colors,
    single_color_options,
    single_colors_as_array,
    two_color_options,
    two_colors_as_array,
    warning_colors,
)
from napari.utils.colormaps.standardize_color import transform_color


@pytest.mark.parametrize(
    ('colors', 'true_colors'),
    zip(single_color_options, single_colors_as_array, strict=False),
)
def test_oned_points(colors, true_colors):
    np.testing.assert_array_equal(true_colors, transform_color(colors))


def test_warns_but_parses():
    """Test collection of colors that raise a warning but do not return
    a default white color array.
    """
    colors = ['', (43, 3, 3, 3), np.array([[3, 3, 3, 3], [0, 0, 0, 1]])]

    true_colors = [
        np.zeros((1, 4), dtype=np.float32),
        np.array([[1, 3 / 43, 3 / 43, 3 / 43]], dtype=np.float32),
        np.array(
            [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        ),
    ]
    with pytest.warns(UserWarning):
        for true, color in zip(true_colors, colors, strict=False):
            np.testing.assert_array_equal(true, transform_color(color))


@pytest.mark.parametrize(
    ('colors', 'true_colors'),
    zip(two_color_options, two_colors_as_array, strict=False),
)
def test_twod_points(colors, true_colors):
    np.testing.assert_array_equal(true_colors, transform_color(colors))


@pytest.mark.parametrize('color', invalid_colors)
def test_invalid_colors(color):
    with pytest.raises((ValueError, AttributeError, KeyError)):
        transform_color(color)


@pytest.mark.parametrize('colors', warning_colors)
def test_warning_colors(colors):
    with pytest.warns(UserWarning):
        np.testing.assert_array_equal(
            np.ones((max(len(colors), 1), 4), dtype=np.float32),
            transform_color(colors),
        )
