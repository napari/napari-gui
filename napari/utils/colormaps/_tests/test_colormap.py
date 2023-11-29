import importlib
from itertools import product
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest

from napari.utils.color import ColorArray
from napari.utils.colormaps import Colormap, DirectLabelColormap, colormap
from napari.utils.colormaps.colormap import (
    DEFAULT_VALUE,
)
from napari.utils.colormaps.colormap_utils import label_colormap


def test_linear_colormap():
    """Test a linear colormap."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap = Colormap(colors, name='testing')

    assert cmap.name == 'testing'
    assert cmap.interpolation == 'linear'
    assert len(cmap.controls) == len(colors)
    np.testing.assert_almost_equal(cmap.colors, colors)
    np.testing.assert_almost_equal(cmap.map([0.75]), [[0, 0.5, 0.5, 1]])


def test_linear_colormap_with_control_points():
    """Test a linear colormap with control points."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap = Colormap(colors, name='testing', controls=[0, 0.75, 1])

    assert cmap.name == 'testing'
    assert cmap.interpolation == 'linear'
    assert len(cmap.controls) == len(colors)
    np.testing.assert_almost_equal(cmap.colors, colors)
    np.testing.assert_almost_equal(cmap.map([0.75]), [[0, 1, 0, 1]])


def test_non_ascending_control_points():
    """Test non ascending control points raises an error."""
    colors = np.array(
        [[0, 0, 0, 1], [0, 0.5, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    )
    with pytest.raises(ValueError):
        Colormap(colors, name='testing', controls=[0, 0.75, 0.25, 1])


def test_wrong_number_control_points():
    """Test wrong number of control points raises an error."""
    colors = np.array(
        [[0, 0, 0, 1], [0, 0.5, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    )
    with pytest.raises(ValueError):
        Colormap(colors, name='testing', controls=[0, 0.75, 1])


def test_wrong_start_control_point():
    """Test wrong start of control points raises an error."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    with pytest.raises(ValueError):
        Colormap(colors, name='testing', controls=[0.1, 0.75, 1])


def test_wrong_end_control_point():
    """Test wrong end of control points raises an error."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    with pytest.raises(ValueError):
        Colormap(colors, name='testing', controls=[0, 0.75, 0.9])


def test_binned_colormap():
    """Test a binned colormap."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap = Colormap(colors, name='testing', interpolation='zero')

    assert cmap.name == 'testing'
    assert cmap.interpolation == 'zero'
    assert len(cmap.controls) == len(colors) + 1
    np.testing.assert_almost_equal(cmap.colors, colors)
    np.testing.assert_almost_equal(cmap.map([0.4]), [[0, 1, 0, 1]])


def test_binned_colormap_with_control_points():
    """Test a binned with control points."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap = Colormap(
        colors,
        name='testing',
        interpolation='zero',
        controls=[0, 0.2, 0.3, 1],
    )

    assert cmap.name == 'testing'
    assert cmap.interpolation == 'zero'
    assert len(cmap.controls) == len(colors) + 1
    np.testing.assert_almost_equal(cmap.colors, colors)
    np.testing.assert_almost_equal(cmap.map([0.4]), [[0, 0, 1, 1]])


def test_colormap_equality():
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap_1 = Colormap(colors, name='testing', controls=[0, 0.75, 1])
    cmap_2 = Colormap(colors, name='testing', controls=[0, 0.75, 1])
    cmap_3 = Colormap(colors, name='testing', controls=[0, 0.25, 1])
    assert cmap_1 == cmap_2
    assert cmap_1 != cmap_3


def test_colormap_recreate():
    c_map = Colormap("black")
    Colormap(**c_map.dict())


@pytest.mark.parametrize('ndim', range(1, 5))
def test_mapped_shape(ndim):
    np.random.seed(0)
    img = np.random.random((5,) * ndim)
    cmap = Colormap(colors=['red'])
    mapped = cmap.map(img)
    assert mapped.shape == img.shape + (4,)


@pytest.mark.parametrize(
    "num,dtype", [(40, np.uint8), (1000, np.uint16), (80000, np.float32)]
)
def test_minimum_dtype_for_labels(num, dtype):
    assert colormap.minimum_dtype_for_labels(num) == dtype


@pytest.fixture()
def disable_jit(monkeypatch):
    pytest.importorskip("numba")
    with patch("numba.core.config.DISABLE_JIT", True):
        importlib.reload(colormap)
        yield
    importlib.reload(colormap)  # revert to original state


@pytest.mark.parametrize(
    "num,dtype", [(40, np.uint8), (1000, np.uint16), (80000, np.float32)]
)
@pytest.mark.usefixtures("disable_jit")
def test_cast_labels_to_minimum_type_auto(num: int, dtype, monkeypatch):
    cmap = label_colormap(num)
    data = np.zeros(3, dtype=np.uint32)
    data[1] = 10
    data[2] = 10**6 + 5
    cast_arr = colormap._cast_labels_to_minimum_dtype_auto(data, cmap)
    assert cast_arr.dtype == dtype
    assert cast_arr[0] == 0
    assert cast_arr[1] == 10
    assert cast_arr[2] == 10**6 % num + 5


@pytest.fixture
def direct_label_colormap():
    return DirectLabelColormap(
        np.zeros(3),
        color_dict={
            0: np.array([0, 0, 0, 0]),
            1: np.array([255, 0, 0, 255]),
            2: np.array([0, 255, 0, 255]),
            3: np.array([0, 0, 255, 255]),
            12: np.array([0, 0, 255, 255]),
            None: np.array([255, 255, 255, 255]),
        },
    )


def test_direct_label_colormap_simple(direct_label_colormap):
    np.testing.assert_array_equal(
        direct_label_colormap.map([0, 2, 7]),
        np.array([[0, 0, 0, 0], [0, 255, 0, 255], [255, 255, 255, 255]]),
    )
    assert direct_label_colormap.unique_colors_num() == 5

    (
        label_mapping,
        color_dict,
    ) = direct_label_colormap.values_mapping_to_minimum_values_set()

    assert len(label_mapping) == 6
    assert len(color_dict) == 5
    assert label_mapping[None] == DEFAULT_VALUE
    assert label_mapping[12] == label_mapping[3]
    np.testing.assert_array_equal(
        color_dict[label_mapping[0]], direct_label_colormap.color_dict[0]
    )
    np.testing.assert_array_equal(
        color_dict[0], direct_label_colormap.color_dict[None]
    )


def test_direct_label_colormap_selection(direct_label_colormap):
    direct_label_colormap.selection = 2
    direct_label_colormap.use_selection = True

    np.testing.assert_array_equal(
        direct_label_colormap.map([0, 2, 7]),
        np.array([[0, 0, 0, 0], [0, 255, 0, 255], [0, 0, 0, 0]]),
    )

    (
        label_mapping,
        color_dict,
    ) = direct_label_colormap.values_mapping_to_minimum_values_set()

    assert len(label_mapping) == 2
    assert len(color_dict) == 2


def test_cast_direct_labels_to_minimum_type(direct_label_colormap):
    data = np.arange(15, dtype=np.uint32)
    casted = colormap._cast_direct_labels_to_minimum_type_impl(
        data, direct_label_colormap
    )
    label_mapping = (
        direct_label_colormap.values_mapping_to_minimum_values_set()[0]
    )
    assert casted.dtype == np.uint8
    np.testing.assert_array_equal(
        casted,
        np.array(
            [
                label_mapping[0],
                label_mapping[1],
                label_mapping[2],
                label_mapping[3],
                DEFAULT_VALUE,
                DEFAULT_VALUE,
                DEFAULT_VALUE,
                DEFAULT_VALUE,
                DEFAULT_VALUE,
                DEFAULT_VALUE,
                DEFAULT_VALUE,
                DEFAULT_VALUE,
                label_mapping[3],
                DEFAULT_VALUE,
                DEFAULT_VALUE,
            ]
        ),
    )


@pytest.mark.parametrize(
    "num,dtype", [(40, np.uint8), (1000, np.uint16), (80000, np.float32)]
)
@pytest.mark.usefixtures("disable_jit")
def test_test_cast_direct_labels_to_minimum_type_no_jit(num, dtype):
    cmap = DirectLabelColormap(
        np.zeros(3),
        color_dict={
            k: np.array([*v, 255])
            for k, v in zip(range(num), product(range(256), repeat=3))
        },
    )
    cmap.color_dict[None] = np.array([255, 255, 255, 255])
    data = np.arange(10, dtype=np.uint32)
    data[2] = 80005
    casted = colormap._cast_direct_labels_to_minimum_type_impl(data, cmap)
    assert casted.dtype == dtype


def test_zero_preserving_modulo_naive():
    pytest.importorskip("numba")
    data = np.arange(1000, dtype=np.uint32)
    res1 = colormap._zero_preserving_modulo_numpy(data, 49, np.uint8)
    res2 = colormap._zero_preserving_modulo(data, 49, np.uint8)
    npt.assert_array_equal(res1, res2)


@pytest.mark.parametrize(
    'dtype', [np.uint8, np.uint16, np.int8, np.int16, np.float32, np.float64]
)
def test_label_colormap_map_with_uint8_values(dtype):
    cmap = colormap.LabelColormap(
        colors=ColorArray(np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]]))
    )
    values = np.array([0, 1, 2], dtype=dtype)
    expected = np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
    npt.assert_array_equal(cmap.map(values), expected)


@pytest.mark.parametrize("selection", [1, -1])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_label_colormap_map_with_selection(selection, dtype):
    cmap = colormap.LabelColormap(
        colors=ColorArray(
            np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
        ),
        use_selection=True,
        selection=selection,
    )
    values = np.array([0, selection, 2], dtype=np.int8)
    expected = np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]])
    npt.assert_array_equal(cmap.map(values), expected)


@pytest.mark.parametrize("background", [1, -1])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_label_colormap_map_with_background(background, dtype):
    cmap = colormap.LabelColormap(
        colors=ColorArray(
            np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
        ),
        background_value=background,
    )
    values = np.array([3, background, 2], dtype=dtype)
    expected = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]])
    npt.assert_array_equal(cmap.map(values), expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_label_colormap_using_cache(dtype, monkeypatch):
    cmap = colormap.LabelColormap(
        colors=ColorArray(np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]]))
    )
    values = np.array([0, 1, 2], dtype=dtype)
    expected = np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
    map1 = cmap.map(values)
    npt.assert_array_equal(map1, expected)
    getattr(cmap, f"_{dtype.__name__}_colors")
    monkeypatch.setattr(colormap, '_zero_preserving_modulo_naive', None)
    map2 = cmap.map(values)
    npt.assert_array_equal(map1, map2)


@pytest.mark.parametrize("size", [100, 1000])
def test_cast_direct_labels_to_minimum_type_naive(size):
    pytest.importorskip("numba")
    data = np.arange(size, dtype=np.uint32)
    dtype = colormap.minimum_dtype_for_labels(size)
    cmap = DirectLabelColormap(
        np.zeros(3),
        color_dict={
            k: np.array([*v, 255])
            for k, v in zip(range(size - 2), product(range(256), repeat=3))
        },
    )
    cmap.color_dict[None] = np.array([255, 255, 255, 255])
    res1 = colormap._cast_direct_labels_to_minimum_type_impl(data, cmap)
    res2 = colormap._cast_direct_labels_to_minimum_type_naive(data, cmap)
    npt.assert_array_equal(res1, res2)
    assert res1.dtype == dtype
    assert res2.dtype == dtype
