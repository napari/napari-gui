import itertools
import time
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import List

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from numpy.core.numerictypes import issubdtype
from numpy.testing import assert_array_almost_equal, assert_raises
from skimage import data
from vispy.color import Colormap as VispyColormap

from napari._tests.utils import check_layer_world_data_extent
from napari.components import ViewerModel
from napari.layers import Labels
from napari.layers.labels._labels_constants import LabelsRendering
from napari.layers.labels._labels_utils import get_contours
from napari.utils import Colormap
from napari.utils.colormaps import label_colormap, low_discrepancy_image


def test_random_labels():
    """Test instantiating Labels layer with random 2D data."""
    shape = (10, 15)
    np.random.seed(0)
    data = np.random.randint(20, size=shape)
    layer = Labels(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], [s - 1 for s in shape])
    assert layer._data_view.shape == shape[-2:]
    assert layer.editable is True


def test_all_zeros_labels():
    """Test instantiating Labels layer with all zeros data."""
    shape = (10, 15)
    data = np.zeros(shape, dtype=int)
    layer = Labels(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], [s - 1 for s in shape])
    assert layer._data_view.shape == shape[-2:]


def test_3D_labels():
    """Test instantiating Labels layer with random 3D data."""
    shape = (6, 10, 15)
    np.random.seed(0)
    data = np.random.randint(20, size=shape)
    layer = Labels(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], [s - 1 for s in shape])
    assert layer._data_view.shape == shape[-2:]
    assert layer.editable is True

    layer._slice_dims(ndisplay=3)
    assert layer._slice_input.ndisplay == 3
    assert layer.editable is True
    assert layer.mode == 'pan_zoom'


def test_float_labels():
    """Test instantiating labels layer with floats"""
    np.random.seed(0)
    data = np.random.uniform(0, 20, size=(10, 10))
    with pytest.raises(TypeError):
        Labels(data)

    data0 = np.random.uniform(20, size=(20, 20))
    data1 = data0[::2, ::2].astype(np.int32)
    data = [data0, data1]
    with pytest.raises(TypeError):
        Labels(data)


def test_bool_labels():
    """Test instantiating labels layer with bools"""
    data = np.zeros((10, 10), dtype=bool)
    layer = Labels(data)
    assert issubdtype(layer.data.dtype, np.integer)

    data0 = np.zeros((20, 20), dtype=bool)
    data1 = data0[::2, ::2].astype(np.int32)
    data = [data0, data1]
    layer = Labels(data)
    assert all(issubdtype(d.dtype, np.integer) for d in layer.data)


def test_changing_labels():
    """Test changing Labels data."""
    shape_a = (10, 15)
    shape_b = (20, 12)
    shape_c = (10, 10)
    np.random.seed(0)
    data_a = np.random.randint(20, size=shape_a)
    data_b = np.random.randint(20, size=shape_b)
    layer = Labels(data_a)
    layer.data = data_b
    assert np.all(layer.data == data_b)
    assert layer.ndim == len(shape_b)
    np.testing.assert_array_equal(
        layer.extent.data[1], [s - 1 for s in shape_b]
    )
    assert layer._data_view.shape == shape_b[-2:]

    data_c = np.zeros(shape_c, dtype=bool)
    layer.data = data_c
    assert np.issubdtype(layer.data.dtype, np.integer)

    data_c = data_c.astype(np.float32)
    with pytest.raises(TypeError):
        layer.data = data_c


def test_changing_labels_dims():
    """Test changing Labels data including dimensionality."""
    shape_a = (10, 15)
    shape_b = (20, 12, 6)
    np.random.seed(0)
    data_a = np.random.randint(20, size=shape_a)
    data_b = np.random.randint(20, size=shape_b)
    layer = Labels(data_a)

    layer.data = data_b
    assert np.all(layer.data == data_b)
    assert layer.ndim == len(shape_b)
    np.testing.assert_array_equal(
        layer.extent.data[1], [s - 1 for s in shape_b]
    )
    assert layer._data_view.shape == shape_b[-2:]


def test_changing_modes():
    """Test changing modes."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.mode == 'pan_zoom'
    assert layer.mouse_pan is True

    layer.mode = 'fill'
    assert layer.mode == 'fill'
    assert layer.mouse_pan is False

    layer.mode = 'paint'
    assert layer.mode == 'paint'
    assert layer.mouse_pan is False

    layer.mode = 'pick'
    assert layer.mode == 'pick'
    assert layer.mouse_pan is False

    layer.mode = 'pan_zoom'
    assert layer.mode == 'pan_zoom'
    assert layer.mouse_pan is True

    layer.mode = 'paint'
    assert layer.mode == 'paint'
    layer.editable = False
    assert layer.mode == 'pan_zoom'
    assert layer.editable is False


def test_name():
    """Test setting layer name."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.name == 'Labels'

    layer = Labels(data, name='random')
    assert layer.name == 'random'

    layer.name = 'lbls'
    assert layer.name == 'lbls'


def test_visiblity():
    """Test setting layer visibility."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer = Labels(data, visible=False)
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_opacity():
    """Test setting layer opacity."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.opacity == 0.7

    layer.opacity = 0.5
    assert layer.opacity == 0.5

    layer = Labels(data, opacity=0.6)
    assert layer.opacity == 0.6

    layer.opacity = 0.3
    assert layer.opacity == 0.3


def test_blending():
    """Test setting layer blending."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Labels(data, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


def test_seed():
    """Test setting seed."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.seed == 0.5

    layer.seed = 0.9
    assert layer.seed == 0.9

    layer = Labels(data, seed=0.7)
    assert layer.seed == 0.7

    # ensure setting seed triggers
    # recalculation of _all_vals
    _all_vals_07 = layer._all_vals.copy()
    layer.seed = 0.4
    _all_vals_04 = layer._all_vals.copy()
    assert_raises(
        AssertionError, assert_array_almost_equal, _all_vals_04, _all_vals_07
    )


def test_num_colors():
    """Test setting number of colors in colormap."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.num_colors == 50

    layer.num_colors = 80
    assert layer.num_colors == 80

    layer = Labels(data, num_colors=60)
    assert layer.num_colors == 60


def test_properties():
    """Test adding labels with properties."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))

    layer = Labels(data)
    assert isinstance(layer.properties, dict)
    assert len(layer.properties) == 0

    properties = {
        'class': np.array(['Background'] + [f'Class {i}' for i in range(20)])
    }
    label_index = {i: i for i in range(len(properties['class']))}
    layer = Labels(data, properties=properties)
    assert isinstance(layer.properties, dict)
    np.testing.assert_equal(layer.properties, properties)
    assert layer._label_index == label_index
    layer = Labels(data)
    layer.properties = properties
    assert isinstance(layer.properties, dict)
    np.testing.assert_equal(layer.properties, properties)
    assert layer._label_index == label_index

    current_label = layer.get_value((0, 0))
    layer_message = layer.get_status((0, 0))
    assert layer_message['coordinates'].endswith(f'Class {current_label - 1}')

    properties = {'class': ['Background']}
    layer = Labels(data, properties=properties)
    layer_message = layer.get_status((0, 0))
    assert layer_message['coordinates'].endswith("[No Properties]")

    properties = {'class': ['Background', 'Class 12'], 'index': [0, 12]}
    label_index = {0: 0, 12: 1}
    layer = Labels(data, properties=properties)
    layer_message = layer.get_status((0, 0))
    assert layer._label_index == label_index
    assert layer_message['coordinates'].endswith('Class 12')

    layer = Labels(data)
    layer.properties = properties
    layer_message = layer.get_status((0, 0))
    assert layer._label_index == label_index
    assert layer_message['coordinates'].endswith('Class 12')

    layer = Labels(data)
    layer.properties = pd.DataFrame(properties)
    layer_message = layer.get_status((0, 0))
    assert layer._label_index == label_index
    assert layer_message['coordinates'].endswith('Class 12')


def test_default_properties_assignment():
    """Test that the default properties value can be assigned to properties
    see https://github.com/napari/napari/issues/2477
    """
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))

    layer = Labels(data)
    layer.properties = {}
    assert layer.properties == {}


def test_multiscale_properties():
    """Test adding labels with multiscale properties."""
    np.random.seed(0)
    data0 = np.random.randint(20, size=(10, 15))
    data1 = data0[::2, ::2]
    data = [data0, data1]

    layer = Labels(data)
    assert isinstance(layer.properties, dict)
    assert len(layer.properties) == 0

    properties = {
        'class': np.array(['Background'] + [f'Class {i}' for i in range(20)])
    }
    label_index = {i: i for i in range(len(properties['class']))}
    layer = Labels(data, properties=properties)
    assert isinstance(layer.properties, dict)
    np.testing.assert_equal(layer.properties, properties)
    assert layer._label_index == label_index

    current_label = layer.get_value((0, 0))[1]
    layer_message = layer.get_status((0, 0))
    assert layer_message['coordinates'].endswith(f'Class {current_label - 1}')

    properties = {'class': ['Background']}
    layer = Labels(data, properties=properties)
    layer_message = layer.get_status((0, 0))
    assert layer_message['coordinates'].endswith("[No Properties]")

    properties = {'class': ['Background', 'Class 12'], 'index': [0, 12]}
    label_index = {0: 0, 12: 1}
    layer = Labels(data, properties=properties)
    layer_message = layer.get_status((0, 0))
    assert layer._label_index == label_index
    assert layer_message['coordinates'].endswith('Class 12')


def test_predefined_labels():
    """Test predefined labels without names"""

    predefined_labels = [10, 20, 30, 255]
    labels = Labels(
        np.zeros((10, 10), dtype=int), predefined_labels=predefined_labels
    )
    assert set(labels._predefined_labels) == set(
        predefined_labels + [labels._background_label]
    )
    assert labels._predefined_labels[labels._background_label] == "background"
    assert labels.get_label_name(labels._background_label) == "background"

    for label in predefined_labels:
        assert labels.get_label_name(label) is None

    labels.selected_label = 5
    assert labels.get_label_name(5) == "unspecified"

    assert (set(labels._predefined_labels) - set(predefined_labels)) == {0, 5}


def test_predefined_labels_with_names():
    """Test predefined labels with names"""

    predefined_labels = {10: 'ten', 20: 'twenty', 30: 'thirty'}
    data = 10 * np.arange(5, dtype=np.int32)[:, np.newaxis].repeat(5, axis=1)
    labels = Labels(data, predefined_labels=predefined_labels)

    for label_value, label_name in predefined_labels.items():
        assert labels.get_label_name(label_value) == label_name

    assert labels.get_label_name(5) is None

    layer_message = labels.get_status((0, 0))
    assert layer_message['coordinates'].endswith(' background')

    for i in range(1, 4):
        layer_message = labels.get_status((i, 2))
        assert layer_message['coordinates'].endswith(predefined_labels[i * 10])

    layer_message = labels.get_status((4, 0))
    assert layer_message['coordinates'].endswith('40')


def test_colormap():
    """Test colormap."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert isinstance(layer.colormap, Colormap)
    assert layer.colormap.name == 'label_colormap'

    layer.new_colormap()
    assert isinstance(layer.colormap, Colormap)
    assert layer.colormap.name == 'label_colormap'


def test_label_colormap():
    """Test a label colormap."""
    colormap = label_colormap(num_colors=4)

    # Make sure color 0 is transparent
    assert not np.any(colormap.map([0.0]))

    # Test that out-of-range values map to last value
    assert np.all(colormap.map([1.0, 1.1, 2.0]) == colormap.colors[-1])


def test_custom_color_dict():
    """Test custom color dict."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(
        data, color={2: 'white', 4: 'red', 8: 'blue', 16: 'red', 32: 'blue'}
    )

    # test with custom color dict
    assert type(layer.get_color(2)) == np.ndarray
    assert type(layer.get_color(1)) == np.ndarray
    assert (layer.get_color(2) == np.array([1.0, 1.0, 1.0, 1.0])).all()
    assert (layer.get_color(4) == layer.get_color(16)).all()
    assert (layer.get_color(8) == layer.get_color(32)).all()

    # Test to see if our label mapped control points map to those in the colormap
    # with an extra half step.
    local_controls = np.array(
        sorted(np.unique([*layer._label_color_index.values(), 1.0]))
    )
    colormap_controls = np.array(layer._colormap.controls)
    assert np.max(np.abs(local_controls - colormap_controls)) == pytest.approx(
        0.5 / (len(colormap_controls) - 1)
    )

    # test disable custom color dict
    # should not initialize as white since we are using random.seed
    layer.color_mode = 'auto'
    assert not (layer.get_color(1) == np.array([1.0, 1.0, 1.0, 1.0])).all()


def test_large_custom_color_dict():
    """Confirm that the napari & vispy colormaps behave the same."""

    label_count = 897
    colors = {
        color: (0, (color / 256.0) / 256.0, (color % 256) / 256.0)
        for color in range(label_count)
    }
    data, _ = np.meshgrid(range(label_count), range(5))
    layer = Labels(data, color=colors)

    # Get color list using layer interface & napari.utils.colormap.ColorMap
    label_color = layer.get_color(list(range(label_count)))

    # Get the color by converting to control points with the layer and passing
    # that to a vispy.color.colormap.Colormap
    vispy_colormap = VispyColormap(
        colors=layer.colormap.colors,
        controls=layer.colormap.controls,
        interpolation='zero',
    )
    label_color_controls = [
        layer._label_color_index[x] for x in range(label_count)
    ]
    vispy_colors = vispy_colormap.map(np.array(list(label_color_controls)))

    assert (label_color == vispy_colors).all()


def test_warning_too_many_colors():
    label_count = 1500
    colors = {
        color: (0, (color / 256.0) / 256.0, (color % 256) / 256.0)
        for color in range(label_count)
    }
    data, _ = np.meshgrid(range(label_count), range(5))
    with pytest.warns(UserWarning):
        # Expect a warning for 1500 colors > 1024 in LUT
        Labels(data, color=colors)


def test_add_colors():
    """Test adding new colors"""
    data = np.random.randint(20, size=(40, 40))
    layer = Labels(data)
    assert len(layer._all_vals) == np.max(data) + 1

    layer.selected_label = 51
    assert len(layer._all_vals) == 52

    layer.show_selected_label = True
    layer.selected_label = 53
    assert len(layer._all_vals) == 54


def test_metadata():
    """Test setting labels metadata."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.metadata == {}

    layer = Labels(data, metadata={'unit': 'cm'})
    assert layer.metadata == {'unit': 'cm'}


def test_brush_size():
    """Test changing brush size."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.brush_size == 10

    layer.brush_size = 20
    assert layer.brush_size == 20


def test_contiguous():
    """Test changing contiguous."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.contiguous is True

    layer.contiguous = False
    assert layer.contiguous is False


def test_n_edit_dimensions():
    """Test changing the number of editable dimensions."""
    np.random.seed(0)
    data = np.random.randint(20, size=(5, 10, 15))
    layer = Labels(data)
    layer.n_edit_dimensions = 2
    layer.n_edit_dimensions = 3


@pytest.mark.parametrize(
    "input_data, expected_data_view",
    [
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
                    [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
                    [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
                    [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
                    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int_,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
                    [0, 0, 1, 1, 1, 5, 0, 5, 0, 0],
                    [0, 0, 1, 0, 1, 5, 0, 5, 0, 0],
                    [0, 0, 1, 1, 1, 5, 0, 5, 0, 0],
                    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int_,
            ),
        ),
        (
            np.array(
                [
                    [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
                    [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
                    [3, 3, 3, 0, 0, 0, 4, 4, 4, 4],
                    [3, 3, 3, 0, 0, 0, 4, 4, 4, 4],
                    [3, 3, 3, 0, 0, 0, 4, 4, 4, 4],
                ],
                dtype=np.int_,
            ),
            np.array(
                [
                    [0, 1, 0, 0, 0, 0, 0, 2, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
                    [3, 3, 3, 0, 0, 0, 4, 0, 0, 0],
                    [0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
                    [0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
                ],
                dtype=np.int_,
            ),
        ),
        (
            5 * np.ones((9, 10), dtype=np.uint32),
            np.zeros((9, 10), dtype=np.uint32),
        ),
    ],
)
def test_contour(input_data, expected_data_view):
    """Test changing contour."""
    layer = Labels(input_data)
    assert layer.contour == 0
    np.testing.assert_array_equal(layer.data, input_data)

    np.testing.assert_array_equal(
        layer._raw_to_displayed(input_data), layer._data_view
    )
    data_view_before_contour = layer._data_view.copy()

    layer.contour = 1
    assert layer.contour == 1

    # Check `layer.data` didn't change
    np.testing.assert_array_equal(layer.data, input_data)

    # Check what is returned in the view of the data
    np.testing.assert_array_equal(
        layer._data_view,
        np.where(
            expected_data_view > 0,
            low_discrepancy_image(expected_data_view),
            0,
        ),
    )

    # Check the view of the data changed after setting the contour
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(
            data_view_before_contour, layer._data_view
        )

    layer.contour = 0
    assert layer.contour == 0

    # Check it's in the same state as before setting the contour
    np.testing.assert_array_equal(
        layer._raw_to_displayed(input_data), layer._data_view
    )

    with pytest.raises(ValueError, match='contour value must be >= 0'):
        layer.contour = -1


def test_contour_large_new_labels():
    """Check that new labels larger than the lookup table work in contour mode.

    References
    ----------
    [1]: https://forum.image.sc/t/data-specific-reason-for-indexerror-in-raw-to-displayed/60808
    [2]: https://github.com/napari/napari/pull/3697
    """
    viewer = ViewerModel()

    labels = np.zeros((5, 10, 10), dtype=int)
    labels[0, 4:6, 4:6] = 1
    labels[4, 4:6, 4:6] = 1000
    labels_layer = viewer.add_labels(labels)
    labels_layer.contour = 1
    # This used to fail with IndexError
    viewer.dims.set_point(axis=0, value=4)


def test_contour_local_updates():
    """Checks if contours are rendered correctly with local updates"""
    data = np.zeros((7, 7), dtype=np.int32)

    layer = Labels(data)
    layer.contour = 1
    assert np.allclose(
        layer._raw_to_displayed(layer._slice.image.raw),
        np.zeros((7, 7), dtype=np.float32),
    )

    painting_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    layer.data_setitem(np.nonzero(painting_mask), 1, refresh=True)

    assert np.alltrue(
        (layer._slice.image.view > 0) == get_contours(painting_mask, 1, 0)
    )


def test_selecting_label():
    """Test selecting label."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.selected_label == 1
    assert (layer._selected_color == layer.get_color(1)).all

    layer.selected_label = 1
    assert layer.selected_label == 1
    assert len(layer._selected_color) == 4


def test_label_color():
    """Test getting label color."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    col = layer.get_color(0)
    assert col is None

    col = layer.get_color(1)
    assert len(col) == 4


def test_show_selected_label():
    """Test color of labels when filtering to selected labels"""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    original_color = layer.get_color(1)

    layer.show_selected_label = True
    original_background_color = layer.get_color(layer._background_label)
    none_color = layer.get_color(None)
    layer.selected_label = 1

    # color of selected label has not changed
    assert np.allclose(layer.get_color(layer.selected_label), original_color)

    current_background_color = layer.get_color(layer._background_label)
    # color of background is background color
    assert current_background_color == original_background_color

    # color of all others is none color
    other_labels = np.unique(layer.data)[2:]
    other_colors = np.array([layer.get_color(x) for x in other_labels])
    assert np.allclose(other_colors, none_color)


def test_paint():
    """Test painting labels with different circle brush sizes."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    data[:10, :10] = 1
    layer = Labels(data)

    assert np.unique(layer.data[:5, :5]) == 1
    assert np.unique(layer.data[5:10, 5:10]) == 1

    layer.brush_size = 9
    layer.paint([0, 0], 2)
    assert np.unique(layer.data[:4, :4]) == 2
    assert np.unique(layer.data[5:10, 5:10]) == 1

    layer.brush_size = 10
    layer.paint([0, 0], 2)
    assert np.unique(layer.data[0:6, 0:3]) == 2
    assert np.unique(layer.data[0:3, 0:6]) == 2
    assert np.unique(layer.data[6:10, 6:10]) == 1

    layer.brush_size = 19
    layer.paint([0, 0], 2)
    assert np.unique(layer.data[0:4, 0:10]) == 2
    assert np.unique(layer.data[0:10, 0:4]) == 2
    assert np.unique(layer.data[3:7, 3:7]) == 2
    assert np.unique(layer.data[7:10, 7:10]) == 1


def test_paint_with_preserve_labels():
    """Test painting labels with square brush while preserving existing labels."""
    data = np.zeros((15, 10), dtype=np.uint32)
    data[:3, :3] = 1
    layer = Labels(data)

    layer.preserve_labels = True
    assert np.unique(layer.data[:3, :3]) == 1

    layer.brush_size = 9
    layer.paint([0, 0], 2)

    assert np.unique(layer.data[3:5, 0:3]) == 2
    assert np.unique(layer.data[0:3, 3:5]) == 2
    assert np.unique(layer.data[:3, :3]) == 1


def test_paint_2d():
    """Test painting labels with circle brush."""
    data = np.zeros((40, 40), dtype=np.uint32)
    layer = Labels(data)
    layer.brush_size = 12
    layer.mode = 'paint'
    layer.paint((0, 0), 3)

    layer.brush_size = 12
    layer.paint((15, 8), 4)

    layer.brush_size = 13
    layer.paint((30.2, 7.8), 5)

    layer.brush_size = 12
    layer.paint((39, 39), 6)

    layer.brush_size = 20
    layer.paint((15, 27), 7)

    assert np.sum(layer.data[:8, :8] == 3) == 41
    assert np.sum(layer.data[9:22, 2:15] == 4) == 137
    assert np.sum(layer.data[24:37, 2:15] == 5) == 137
    assert np.sum(layer.data[33:, 33:] == 6) == 41
    assert np.sum(layer.data[5:26, 17:38] == 7) == 349


def test_paint_2d_xarray():
    """Test the memory usage of painting an xarray indirectly via timeout."""
    now = time.monotonic()
    data = xr.DataArray(np.zeros((3, 3, 1024, 1024), dtype=np.uint32))

    layer = Labels(data)
    layer.brush_size = 12
    layer.mode = 'paint'
    layer.paint((1, 1, 512, 512), 3)
    assert isinstance(layer.data, xr.DataArray)
    assert layer.data.sum() == 411
    elapsed = time.monotonic() - now
    assert elapsed < 1, "test was too slow, computation was likely not lazy"


def test_paint_3d():
    """Test painting labels with circle brush on 3D image."""
    data = np.zeros((30, 40, 40), dtype=np.uint32)
    layer = Labels(data)
    layer.brush_size = 12
    layer.mode = 'paint'

    # Paint in 2D
    layer.paint((10, 10, 10), 3)

    # Paint in 3D
    layer.n_edit_dimensions = 3
    layer.paint((10, 25, 10), 4)

    # Paint in 3D, preserve labels
    layer.n_edit_dimensions = 3
    layer.preserve_labels = True
    layer.paint((10, 15, 15), 5)

    assert np.sum(layer.data[4:17, 4:17, 4:17] == 3) == 137
    assert np.sum(layer.data[4:17, 19:32, 4:17] == 4) == 1189
    assert np.sum(layer.data[4:17, 9:32, 9:32] == 5) == 1103


def test_fill():
    """Test filling labels with different brush sizes."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    data[:10, :10] = 2
    data[:5, :5] = 1
    layer = Labels(data)
    assert np.unique(layer.data[:5, :5]) == 1
    assert np.unique(layer.data[5:10, 5:10]) == 2

    layer.fill([0, 0], 3)
    assert np.unique(layer.data[:5, :5]) == 3
    assert np.unique(layer.data[5:10, 5:10]) == 2


def test_value():
    """Test getting the value of the data at the current coordinates."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    value = layer.get_value((0, 0))
    assert value == data[0, 0]


@pytest.mark.parametrize(
    'position,view_direction,dims_displayed,world',
    [
        ([10, 5, 5], [1, 0, 0], [0, 1, 2], False),
        ([10, 5, 5], [1, 0, 0], [0, 1, 2], True),
        ([0, 10, 5, 5], [0, 1, 0, 0], [1, 2, 3], True),
    ],
)
def test_value_3d(position, view_direction, dims_displayed, world):
    """get_value should return label value in 3D"""
    data = np.zeros((20, 20, 20), dtype=int)
    data[0:10, 0:10, 0:10] = 1
    layer = Labels(data)
    layer._slice_dims([0, 0, 0], ndisplay=3)
    value = layer.get_value(
        position,
        view_direction=view_direction,
        dims_displayed=dims_displayed,
        world=world,
    )
    assert value == 1


def test_message():
    """Test converting value and coords to message."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    msg = layer.get_status((0, 0))
    assert type(msg) == dict


def test_thumbnail():
    """Test the image thumbnail for square data."""
    np.random.seed(0)
    data = np.random.randint(20, size=(30, 30))
    layer = Labels(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_world_data_extent():
    """Test extent after applying transforms."""
    np.random.seed(0)
    shape = (6, 10, 15)
    data = np.random.randint(20, size=(shape))
    layer = Labels(data)
    extent = np.array(((0,) * 3, [s - 1 for s in shape]))
    check_layer_world_data_extent(layer, extent, (3, 1, 1), (10, 20, 5))


@pytest.mark.parametrize(
    'brush_size, mode, selected_label, preserve_labels, n_dimensional',
    list(
        itertools.product(
            list(range(1, 22, 5)),
            ['fill', 'erase', 'paint'],
            [1, 20, 100],
            [True, False],
            [True, False],
        )
    ),
)
def test_undo_redo(
    brush_size,
    mode,
    selected_label,
    preserve_labels,
    n_dimensional,
):
    blobs = data.binary_blobs(length=64, volume_fraction=0.3, n_dim=3)
    layer = Labels(blobs)
    data_history = [blobs.copy()]
    layer.brush_size = brush_size
    layer.mode = mode
    layer.selected_label = selected_label
    layer.preserve_labels = preserve_labels
    layer.n_edit_dimensions = 3 if n_dimensional else 2
    coord = np.random.random((3,)) * (np.array(blobs.shape) - 1)
    while layer.data[tuple(coord.astype(int))] == 0 and np.any(layer.data):
        coord = np.random.random((3,)) * (np.array(blobs.shape) - 1)
    if layer.mode == 'fill':
        layer.fill(coord, layer.selected_label)
    if layer.mode == 'erase':
        layer.paint(coord, 0)
    if layer.mode == 'paint':
        layer.paint(coord, layer.selected_label)
    data_history.append(np.copy(layer.data))
    layer.undo()
    np.testing.assert_array_equal(layer.data, data_history[0])
    layer.redo()
    np.testing.assert_array_equal(layer.data, data_history[1])


def test_ndim_fill():
    test_array = np.zeros((5, 5, 5, 5), dtype=int)

    test_array[:, 1:3, 1:3, 1:3] = 1

    layer = Labels(test_array)
    layer.n_edit_dimensions = 3

    layer.fill((0, 1, 1, 1), 2)

    np.testing.assert_equal(layer.data[0, 1:3, 1:3, 1:3], 2)
    np.testing.assert_equal(layer.data[1, 1:3, 1:3, 1:3], 1)

    layer.n_edit_dimensions = 4

    layer.fill((1, 1, 1, 1), 3)

    np.testing.assert_equal(layer.data[0, 1:3, 1:3, 1:3], 2)
    np.testing.assert_equal(layer.data[1:, 1:3, 1:3, 1:3], 3)


def test_ndim_paint():
    test_array = np.zeros((5, 6, 7, 8), dtype=int)
    layer = Labels(test_array)
    layer.n_edit_dimensions = 3
    layer.brush_size = 2  # equivalent to 18-connected 3D neighborhood
    layer.paint((1, 1, 1, 1), 1)

    assert np.sum(layer.data) == 19  # 18 + center
    assert not np.any(layer.data[0]) and not np.any(layer.data[2:])

    layer.n_edit_dimensions = 2  # 3x3 square
    layer._slice_dims(order=[1, 2, 0, 3])
    layer.paint((4, 5, 6, 7), 8)
    assert len(np.flatnonzero(layer.data == 8)) == 4  # 2D square is in corner
    np.testing.assert_array_equal(
        test_array[:, 5, 6, :],
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 8, 8],
                [0, 0, 0, 0, 0, 0, 8, 8],
            ]
        ),
    )


def test_switching_display_func():
    label_data = np.random.randint(2**25, 2**25 + 5, size=(50, 50))
    layer = Labels(label_data)
    assert layer._color_lookup_func == layer._lookup_with_low_discrepancy_image

    label_data = np.random.randint(0, 5, size=(50, 50))
    layer = Labels(label_data)
    assert layer._color_lookup_func == layer._lookup_with_index


def test_cursor_size_with_negative_scale():
    layer = Labels(np.zeros((5, 5), dtype=int), scale=[-1, -1])
    layer.mode = 'paint'
    assert layer.cursor_size > 0


def test_switching_display_func_during_slicing():
    label_array = (5e6 * np.ones((2, 2, 2))).astype(np.uint64)
    label_array[0, :, :] = [[0, 1], [2, 3]]
    layer = Labels(label_array)
    layer._slice_dims(point=(1, 0, 0))
    assert layer._color_lookup_func == layer._lookup_with_low_discrepancy_image
    assert layer._all_vals.size < 1026


def test_add_large_colors():
    label_array = (5e6 * np.ones((2, 2, 2))).astype(np.uint64)
    label_array[0, :, :] = [[0, 1], [2, 3]]
    layer = Labels(label_array)
    assert len(layer._all_vals) == 4

    layer.show_selected_label = True
    layer.selected_label = int(5e6)
    assert layer._all_vals.size < 1026


def test_fill_tensorstore():
    ts = pytest.importorskip('tensorstore')

    labels = np.zeros((5, 7, 8, 9), dtype=int)
    labels[1, 2:4, 4:6, 4:6] = 1
    labels[1, 3:5, 5:7, 6:8] = 2
    labels[2, 3:5, 5:7, 6:8] = 3
    with TemporaryDirectory(suffix='.zarr') as fout:
        labels_temp = zarr.open(
            fout,
            mode='w',
            shape=labels.shape,
            dtype=np.uint32,
            chunks=(1, 1, 8, 9),
        )
        labels_temp[:] = labels
        labels_ts_spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': fout},
            'path': '',
            'metadata': {
                'dtype': labels_temp.dtype.str,
                'order': labels_temp.order,
                'shape': labels.shape,
            },
        }
        data = ts.open(labels_ts_spec, create=False, open=True).result()
        layer = Labels(data)
        layer.n_edit_dimensions = 3
        layer.fill((1, 4, 6, 7), 4)
        modified_labels = np.where(labels == 2, 4, labels)
        np.testing.assert_array_equal(modified_labels, np.asarray(data))


def test_fill_with_xarray():
    """See https://github.com/napari/napari/issues/2374"""
    data = xr.DataArray(np.zeros((5, 4, 4), dtype=int))
    layer = Labels(data)

    layer.fill((0, 2, 2), 1)

    np.testing.assert_array_equal(layer.data[0, :, :], np.ones((4, 4)))
    np.testing.assert_array_equal(layer.data[1:, :, :], np.zeros((4, 4, 4)))
    # In the associated issue, using xarray.DataArray caused memory allocation
    # problems due to different read indexing rules, so check that the data
    # saved for undo has the expected vectorized shape and values.
    undo_data = layer._undo_history[0][0][1]
    np.testing.assert_array_equal(undo_data, np.zeros((16,)))


@pytest.mark.parametrize(
    'scale', list(itertools.product([-2, 2], [-0.5, 0.5], [-0.5, 0.5]))
)
def test_paint_3d_negative_scale(scale):
    labels = np.zeros((3, 5, 11, 11), dtype=int)
    labels_layer = Labels(
        labels, scale=(1, *scale), translate=(-200, 100, 100)
    )
    labels_layer.n_edit_dimensions = 3
    labels_layer.brush_size = 8
    labels_layer.paint((1, 2, 5, 5), 1)
    np.testing.assert_array_equal(
        np.sum(labels_layer.data, axis=(1, 2, 3)), [0, 95, 0]
    )


def test_rendering_init():
    shape = (6, 10, 15)
    np.random.seed(0)
    data = np.random.randint(20, size=shape)
    layer = Labels(data, rendering='iso_categorical')

    assert layer.rendering == LabelsRendering.ISO_CATEGORICAL.value


def test_3d_video_and_3d_scale_translate_then_scale_translate_padded():
    # See the GitHub issue for more details:
    # https://github.com/napari/napari/issues/2967
    data = np.zeros((3, 5, 11, 11), dtype=int)
    labels = Labels(data, scale=(2, 1, 1), translate=(5, 5, 5))

    np.testing.assert_array_equal(labels.scale, (1, 2, 1, 1))
    np.testing.assert_array_equal(labels.translate, (0, 5, 5, 5))


@dataclass
class MouseEvent:
    # mock mouse event class
    pos: List[int]
    position: List[int]
    dims_point: List[int]
    dims_displayed: List[int]
    view_direction: List[int]


def test_get_value_ray_3d():
    """Test using _get_value_ray to interrogate labels in 3D"""
    # make a mock mouse event
    mouse_event = MouseEvent(
        pos=[25, 25],
        position=[10, 5, 5],
        dims_point=[1, 0, 0, 0],
        dims_displayed=[1, 2, 3],
        view_direction=[1, 0, 0],
    )
    data = np.zeros((5, 20, 20, 20), dtype=int)
    data[1, 0:10, 0:10, 0:10] = 1
    labels = Labels(data, scale=(1, 2, 1, 1), translate=(5, 5, 5))

    # set the dims to the slice with labels
    labels._slice_dims([1, 0, 0, 0], ndisplay=3)

    value = labels._get_value_ray(
        start_point=np.array([1, 0, 5, 5]),
        end_point=np.array([1, 20, 5, 5]),
        dims_displayed=mouse_event.dims_displayed,
    )
    assert value == 1

    # check with a ray that only goes through background
    value = labels._get_value_ray(
        start_point=np.array([1, 0, 15, 15]),
        end_point=np.array([1, 20, 15, 15]),
        dims_displayed=mouse_event.dims_displayed,
    )
    assert value is None

    # set the dims to a slice without labels
    labels._slice_dims([0, 0, 0, 0], ndisplay=3)

    value = labels._get_value_ray(
        start_point=np.array([0, 0, 5, 5]),
        end_point=np.array([0, 20, 5, 5]),
        dims_displayed=mouse_event.dims_displayed,
    )
    assert value is None


def test_get_value_ray_3d_rolled():
    """Test using _get_value_ray to interrogate labels in 3D
    with the dimensions rolled.
    """
    # make a mock mouse event
    mouse_event = MouseEvent(
        pos=[25, 25],
        position=[10, 5, 5, 1],
        dims_point=[0, 0, 0, 1],
        dims_displayed=[0, 1, 2],
        view_direction=[1, 0, 0, 0],
    )
    data = np.zeros((20, 20, 20, 5), dtype=int)
    data[0:10, 0:10, 0:10, 1] = 1
    labels = Labels(data, scale=(1, 2, 1, 1), translate=(5, 5, 5, 0))

    # set the dims to the slice with labels
    labels._slice_dims((0, 0, 0, 1), ndisplay=3, order=(3, 0, 1, 2))
    labels.set_view_slice()

    value = labels._get_value_ray(
        start_point=np.array([0, 5, 5, 1]),
        end_point=np.array([20, 5, 5, 1]),
        dims_displayed=mouse_event.dims_displayed,
    )
    assert value == 1


def test_get_value_ray_3d_transposed():
    """Test using _get_value_ray to interrogate labels in 3D
    with the dimensions transposed.
    """
    # make a mock mouse event
    mouse_event = MouseEvent(
        pos=[25, 25],
        position=[10, 5, 5, 1],
        dims_point=[0, 0, 0, 1],
        dims_displayed=[1, 3, 2],
        view_direction=[1, 0, 0, 0],
    )
    data = np.zeros((5, 20, 20, 20), dtype=int)
    data[1, 0:10, 0:10, 0:10] = 1
    labels = Labels(data, scale=(1, 2, 1, 1), translate=(0, 5, 5, 5))

    # set the dims to the slice with labels
    labels._slice_dims((1, 0, 0, 0), ndisplay=3, order=(0, 1, 3, 2))
    labels.set_view_slice()

    value = labels._get_value_ray(
        start_point=np.array([1, 0, 5, 5]),
        end_point=np.array([1, 20, 5, 5]),
        dims_displayed=mouse_event.dims_displayed,
    )
    assert value == 1


def test_get_value_ray_2d():
    """_get_value_ray currently only returns None in 2D
    (i.e., it shouldn't be used for 2D).
    """
    # make a mock mouse event
    mouse_event = MouseEvent(
        pos=[25, 25],
        position=[5, 5],
        dims_point=[1, 10, 0, 0],
        dims_displayed=[2, 3],
        view_direction=[1, 0, 0],
    )
    data = np.zeros((5, 20, 20, 20), dtype=int)
    data[1, 0:10, 0:10, 0:10] = 1
    labels = Labels(data, scale=(1, 2, 1, 1), translate=(5, 5, 5))

    # set the dims to the slice with labels, but 2D
    labels._slice_dims([1, 10, 0, 0], ndisplay=2)

    value = labels._get_value_ray(
        start_point=np.empty([]),
        end_point=np.empty([]),
        dims_displayed=mouse_event.dims_displayed,
    )
    assert value is None


def test_cursor_ray_3d():
    # make a mock mouse event
    mouse_event_1 = MouseEvent(
        pos=[25, 25],
        position=[1, 10, 27, 10],
        dims_point=[1, 0, 0, 0],
        dims_displayed=[1, 2, 3],
        view_direction=[0, 1, 0, 0],
    )
    data = np.zeros((5, 20, 20, 20), dtype=int)
    data[1, 0:10, 0:10, 0:10] = 1
    labels = Labels(data, scale=(1, 1, 2, 1), translate=(5, 5, 5))

    # set the slice to one with data and the view to 3D
    labels._slice_dims([1, 0, 0, 0], ndisplay=3)

    # axis 0 : [0, 20], bounding box extents along view axis, [1, 0, 0]
    # click is transformed: (value - translation) / scale
    # axis 1: click at 27 in world coords -> (27 - 5) / 2 = 11
    # axis 2: click at 10 in world coords -> (10 - 5) / 1 = 5
    start_point, end_point = labels.get_ray_intersections(
        mouse_event_1.position,
        mouse_event_1.view_direction,
        mouse_event_1.dims_displayed,
    )
    np.testing.assert_allclose(start_point, [1, 0, 11, 5])
    np.testing.assert_allclose(end_point, [1, 20, 11, 5])

    # click in the background
    mouse_event_2 = MouseEvent(
        pos=[25, 25],
        position=[1, 10, 65, 10],
        dims_point=[1, 0, 0, 0],
        dims_displayed=[1, 2, 3],
        view_direction=[0, 1, 0, 0],
    )
    start_point, end_point = labels.get_ray_intersections(
        mouse_event_2.position,
        mouse_event_2.view_direction,
        mouse_event_2.dims_displayed,
    )
    assert start_point is None
    assert end_point is None

    # click in a slice with no labels
    mouse_event_3 = MouseEvent(
        pos=[25, 25],
        position=[0, 10, 27, 10],
        dims_point=[0, 0, 0, 0],
        dims_displayed=[1, 2, 3],
        view_direction=[0, 1, 0, 0],
    )
    labels._slice_dims([0, 0, 0, 0], ndisplay=3)
    start_point, end_point = labels.get_ray_intersections(
        mouse_event_3.position,
        mouse_event_3.view_direction,
        mouse_event_3.dims_displayed,
    )
    np.testing.assert_allclose(start_point, [0, 0, 11, 5])
    np.testing.assert_allclose(end_point, [0, 20, 11, 5])


def test_cursor_ray_3d_rolled():
    """Test that the cursor works when the displayed
    viewer axes have been rolled
    """
    # make a mock mouse event
    mouse_event_1 = MouseEvent(
        pos=[25, 25],
        position=[10, 27, 10, 1],
        dims_point=[0, 0, 0, 1],
        dims_displayed=[0, 1, 2],
        view_direction=[1, 0, 0, 0],
    )
    data = np.zeros((20, 20, 20, 5), dtype=int)
    data[0:10, 0:10, 0:10, 1] = 1
    labels = Labels(data, scale=(1, 2, 1, 1), translate=(5, 5, 5, 0))

    # set the slice to one with data and the view to 3D
    labels._slice_dims([0, 0, 0, 1], ndisplay=3)

    start_point, end_point = labels.get_ray_intersections(
        mouse_event_1.position,
        mouse_event_1.view_direction,
        mouse_event_1.dims_displayed,
    )
    np.testing.assert_allclose(start_point, [0, 11, 5, 1])
    np.testing.assert_allclose(end_point, [20, 11, 5, 1])


def test_cursor_ray_3d_transposed():
    """Test that the cursor works when the displayed
    viewer axes have been transposed
    """
    # make a mock mouse event
    mouse_event_1 = MouseEvent(
        pos=[25, 25],
        position=[10, 27, 10, 1],
        dims_point=[0, 0, 0, 1],
        dims_displayed=[0, 2, 1],
        view_direction=[1, 0, 0, 0],
    )
    data = np.zeros((20, 20, 20, 5), dtype=int)
    data[0:10, 0:10, 0:10, 1] = 1
    labels = Labels(data, scale=(1, 2, 1, 1), translate=(5, 5, 5, 0))

    # set the slice to one with data and the view to 3D
    labels._slice_dims([0, 0, 0, 1], ndisplay=3)

    start_point, end_point = labels.get_ray_intersections(
        mouse_event_1.position,
        mouse_event_1.view_direction,
        mouse_event_1.dims_displayed,
    )
    np.testing.assert_allclose(start_point, [0, 11, 5, 1])
    np.testing.assert_allclose(end_point, [20, 11, 5, 1])


def test_labels_state_update():
    """Test that a labels layer can be updated from the output of its
    _get_state() method
    """
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    state = layer._get_state()
    for k, v in state.items():
        setattr(layer, k, v)


def test_is_default_color():
    """Test labels layer default color for None and background

    Previously, setting color to just default values would
    change color mode to DIRECT and display a black layer.
    This test ensures `is_default_color` is
    correctly checking against layer defaults, and `color_mode`
    is only changed when appropriate.

    See
        - https://github.com/napari/napari/issues/2479
        - https://github.com/napari/napari/issues/2953
    """
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)

    # layer gets instantiated with defaults
    current_color = layer.color
    assert layer._is_default_colors(current_color)

    # setting color to default colors doesn't update color mode
    layer.color = current_color
    assert layer.color_mode == 'auto'

    # new colors are not default
    new_color = {0: 'white', 1: 'red', 3: 'green'}
    assert not layer._is_default_colors(new_color)
    # setting the color with non-default colors updates color mode
    layer.color = new_color
    assert layer.color_mode == 'direct'


def test_large_labels_direct_color():
    """Make sure direct color works with large label ranges"""
    data = np.array([[0, 1], [2**16, 2**20]], dtype=np.uint32)
    colors = {1: 'white', 2**16: 'green', 2**20: 'magenta'}
    layer = Labels(data)
    layer.color = colors

    assert layer.color_mode == 'direct'
    np.testing.assert_allclose(layer.get_color(2**20), [1.0, 0.0, 1.0, 1.0])


def test_invalidate_cache_when_change_color_mode():
    """Checks if the cache is invalidated when color mode is changed."""
    data = np.zeros((4, 10), dtype=np.int32)
    data[1, :] = np.arange(0, 10)

    layer = Labels(data)
    layer.selected_label = 0
    gt_auto = layer._raw_to_displayed(layer._slice.image.raw)
    assert gt_auto.dtype == np.float32

    layer.color_mode = 'direct'
    layer._cached_labels = None
    assert layer._raw_to_displayed(layer._slice.image.raw).dtype == np.float32

    layer.color_mode = 'auto'
    # If the cache is not invalidated, it returns colors for
    # the direct color mode instead of the color for the auto mode
    assert np.allclose(
        layer._raw_to_displayed(layer._slice.image.raw), gt_auto
    )


def test_color_mapping_when_color_is_changed():
    """Checks if the color mapping is computed correctly when the color palette is changed."""

    data = np.zeros((4, 5), dtype=np.int32)
    data[1, :] = np.arange(0, 5)
    layer = Labels(data, color={1: 'green', 2: 'red', 3: 'white'})
    gt_direct_3colors = layer._raw_to_displayed(layer._slice.image.raw)

    layer = Labels(data, color={1: 'green', 2: 'red'})
    assert layer._raw_to_displayed(layer._slice.image.raw).dtype == np.float32
    layer.color = {1: 'green', 2: 'red', 3: 'white'}

    assert np.allclose(
        layer._raw_to_displayed(layer._slice.image.raw), gt_direct_3colors
    )


def test_color_mapping_with_show_selected_label():
    """Checks if the color mapping is computed correctly when show_selected_label is activated."""

    data = np.arange(5, dtype=np.int32)[:, np.newaxis].repeat(5, axis=1)
    layer = Labels(data)
    mapped_colors_all = layer._raw_to_displayed(layer._slice.image.raw).copy()

    layer.selected_label = 1
    layer.show_selected_label = True

    for selected_label in range(1, 5):
        layer.selected_label = selected_label
        label_mask = data == selected_label
        mapped_colors = layer._raw_to_displayed(layer._slice.image.raw)

        assert np.allclose(
            mapped_colors[label_mask], mapped_colors_all[label_mask]
        )
        assert np.allclose(mapped_colors[np.logical_not(label_mask)], 0)

    layer.show_selected_label = False
    assert np.allclose(
        layer._raw_to_displayed(layer._slice.image.raw), mapped_colors_all
    )


def test_color_mapping_when_seed_is_changed():
    """Checks if the color mapping is updated when the color palette seed is changed."""
    np.random.seed(0)
    layer = Labels(np.random.randint(50, size=(10, 10)))
    mapped_colors1 = layer._raw_to_displayed(layer._slice.image.raw)

    layer.new_colormap()
    mapped_colors2 = layer._raw_to_displayed(layer._slice.image.raw)

    assert not np.allclose(mapped_colors1, mapped_colors2)


def test_negative_label():
    """Test negative label values are supported."""
    data = np.random.randint(low=-1, high=20, size=(10, 10))
    original_data = np.copy(data)
    layer = Labels(data)
    layer.selected_label = -1
    layer.brush_size = 3
    layer.paint((5, 5), -1)
    assert np.count_nonzero(layer.data == -1) > np.count_nonzero(
        original_data == -1
    )


def test_negative_label_slicing():
    """Test negative label color doesn't change during slicing."""
    data = np.array([[[0, 1], [-1, -1]], [[100, 100], [-1, -2]]])
    layer = Labels(data)
    assert tuple(layer.get_color(1)) != tuple(layer.get_color(-1))
    layer._slice_dims(point=(1, 0, 0))
    assert tuple(layer.get_color(-1)) != tuple(layer.get_color(100))
    assert tuple(layer.get_color(-2)) != tuple(layer.get_color(100))


@pytest.mark.xfail(
    reason='This is a known bug with the current label color implementation'
)
def test_negative_label_doesnt_flicker():
    data = np.array(
        [
            [[0, 5], [0, 5]],
            [[-1, 5], [-1, 5]],
            [[-1, 6], [-1, 6]],
        ]
    )
    layer = Labels(data)
    layer._slice_dims(point=(1, 0, 0))
    # this is expected to fail: -1 doesn't trigger an index error in
    # layer._all_vals, it instead just wraps to 5, the previous max label.
    assert tuple(layer.get_color(-1)) != tuple(layer.get_color(5))
    minus_one_color_original = tuple(layer.get_color(-1))
    layer.dims_point = (2, 0, 0)
    layer._set_view_slice()
    # this is also expected to fail: when we switch layers, we see the 6
    # label, which causes an index error, which triggers a recalculation of
    # the label colors. Now -1 is seen so it is taken into account in the
    # indexing calculation, and changes color
    assert tuple(layer.get_color(-1)) == minus_one_color_original


def test_get_status_with_custom_index():
    """See https://github.com/napari/napari/issues/3811"""
    data = np.zeros((10, 10), dtype=np.uint8)
    data[2:5, 2:-2] = 1
    data[5:-2, 2:-2] = 2
    layer = Labels(data)
    df = pd.DataFrame(
        {'text1': [1, 3], 'text2': [7, -2], 'index': [1, 2]}, index=[1, 2]
    )
    layer.properties = df
    assert (
        layer.get_status((0, 0))['coordinates'] == ' [0 0]: 0; [No Properties]'
    )
    assert (
        layer.get_status((3, 3))['coordinates']
        == ' [3 3]: 1; text1: 1, text2: 7'
    )
    assert (
        layer.get_status((6, 6))['coordinates']
        == ' [6 6]: 2; text1: 3, text2: -2'
    )


def test_labels_features_event():
    event_emitted = False

    def on_event():
        nonlocal event_emitted
        event_emitted = True

    layer = Labels(np.zeros((4, 5), dtype=np.uint8))
    layer.events.features.connect(on_event)

    layer.features = {'some_feature': []}

    assert event_emitted


class TestLabels:
    @staticmethod
    def get_objects():
        return [(Labels(np.zeros((10, 10), dtype=np.uint8)))]

    def test_events_defined(self, event_define_check, obj):
        event_define_check(
            obj,
            {"seed", "num_colors", "color"},
        )
