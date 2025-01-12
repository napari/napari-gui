from copy import copy
from itertools import cycle, islice
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from psygnal.containers import Selection
from vispy.color import get_colormap

from napari._pydantic_compat import ValidationError
from napari._tests.utils import (
    assert_colors_equal,
    assert_layer_state_equal,
    check_layer_world_data_extent,
)
from napari.components.dims import Dims
from napari.layers import Points
from napari.layers.base._base_constants import ActionType
from napari.layers.points._points_constants import Mode
from napari.layers.points._points_utils import points_to_squares
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice
from napari.layers.utils._text_constants import Anchor
from napari.layers.utils.color_encoding import ConstantColorEncoding
from napari.layers.utils.color_manager import ColorProperties
from napari.utils._test_utils import (
    validate_all_params_in_docstring,
    validate_docstring_parent_class_consistency,
    validate_kwargs_sorted,
)
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.transforms import CompositeAffine


def _make_cycled_properties(values, length):
    """Helper function to make property values

    Parameters
    ----------
    values
        The values to be cycled.
    length : int
        The length of the resulting property array

    Returns
    -------
    cycled_properties : np.ndarray
        The property array comprising the cycled values.
    """
    cycled_properties = np.array(list(islice(cycle(values), 0, length)))
    return cycled_properties


def test_empty_points():
    pts = Points()
    assert pts.data.shape == (0, 2)
    assert pts.ndim == 2


def test_3d_empty_points():
    pts = Points(np.empty((0, 3)))
    assert pts.ndim == 3


def test_empty_points_with_features():
    """See the following for the issues this covers:
    https://github.com/napari/napari/issues/5632
    https://github.com/napari/napari/issues/5634
    """
    points = Points(
        features={'a': np.empty(0, int)},
        feature_defaults={'a': 0},
        face_color='a',
        face_color_cycle=list('rgb'),
    )

    points.add([0, 0])
    points.feature_defaults['a'] = 1
    points.add([50, 50])
    points.feature_defaults = {'a': 2}
    points.add([100, 100])

    assert_colors_equal(points.face_color, list('rgb'))


def test_empty_points_with_properties():
    """Test instantiating an empty Points layer with properties

    See: https://github.com/napari/napari/pull/1069
    """
    properties = {
        'label': np.array(['label1', 'label2']),
        'cont_prop': np.array([0], dtype=float),
    }
    pts = Points(property_choices=properties)
    current_props = {k: v[0] for k, v in properties.items()}
    np.testing.assert_equal(pts.current_properties, current_props)

    # verify the property datatype is correct
    assert pts.properties['cont_prop'].dtype == float

    # add two points and verify the default property was applied
    pts.add([10, 10])
    pts.add([20, 20])
    props = {
        'label': np.array(['label1', 'label1']),
        'cont_prop': np.array([0, 0], dtype=float),
    }
    np.testing.assert_equal(pts.properties, props)


def test_empty_points_with_properties_list():
    """Test instantiating an empty Points layer with properties
    stored in a list

    See: https://github.com/napari/napari/pull/1069
    """
    properties = {'label': ['label1', 'label2'], 'cont_prop': [0]}
    pts = Points(property_choices=properties)
    current_props = {k: np.asarray(v[0]) for k, v in properties.items()}
    np.testing.assert_equal(pts.current_properties, current_props)

    # add two points and verify the default property was applied
    pts.add([10, 10])
    pts.add([20, 20])
    props = {
        'label': np.array(['label1', 'label1']),
        'cont_prop': np.array([0, 0], dtype=float),
    }
    np.testing.assert_equal(pts.properties, props)


def test_empty_layer_with_face_colormap():
    """Test creating an empty layer where the face color is a colormap
    See: https://github.com/napari/napari/pull/1069
    """
    default_properties = {'point_type': np.array([1.5], dtype=float)}
    layer = Points(
        property_choices=default_properties,
        face_color='point_type',
        face_colormap='gray',
    )

    assert layer.face_color_mode == 'colormap'

    # verify the current_face_color is correct
    face_color = np.array([1, 1, 1, 1])
    np.testing.assert_allclose(layer._face.current_color, face_color)


def test_empty_layer_with_border_colormap():
    """Test creating an empty layer where the face color is a colormap
    See: https://github.com/napari/napari/pull/1069
    """
    default_properties = {'point_type': np.array([1.5], dtype=float)}
    layer = Points(
        property_choices=default_properties,
        border_color='point_type',
        border_colormap='gray',
    )

    assert layer.border_color_mode == 'colormap'

    # verify the current_face_color is correct
    border_color = np.array([1, 1, 1, 1])
    np.testing.assert_allclose(layer._border.current_color, border_color)


@pytest.mark.parametrize('feature_name', ['border', 'face'])
def test_set_current_properties_on_empty_layer_with_color_cycle(feature_name):
    """Test setting current_properties an empty layer where the face/border color
    is a color cycle.

    See: https://github.com/napari/napari/pull/3110
    """
    default_properties = {'annotation': np.array(['tail', 'nose', 'paw'])}
    color_cycle = [[0, 1, 0, 1], [1, 0, 1, 1]]
    color_parameters = {
        'colors': 'annotation',
        'categorical_colormap': color_cycle,
        'mode': 'cycle',
    }
    color_name = f'{feature_name}_color'
    points_kwargs = {
        'property_choices': default_properties,
        color_name: color_parameters,
    }
    layer = Points(**points_kwargs)

    color_mode = getattr(layer, f'{feature_name}_color_mode')
    assert color_mode == 'cycle'
    layer.current_properties = {'annotation': np.array(['paw'])}

    layer.add([10, 10])
    colors = getattr(layer, color_name)
    np.testing.assert_allclose(colors, [color_cycle[1]])
    assert len(layer.data) == 1
    cm = getattr(layer, f'_{feature_name}')
    assert cm.color_properties.current_value == 'paw'


def test_empty_layer_with_text_properties():
    """Test initializing an empty layer with text defined"""
    default_properties = {'point_type': np.array([1.5], dtype=float)}
    text_kwargs = {'string': 'point_type', 'color': 'red'}
    layer = Points(
        property_choices=default_properties,
        text=text_kwargs,
    )
    assert layer.text.values.size == 0
    np.testing.assert_allclose(layer.text.color.constant, [1, 0, 0, 1])

    # add a point and check that the appropriate text value was added
    layer.add([1, 1])
    np.testing.assert_equal(layer.text.values, ['1.5'])
    np.testing.assert_allclose(layer.text.color.constant, [1, 0, 0, 1])


def test_empty_layer_with_text_formatted():
    """Test initializing an empty layer with text defined"""
    default_properties = {'point_type': np.array([1.5], dtype=float)}
    layer = Points(
        property_choices=default_properties,
        text='point_type: {point_type:.2f}',
    )
    assert layer.text.values.size == 0

    # add a point and check that the appropriate text value was added
    layer.add([1, 1])
    np.testing.assert_equal(layer.text.values, ['point_type: 1.50'])


def test_random_points():
    """Test instantiating Points layer with random 2D data."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == shape[1]
    assert layer._view_data.ndim == 2
    assert len(layer.data) == 10
    assert len(layer.selected_data) == 0


def test_integer_points():
    """Test instantiating Points layer with integer data."""
    shape = (10, 2)
    np.random.seed(0)
    data = np.random.randint(20, size=shape)
    layer = Points(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == shape[1]
    assert layer._view_data.ndim == 2
    assert len(layer.data) == 10


def test_negative_points():
    """Test instantiating Points layer with negative data."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape) - 10
    layer = Points(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == shape[1]
    assert layer._view_data.ndim == 2
    assert len(layer.data) == 10


def test_empty_points_array():
    """Test instantiating Points layer with empty array."""
    shape = (0, 2)
    data = np.empty(shape)
    layer = Points(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == shape[1]
    assert layer._view_data.ndim == 2
    assert len(layer.data) == 0


def test_3D_points():
    """Test instantiating Points layer with random 3D data."""
    shape = (10, 3)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == shape[1]
    assert layer._view_data.ndim == 2
    assert len(layer.data) == 10


def test_single_point_extent():
    """Test extent of a single 3D point at the origin."""
    shape = (1, 3)
    data = np.zeros(shape)
    layer = Points(data)
    assert np.array_equal(layer.extent.data, np.zeros((2, 3)))
    assert np.array_equal(layer.extent.world, np.zeros((2, 3)))
    assert np.array_equal(layer.extent.step, np.ones(3))


def test_4D_points():
    """Test instantiating Points layer with random 4D data."""
    shape = (10, 4)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == shape[1]
    assert layer._view_data.ndim == 2
    assert len(layer.data) == 10


def test_changing_points():
    """Test changing Points data."""
    shape_a = (10, 2)
    shape_b = (20, 2)
    np.random.seed(0)
    data_a = 20 * np.random.random(shape_a)
    data_b = 20 * np.random.random(shape_b)
    layer = Points(data_a)
    layer.data = data_b
    assert np.array_equal(layer.data, data_b)
    assert layer.ndim == shape_b[1]
    assert layer._view_data.ndim == 2
    assert len(layer.data) == 20


def test_selecting_points():
    """Test selecting points."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    layer.mode = 'select'
    data_to_select = {1, 2}
    layer.selected_data = data_to_select
    assert layer.selected_data == data_to_select

    # test switching to 3D
    layer._slice_dims(Dims(ndisplay=3))
    assert layer.selected_data == data_to_select

    # select different points while in 3D mode
    other_data_to_select = {0}
    layer.selected_data = other_data_to_select
    assert layer.selected_data == other_data_to_select

    # selection should persist when going back to 2D mode
    layer._slice_dims(Dims(ndisplay=2))
    assert layer.selected_data == other_data_to_select

    # selection should persist when switching between between select and pan_zoom
    layer.mode = 'pan_zoom'
    assert layer.selected_data == other_data_to_select
    layer.mode = 'select'
    assert layer.selected_data == other_data_to_select

    # add mode should clear the selection
    layer.mode = 'add'
    assert layer.selected_data == set()


def test_adding_points():
    """Test adding Points data."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert len(layer.data) == 10

    coord = [20, 20]
    layer.add(coord)
    assert len(layer.data) == 11
    assert np.array_equal(layer.data[10], coord)
    # the added point should be selected
    assert layer.selected_data == {10}

    # test adding multiple points
    coords = [[10, 10], [15, 15]]
    layer.add(coords)
    assert len(layer.data) == 13
    assert np.array_equal(layer.data[11:, :], coords)
    assert layer.selected_data == {11, 12}

    # test that the last added points can be deleted
    layer.remove_selected()
    np.testing.assert_equal(layer.data, np.vstack((data, coord)))


def test_points_selection_with_setter():
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)

    coords = [[10, 10], [15, 15]]
    layer.data = np.append(layer.data, np.atleast_2d(coords), axis=0)
    assert len(layer.data) == 12
    assert layer.selected_data == set()


def test_adding_points_to_empty():
    """Test adding Points data to empty."""
    shape = (0, 2)
    data = np.empty(shape)
    layer = Points(data)
    assert len(layer.data) == 0

    coord = [20, 20]
    layer.add(coord)
    assert len(layer.data) == 1
    assert np.array_equal(layer.data[0], coord)
    assert layer.selected_data == {0}


def test_removing_selected_points():
    """Test selecting points."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)

    # With nothing selected no points should be removed
    layer.remove_selected()
    assert len(layer.data) == shape[0]

    # Select two points and remove them
    layer.selected_data = {0, 3}
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 2
    assert len(layer.selected_data) == 0
    keep = [1, 2, *range(4, 10)]
    assert np.array_equal(layer.data, data[keep])
    assert layer._value is None

    # Select another point and remove it
    layer.selected_data = {4}
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 3


def test_deleting_selected_value_changes():
    """Test deleting selected points appropriately sets self._value"""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)

    # removing with self._value selected resets self._value to None
    layer._value = 1
    layer.selected_data = {1, 2}
    layer.remove_selected()
    assert layer._value is None

    # removing with self._value outside selection doesn't change self._value
    layer._value = 3
    layer.selected_data = {4}
    layer.remove_selected()
    assert layer._value == 3


def test_remove_selected_updates_value():
    """Test that removing a point that is not layer._value
    updates the index to account for the removed data.
    """
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)

    old_data = layer.data
    layer.events.data = Mock()
    # set the value
    layer._value = 3
    layer._value_stored = 3

    selection = {0, 5, 6, 7}
    layer.selected_data = selection
    layer.remove_selected()
    assert layer.events.data.call_args_list[0][1] == {
        'value': old_data,
        'action': ActionType.REMOVING,
        'data_indices': tuple(selection),
        'vertex_indices': ((),),
    }
    assert layer.events.data.call_args[1] == {
        'value': layer.data,
        'action': ActionType.REMOVED,
        'data_indices': tuple(selection),
        'vertex_indices': ((),),
    }
    assert layer._value == 2


def test_remove_selected_removes_corresponding_attributes():
    """Test that removing points at specific indices also removes any per-point
    attribute at the same index"""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    size = np.random.rand(shape[0])
    symbol = np.random.choice(['o', 's'], shape[0])
    color = np.random.rand(shape[0], 4)
    feature = np.random.rand(shape[0])
    shown = np.random.randint(2, size=shape[0]).astype(bool)
    text = 'feature'

    layer = Points(
        data,
        size=size,
        border_width=size,
        symbol=symbol,
        features={'feature': feature},
        face_color=color,
        border_color=color,
        text=text,
        shown=shown,
    )

    layer_expected = Points(
        data[1:],
        size=size[1:],
        symbol=symbol[1:],
        border_width=size[1:],
        features={'feature': feature[1:]},
        feature_defaults={'feature': feature[0]},
        face_color=color[1:],
        border_color=color[1:],
        text=text,  # computed from feature
        shown=shown[1:],
    )

    layer.selected_data = {0}
    layer.remove_selected()

    state_layer = layer._get_state()
    state_expected = layer_expected._get_state()

    assert_layer_state_equal(state_layer, state_expected)


def test_move():
    """Test moving points."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    unmoved = copy(data)
    layer = Points(data)
    layer.events.data = Mock()

    # Move one point relative to an initial drag start location
    layer._move([0], [0, 0])
    layer._move([0], [10, 10])
    layer._drag_start = None
    assert np.array_equal(layer.data[0], unmoved[0] + [10, 10])
    assert np.array_equal(layer.data[1:], unmoved[1:])
    assert layer.events.data.call_args[1] == {
        'value': layer.data,
        'action': ActionType.CHANGED,
        'data_indices': (0,),
        'vertex_indices': ((),),
    }

    # Move two points relative to an initial drag start location
    layer._move([1, 2], [2, 2])
    layer._move([1, 2], np.add([2, 2], [-3, 4]))
    assert layer.events.data.call_args[1] == {
        'value': layer.data,
        'action': ActionType.CHANGED,
        'data_indices': (1, 2),
        'vertex_indices': ((),),
    }
    assert np.array_equal(layer.data[1:2], unmoved[1:2] + [-3, 4])


def test_changing_modes():
    """Test changing modes."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.mode == 'pan_zoom'
    assert layer.mouse_pan is True

    layer.mode = 'add'
    assert layer.mode == 'add'

    layer.mode = 'select'
    assert layer.mode == 'select'
    assert layer.mouse_pan is False

    layer.mode = 'pan_zoom'
    assert layer.mode == 'pan_zoom'
    assert layer.mouse_pan is True

    with pytest.raises(ValueError, match='not a valid Mode'):
        layer.mode = 'not_a_mode'


def test_name():
    """Test setting layer name."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    layer = Points(data)
    assert layer.name == 'Points'

    layer = Points(data, name='random')
    assert layer.name == 'random'

    layer.name = 'pts'
    assert layer.name == 'pts'


def test_visibility():
    """Test setting layer visibility."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    layer = Points(data)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer = Points(data, visible=False)
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_opacity():
    """Test setting layer opacity."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    layer = Points(data)
    assert layer.opacity == 1.0

    layer.opacity = 0.5
    assert layer.opacity == 0.5

    layer = Points(data, opacity=0.6)
    assert layer.opacity == 0.6

    layer.opacity = 0.3
    assert layer.opacity == 0.3


def test_blending():
    """Test setting layer blending."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    layer = Points(data)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Points(data, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


def test_symbol():
    """Test setting symbol."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert np.array_equiv(layer.symbol, 'disc')

    layer.symbol = 'cross'
    assert np.array_equiv(layer.symbol, 'cross')

    symbol = ['o', 's'] * 5
    expected = ['disc', 'square'] * 5
    layer.symbol = symbol
    assert np.array_equal(layer.symbol, expected)

    with pytest.raises(
        ValueError, match='Symbol array must be the same length as data'
    ):
        layer.symbol = symbol[1:5]

    layer = Points(data, symbol='star')
    assert np.array_equiv(layer.symbol, 'star')


properties_array = {'point_type': _make_cycled_properties(['A', 'B'], 10)}
properties_list = {'point_type': list(_make_cycled_properties(['A', 'B'], 10))}


@pytest.mark.parametrize('properties', [properties_array, properties_list])
def test_properties(properties):
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data, properties=copy(properties))
    np.testing.assert_equal(layer.properties, properties)

    current_prop = {'point_type': np.array(['B'])}
    assert layer.current_properties == current_prop

    # test removing points
    layer.selected_data = {0, 1}
    layer.remove_selected()
    remove_properties = properties['point_type'][2::]
    assert len(layer.properties['point_type']) == (shape[0] - 2)
    assert np.array_equal(layer.properties['point_type'], remove_properties)

    # test selection of properties
    layer.selected_data = {0}
    selected_annotation = layer.current_properties['point_type']
    assert len(selected_annotation) == 1
    assert selected_annotation[0] == 'A'

    # test adding points with properties
    layer.add([10, 10])
    add_annotations = np.concatenate((remove_properties, ['A']), axis=0)
    assert np.array_equal(layer.properties['point_type'], add_annotations)

    # test copy/paste
    layer.selected_data = {0, 1}
    layer._copy_data()
    assert np.array_equal(
        layer._clipboard['features']['point_type'], ['A', 'B']
    )

    layer._paste_data()
    paste_annotations = np.concatenate((add_annotations, ['A', 'B']), axis=0)
    assert np.array_equal(layer.properties['point_type'], paste_annotations)

    assert layer.get_status(data[0])['coordinates'].endswith('point_type: B')
    assert layer.get_status(data[1])['coordinates'].endswith('point_type: A')


@pytest.mark.parametrize('attribute', ['border', 'face'])
def test_adding_properties(attribute):
    """Test adding properties to an existing layer"""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)

    # add properties
    properties = {'point_type': _make_cycled_properties(['A', 'B'], shape[0])}
    layer.properties = properties
    np.testing.assert_equal(layer.properties, properties)

    # add properties as a dataframe
    properties_df = pd.DataFrame(properties)
    layer.properties = properties_df
    np.testing.assert_equal(layer.properties, properties)

    # add properties as a dictionary with list values
    properties_list = {
        'point_type': list(_make_cycled_properties(['A', 'B'], shape[0]))
    }
    layer.properties = properties_list
    assert isinstance(layer.properties['point_type'], np.ndarray)

    # removing a property that was the _*_color_property should give a warning
    color_manager = getattr(layer, f'_{attribute}')
    color_manager.color_properties = {
        'name': 'point_type',
        'values': np.empty(0),
        'current_value': 'A',
    }
    properties_2 = {
        'not_point_type': _make_cycled_properties(['A', 'B'], shape[0])
    }
    with pytest.warns(RuntimeWarning):
        layer.properties = properties_2


def test_properties_dataframe():
    """Test if properties can be provided as a DataFrame"""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'point_type': _make_cycled_properties(['A', 'B'], shape[0])}
    properties_df = pd.DataFrame(properties)
    properties_df = properties_df.astype(properties['point_type'].dtype)
    layer = Points(data, properties=properties_df)
    np.testing.assert_equal(layer.properties, properties)


def test_add_points_with_properties_as_list():
    # test adding points initialized with properties as list
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {
        'point_type': list(_make_cycled_properties(['A', 'B'], shape[0]))
    }
    layer = Points(data, properties=copy(properties))

    coord = [18, 18]
    layer.add(coord)
    new_prop = {'point_type': np.append(properties['point_type'], 'B')}
    np.testing.assert_equal(layer.properties, new_prop)


def test_updating_points_properties():
    # test adding points initialized with properties
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'point_type': _make_cycled_properties(['A', 'B'], shape[0])}
    layer = Points(data, properties=copy(properties))

    layer.mode = 'select'
    layer.selected_data = [len(data) - 1]
    layer.current_properties = {'point_type': np.array(['A'])}

    updated_properties = properties
    updated_properties['point_type'][-1] = 'A'
    np.testing.assert_equal(layer.properties, updated_properties)


def test_setting_current_properties():
    shape = (2, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {
        'annotation': ['paw', 'leg'],
        'confidence': [0.5, 0.75],
        'annotator': ['jane', 'ash'],
        'model': ['worst', 'best'],
    }
    layer = Points(data, properties=copy(properties))
    current_properties = {
        'annotation': ['leg'],
        'confidence': 1,
        'annotator': 'ash',
        'model': np.array(['best']),
    }
    layer.current_properties = current_properties

    expected_current_properties = {
        'annotation': np.array(['leg']),
        'confidence': np.array([1]),
        'annotator': np.array(['ash']),
        'model': np.array(['best']),
    }

    coerced_current_properties = layer.current_properties
    for k in coerced_current_properties:
        value = coerced_current_properties[k]
        assert isinstance(value, np.ndarray)
        np.testing.assert_equal(value, expected_current_properties[k])


properties_array = {'point_type': _make_cycled_properties(['A', 'B'], 10)}
properties_list = {'point_type': list(_make_cycled_properties(['A', 'B'], 10))}


@pytest.mark.parametrize('properties', [properties_array, properties_list])
def test_text_from_property_value(properties):
    """Test setting text from a property value"""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data, properties=copy(properties), text='point_type')

    np.testing.assert_equal(layer.text.values, properties['point_type'])


@pytest.mark.parametrize('properties', [properties_array, properties_list])
def test_text_from_property_fstring(properties):
    """Test setting text with an f-string from the property value"""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(
        data, properties=copy(properties), text='type: {point_type}'
    )

    expected_text = ['type: ' + v for v in properties['point_type']]
    np.testing.assert_equal(layer.text.values, expected_text)

    # test updating the text
    layer.text = 'type-ish: {point_type}'
    expected_text_2 = ['type-ish: ' + v for v in properties['point_type']]
    np.testing.assert_equal(layer.text.values, expected_text_2)

    # copy/paste
    layer.selected_data = {0}
    layer._copy_data()
    layer._paste_data()
    expected_text_3 = [*expected_text_2, 'type-ish: A']
    np.testing.assert_equal(layer.text.values, expected_text_3)

    # add point
    layer.selected_data = {0}
    new_shape = np.random.random((1, 2))
    layer.add(new_shape)
    expected_text_4 = [*expected_text_3, 'type-ish: A']
    np.testing.assert_equal(layer.text.values, expected_text_4)


@pytest.mark.parametrize('properties', [properties_array, properties_list])
def test_set_text_with_kwarg_dict(properties):
    text_kwargs = {
        'string': 'type: {point_type}',
        'color': ConstantColorEncoding(constant=[0, 0, 0, 1]),
        'rotation': 10,
        'translation': [5, 5],
        'anchor': Anchor.UPPER_LEFT,
        'size': 10,
        'visible': True,
    }
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data, properties=copy(properties), text=text_kwargs)

    expected_text = ['type: ' + v for v in properties['point_type']]
    np.testing.assert_equal(layer.text.values, expected_text)

    for property_, value in text_kwargs.items():
        if property_ == 'string':
            continue
        layer_value = getattr(layer._text, property_)
        np.testing.assert_equal(layer_value, value)


@pytest.mark.parametrize('properties', [properties_array, properties_list])
def test_text_error(properties):
    """creating a layer with text as the wrong type should raise an error"""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    # try adding text as the wrong type
    with pytest.raises(ValidationError):
        Points(data, properties=copy(properties), text=123)


def test_select_properties_object_dtype():
    """selecting points when they have a property of object dtype should not fail"""
    # pandas uses object as dtype for strings by default
    properties = pd.DataFrame({'color': ['red', 'green']})
    pl = Points(np.ones((2, 2)), properties=properties)
    selection = {0, 1}
    pl.selected_data = selection
    assert pl.selected_data == selection


def test_select_properties_unsortable():
    """selecting multiple points when they have properties that cannot be sorted should not fail

    see https://github.com/napari/napari/issues/5174
    """
    properties = pd.DataFrame({'unsortable': [{}, {}]})
    pl = Points(np.ones((2, 2)), properties=properties)
    selection = {0, 1}
    pl.selected_data = selection
    assert pl.selected_data == selection


def test_refresh_text():
    """Test refreshing the text after setting new properties"""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'point_type': ['A'] * shape[0]}
    layer = Points(data, properties=copy(properties), text='point_type')

    new_properties = {'point_type': ['B'] * shape[0]}
    layer.properties = new_properties
    np.testing.assert_equal(layer.text.values, new_properties['point_type'])


def test_points_errors():
    shape = (3, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)

    annotations = {'point_type': np.array(['A', 'B'])}

    # try adding properties with the wrong number of properties
    with pytest.raises(
        ValueError, match='(does not match length)|(indices imply)'
    ):
        Points(data, properties=copy(annotations))


def test_border_width():
    """Test setting border width."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    np.testing.assert_array_equal(layer.border_width, 0.05)

    layer.border_width = 0.5
    np.testing.assert_array_equal(layer.border_width, 0.5)

    # fail outside of range 0, 1 if relative is enabled (default)
    with pytest.raises(ValueError, match='must be between 0 and 1'):
        layer.border_width = 2

    layer.border_width_is_relative = False
    layer.border_width = 2
    np.testing.assert_array_equal(layer.border_width, 2)

    # fail if we try to come back again
    with pytest.raises(ValueError, match='between 0 and 1'):
        layer.border_width_is_relative = True

    # all should work on instantiation too
    layer = Points(data, border_width=3, border_width_is_relative=False)
    np.testing.assert_array_equal(layer.border_width, 3)
    assert layer.border_width_is_relative is False
    with pytest.raises(ValueError, match='must be > 0'):
        layer.border_width = -2


@pytest.mark.parametrize(
    'border_width',
    [1, float(1), np.array([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5]],
)
def test_border_width_types(border_width):
    """Test border_width dtypes with valid values"""
    shape = (5, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(
        data, border_width=border_width, border_width_is_relative=False
    )
    np.testing.assert_array_equal(layer.border_width, border_width)


@pytest.mark.parametrize(
    'border_width',
    [-1, float(-1), np.array([-1, 2, 3, 4, 5]), [-1, 2, 3, 4, 5]],
)
def test_border_width_types_negative(border_width):
    """Test negative values in all border_width dtypes"""
    shape = (5, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    with pytest.raises(ValueError, match='must be > 0'):
        Points(data, border_width=border_width, border_width_is_relative=False)


def test_out_of_slice_display():
    """Test setting out_of_slice_display flag for 2D and 4D data."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.out_of_slice_display is False

    layer.out_of_slice_display = True
    assert layer.out_of_slice_display is True

    layer = Points(data, out_of_slice_display=True)
    assert layer.out_of_slice_display is True

    shape = (10, 4)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.out_of_slice_display is False

    layer.out_of_slice_display = True
    assert layer.out_of_slice_display is True

    layer = Points(data, out_of_slice_display=True)
    assert layer.out_of_slice_display is True


@pytest.mark.parametrize('attribute', ['border', 'face'])
def test_switch_color_mode(attribute):
    """Test switching between color modes"""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    # create a continuous property with a known value in the last element
    continuous_prop = np.random.random((shape[0],))
    continuous_prop[-1] = 1
    properties = {
        'point_truthiness': continuous_prop,
        'point_type': _make_cycled_properties(['A', 'B'], shape[0]),
    }
    initial_color = [1, 0, 0, 1]
    color_cycle = ['red', 'blue']
    color_kwarg = f'{attribute}_color'
    colormap_kwarg = f'{attribute}_colormap'
    color_cycle_kwarg = f'{attribute}_color_cycle'
    args = {
        color_kwarg: initial_color,
        colormap_kwarg: 'gray',
        color_cycle_kwarg: color_cycle,
    }
    layer = Points(data, properties=properties, **args)

    layer_color_mode = getattr(layer, f'{attribute}_color_mode')
    layer_color = getattr(layer, f'{attribute}_color')
    assert layer_color_mode == 'direct'
    np.testing.assert_allclose(
        layer_color, np.repeat([initial_color], shape[0], axis=0)
    )

    # there should not be an border_color_property
    color_manager = getattr(layer, f'_{attribute}')
    color_property = color_manager.color_properties
    assert color_property is None

    # transitioning to colormap should raise a warning
    # because there isn't an border color property yet and
    # the first property in points.properties is being automatically selected
    with pytest.warns(UserWarning):
        setattr(layer, f'{attribute}_color_mode', 'colormap')
    color_manager = getattr(layer, f'_{attribute}')
    color_property_name = color_manager.color_properties.name
    assert color_property_name == next(iter(properties))
    layer_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(layer_color[-1], [1, 1, 1, 1])

    # switch to color cycle
    setattr(layer, f'{attribute}_color_mode', 'cycle')
    setattr(layer, f'{attribute}_color', 'point_type')
    color = getattr(layer, f'{attribute}_color')
    layer_color = transform_color(color_cycle * int(shape[0] / 2))
    np.testing.assert_allclose(color, layer_color)

    # switch back to direct, border_colors shouldn't change
    setattr(layer, f'{attribute}_color_mode', 'direct')
    new_border_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(new_border_color, color)


@pytest.mark.parametrize('attribute', ['border', 'face'])
def test_colormap_without_properties(attribute):
    """Setting the colormode to colormap should raise an exception"""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)

    with pytest.raises(ValueError, match='must be a valid Points.properties'):
        setattr(layer, f'{attribute}_color_mode', 'colormap')


@pytest.mark.parametrize('attribute', ['border', 'face'])
def test_colormap_with_categorical_properties(attribute):
    """Setting the colormode to colormap should raise an exception"""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'point_type': _make_cycled_properties(['A', 'B'], shape[0])}
    layer = Points(data, properties=properties)

    with pytest.raises(TypeError), pytest.warns(UserWarning):
        setattr(layer, f'{attribute}_color_mode', 'colormap')


@pytest.mark.parametrize('attribute', ['border', 'face'])
def test_add_colormap(attribute):
    """Test  directly adding a vispy Colormap object"""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    annotations = {'point_type': _make_cycled_properties([0, 1.5], shape[0])}
    color_kwarg = f'{attribute}_color'
    colormap_kwarg = f'{attribute}_colormap'
    args = {color_kwarg: 'point_type', colormap_kwarg: 'viridis'}
    layer = Points(data, properties=annotations, **args)

    setattr(layer, f'{attribute}_colormap', get_colormap('gray'))
    layer_colormap = getattr(layer, f'{attribute}_colormap')
    assert 'unnamed colormap' in layer_colormap.name


@pytest.mark.parametrize('attribute', ['border', 'face'])
def test_add_point_direct(attribute: str):
    """Test adding points to layer directly"""
    layer = Points()
    old_data = layer.data
    assert len(getattr(layer, f'{attribute}_color')) == 0

    layer.events.data = Mock()
    setattr(layer, f'current_{attribute}_color', 'red')
    coord = [18, 18]

    layer.add(coord)
    assert layer.events.data.call_args_list[0][1] == {
        'value': old_data,
        'action': ActionType.ADDING,
        'data_indices': (-1,),
        'vertex_indices': ((),),
    }
    assert layer.events.data.call_args[1] == {
        'value': layer.data,
        'action': ActionType.ADDED,
        'data_indices': (-1,),
        'vertex_indices': ((),),
    }
    np.testing.assert_allclose(
        [[1, 0, 0, 1]], getattr(layer, f'{attribute}_color')
    )


@pytest.mark.parametrize('attribute', ['border', 'face'])
def test_color_direct(attribute: str):
    """Test setting colors directly"""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer_kwargs = {f'{attribute}_color': 'black'}
    layer = Points(data, **layer_kwargs)
    color_array = transform_color(['black'] * shape[0])
    current_color = getattr(layer, f'current_{attribute}_color')
    layer_color = getattr(layer, f'{attribute}_color')
    assert current_color == 'black'
    assert len(layer.border_color) == shape[0]
    np.testing.assert_allclose(color_array, layer_color)

    # With no data selected changing color has no effect
    setattr(layer, f'current_{attribute}_color', 'blue')
    current_color = getattr(layer, f'current_{attribute}_color')
    assert current_color == 'blue'
    np.testing.assert_allclose(color_array, layer_color)

    # Select data and change border color of selection
    selected_data = {0, 1}
    layer.selected_data = {0, 1}
    current_color = getattr(layer, f'current_{attribute}_color')
    assert current_color == 'black'
    setattr(layer, f'current_{attribute}_color', 'green')
    colorarray_green = transform_color(['green'] * len(layer.selected_data))
    color_array[list(selected_data)] = colorarray_green
    layer_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(color_array, layer_color)

    # Add new point and test its color
    coord = [18, 18]
    layer.selected_data = {}
    setattr(layer, f'current_{attribute}_color', 'blue')
    layer.add(coord)
    color_array = np.vstack([color_array, transform_color('blue')])
    layer_color = getattr(layer, f'{attribute}_color')
    assert len(layer_color) == shape[0] + 1
    np.testing.assert_allclose(color_array, layer_color)

    # Check removing data adjusts colors correctly
    layer.selected_data = {0, 2}
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1

    layer_color = getattr(layer, f'{attribute}_color')
    assert len(layer_color) == shape[0] - 1
    np.testing.assert_allclose(
        layer_color,
        np.vstack((color_array[1], color_array[3:])),
    )


color_cycle_str = ['red', 'blue']
color_cycle_rgb = [[1, 0, 0], [0, 0, 1]]
color_cycle_rgba = [[1, 0, 0, 1], [0, 0, 1, 1]]


@pytest.mark.parametrize('attribute', ['border', 'face'])
@pytest.mark.parametrize(
    'color_cycle',
    [color_cycle_str, color_cycle_rgb, color_cycle_rgba],
)
def test_color_cycle(attribute, color_cycle):
    """Test setting border/face color with a color cycle list"""
    # create Points using list color cycle
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'point_type': _make_cycled_properties(['A', 'B'], shape[0])}
    points_kwargs = {
        'properties': properties,
        f'{attribute}_color': 'point_type',
        f'{attribute}_color_cycle': color_cycle,
    }
    layer = Points(data, **points_kwargs)

    np.testing.assert_equal(layer.properties, properties)

    color_array = transform_color(
        list(islice(cycle(color_cycle), 0, shape[0]))
    )
    layer_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(layer_color, color_array)

    # Add new point and test its color
    coord = [18, 18]
    layer.selected_data = {0}
    layer.add(coord)
    layer_color = getattr(layer, f'{attribute}_color')
    assert len(layer_color) == shape[0] + 1
    np.testing.assert_allclose(
        layer_color,
        np.vstack((color_array, transform_color('red'))),
    )

    # Check removing data adjusts colors correctly
    layer.selected_data = {0, 2}
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1

    layer_color = getattr(layer, f'{attribute}_color')
    assert len(layer_color) == shape[0] - 1
    np.testing.assert_allclose(
        layer_color,
        np.vstack((color_array[1], color_array[3:], transform_color('red'))),
    )

    # test adding a point with a new property value
    layer.selected_data = {}
    current_properties = layer.current_properties
    current_properties['point_type'] = np.array(['new'])
    layer.current_properties = current_properties
    layer.add([10, 10])
    color_manager = getattr(layer, f'_{attribute}')
    color_cycle_map = color_manager.categorical_colormap.colormap

    assert 'new' in color_cycle_map
    np.testing.assert_allclose(
        color_cycle_map['new'], np.squeeze(transform_color(color_cycle[0]))
    )


@pytest.mark.parametrize('attribute', ['border', 'face'])
def test_color_cycle_dict(attribute):
    """Test setting border/face color with a color cycle dict"""
    data = np.array([[0, 0], [100, 0], [0, 100]])
    properties = {'my_colors': [2, 6, 3]}
    points_kwargs = {
        'properties': properties,
        f'{attribute}_color': 'my_colors',
        f'{attribute}_color_cycle': {1: 'green', 2: 'red', 3: 'blue'},
    }
    layer = Points(data, **points_kwargs)

    color_manager = getattr(layer, f'_{attribute}')
    color_cycle_map = color_manager.categorical_colormap.colormap
    np.testing.assert_allclose(color_cycle_map[2], [1, 0, 0, 1])  # 2 is red
    np.testing.assert_allclose(color_cycle_map[3], [0, 0, 1, 1])  # 3 is blue
    np.testing.assert_allclose(color_cycle_map[6], [1, 1, 1, 1])  # 6 is white


@pytest.mark.parametrize('attribute', ['border', 'face'])
def test_add_color_cycle_to_empty_layer(attribute):
    """Test adding a point to an empty layer when border/face color is a color cycle

    See: https://github.com/napari/napari/pull/1069
    """
    default_properties = {'point_type': np.array(['A'])}
    color_cycle = ['red', 'blue']
    points_kwargs = {
        'property_choices': default_properties,
        f'{attribute}_color': 'point_type',
        f'{attribute}_color_cycle': color_cycle,
    }
    layer = Points(**points_kwargs)

    # verify the current_border_color is correct
    expected_color = transform_color(color_cycle[0])[0]
    color_manager = getattr(layer, f'_{attribute}')
    current_color = color_manager.current_color
    np.testing.assert_allclose(current_color, expected_color)

    # add a point
    layer.add([10, 10])
    props = {'point_type': np.array(['A'])}
    expected_color = np.array([[1, 0, 0, 1]])
    np.testing.assert_equal(layer.properties, props)
    attribute_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(attribute_color, expected_color)

    # add a point with a new property
    layer.selected_data = []
    layer.current_properties = {'point_type': np.array(['B'])}
    layer.add([12, 12])
    new_color = np.array([0, 0, 1, 1])
    expected_color = np.vstack((expected_color, new_color))
    new_properties = {'point_type': np.array(['A', 'B'])}
    attribute_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(attribute_color, expected_color)
    np.testing.assert_equal(layer.properties, new_properties)


@pytest.mark.parametrize('attribute', ['border', 'face'])
def test_adding_value_color_cycle(attribute):
    """Test that adding values to properties used to set a color cycle
    and then calling Points.refresh_colors() performs the update and adds the
    new value to the face/border_color_cycle_map.

    See: https://github.com/napari/napari/issues/988
    """
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'point_type': _make_cycled_properties(['A', 'B'], shape[0])}
    color_cycle = ['red', 'blue']
    points_kwargs = {
        'properties': properties,
        f'{attribute}_color': 'point_type',
        f'{attribute}_color_cycle': color_cycle,
    }
    layer = Points(data, **points_kwargs)

    # make point 0 point_type C
    props = layer.properties
    point_types = props['point_type']
    point_types[0] = 'C'
    props['point_type'] = point_types
    layer.properties = props

    color_manager = getattr(layer, f'_{attribute}')
    color_cycle_map = color_manager.categorical_colormap.colormap
    color_map_keys = [*color_cycle_map]
    assert 'C' in color_map_keys


@pytest.mark.parametrize('attribute', ['border', 'face'])
def test_color_colormap(attribute):
    """Test setting border/face color with a colormap"""
    # create Points using with a colormap
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'point_type': _make_cycled_properties([0, 1.5], shape[0])}
    points_kwargs = {
        'properties': properties,
        f'{attribute}_color': 'point_type',
        f'{attribute}_colormap': 'gray',
    }
    layer = Points(data, **points_kwargs)

    np.testing.assert_equal(layer.properties, properties)

    color_mode = getattr(layer, f'{attribute}_color_mode')
    assert color_mode == 'colormap'
    color_array = transform_color(['black', 'white'] * int(shape[0] / 2))
    attribute_color = getattr(layer, f'{attribute}_color')
    assert np.array_equal(attribute_color, color_array)

    # change the color cycle - face_color should not change
    setattr(layer, f'{attribute}_color_cycle', ['red', 'blue'])
    attribute_color = getattr(layer, f'{attribute}_color')
    assert np.array_equal(attribute_color, color_array)

    # Add new point and test its color
    coord = [18, 18]
    layer.selected_data = {0}
    layer.add(coord)
    attribute_color = getattr(layer, f'{attribute}_color')
    assert len(attribute_color) == shape[0] + 1
    np.testing.assert_allclose(
        attribute_color,
        np.vstack((color_array, transform_color('black'))),
    )

    # Check removing data adjusts colors correctly
    layer.selected_data = {0, 2}
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    attribute_color = getattr(layer, f'{attribute}_color')
    assert len(attribute_color) == shape[0] - 1
    np.testing.assert_allclose(
        attribute_color,
        np.vstack(
            (
                color_array[1],
                color_array[3:],
                transform_color('black'),
            )
        ),
    )

    # adjust the clims
    setattr(layer, f'{attribute}_contrast_limits', (0, 3))
    attribute_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(attribute_color[-2], [0.5, 0.5, 0.5, 1])

    # change the colormap
    new_colormap = 'viridis'
    setattr(layer, f'{attribute}_colormap', new_colormap)
    attribute_colormap = getattr(layer, f'{attribute}_colormap')
    assert attribute_colormap.name == new_colormap


def test_size():
    """Test setting size with scalar."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.current_size == 10
    assert layer.size.shape == (10,)
    assert np.unique(layer.size)[0] == 10

    # Add a new point, it should get current size
    coord = [17, 17]
    layer.add(coord)
    assert layer.size.shape == (11,)
    assert np.unique(layer.size)[0] == 10

    # Setting size affects newly added points not current points
    layer.current_size = 20
    assert layer.current_size == 20
    assert layer.size.shape == (11,)
    assert np.unique(layer.size)[0] == 10

    # Add new point, should have new size
    coord = [18, 18]
    layer.add(coord)
    assert layer.size.shape == (12,)
    assert np.unique(layer.size[:11])[0] == 10
    assert np.array_equal(layer.size[11], 20)

    # Select data and change size
    layer.selected_data = {0, 1}
    assert layer.current_size == 10
    layer.current_size = 16
    assert layer.size.shape == (12,)
    assert np.unique(layer.size[2:11])[0] == 10
    assert np.unique(layer.size[:2])[0] == 16

    # Select data and size changes
    layer.selected_data = {11}
    assert layer.current_size == 20


@pytest.mark.parametrize('ndim', [2, 3])
def test_size_with_arrays(ndim):
    """Test setting size with arrays."""
    shape = (10, ndim)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    sizes = 5 * np.random.random(10)
    layer.size = sizes
    assert np.array_equal(layer.size, sizes)

    # Un-broadcastable array should raise an exception
    sizes = [5, 5]
    with pytest.raises(ValueError, match='not compatible for broadcasting'):
        layer.size = sizes

    # Create new layer with new size array data
    sizes = 5 * np.random.random(10)
    layer = Points(data, size=sizes)
    assert layer.current_size == 10
    assert layer.size.shape == (10,)
    np.testing.assert_array_equal(layer.size, sizes)

    # Add new point, should have new size
    coord = [18] * ndim
    layer.current_size = 13
    layer.add(coord)
    assert layer.size.shape == (11,)
    np.testing.assert_array_equal(layer.size[:10], sizes[:10])
    assert layer.size[10] == 13

    # Select data and change size
    layer.selected_data = {0, 1}
    # current_size does not change because idx 0 and 1 are different sizes
    assert layer.current_size == 13
    layer.current_size = 16
    assert layer.size.shape == (11,)
    np.testing.assert_array_equal(layer.size[2:10], sizes[2:10])
    np.testing.assert_array_equal(layer.size[:2], 16)

    # check that current size is correctly set if all points are the same size
    layer.selected_data = {10}
    assert layer.current_size == 13
    layer.selected_data = {0, 1}
    assert layer.current_size == 16

    # Check removing data adjusts sizes correctly
    layer.selected_data = {0, 2}
    layer.remove_selected()
    assert len(layer.data) == 9
    assert len(layer.size) == 9
    assert layer.size[0] == 16
    assert layer.size[1] == sizes[3]


def test_copy_and_paste():
    """Test copying and pasting selected points."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    # Clipboard starts empty
    assert layer._clipboard == {}

    # Pasting empty clipboard doesn't change data
    layer._paste_data()
    assert len(layer.data) == 10

    # Copying with nothing selected leave clipboard empty
    layer._copy_data()
    assert layer._clipboard == {}

    # Copying and pasting with two points selected adds to clipboard and data
    layer.selected_data = {0, 1}
    layer._copy_data()
    layer._paste_data()
    assert len(layer._clipboard.keys()) > 0
    assert len(layer.data) == shape[0] + 2
    assert np.array_equal(layer.data[:2], layer.data[-2:])

    # Pasting again adds two more points to data
    layer._paste_data()
    assert len(layer.data) == shape[0] + 4
    assert np.array_equal(layer.data[:2], layer.data[-2:])

    # Unselecting everything and copying and pasting will empty the clipboard
    # and add no new data
    layer.selected_data = {}
    layer._copy_data()
    layer._paste_data()
    assert layer._clipboard == {}
    assert len(layer.data) == shape[0] + 4


def test_value():
    """Test getting the value of the data at the current coordinates."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[-1] = [0, 0]
    layer = Points(data)
    value = layer.get_value((0, 0))
    assert value == 9

    layer.data = layer.data + 20
    value = layer.get_value((0, 0))
    assert value is None


@pytest.mark.parametrize(
    (
        'position',
        'view_direction',
        'dims_displayed',
        'world',
        'scale',
        'expected',
    ),
    [
        ((0, 5, 15, 15), [0, 1, 0, 0], [1, 2, 3], False, (1, 1, 1, 1), 2),
        ((0, 5, 15, 15), [0, -1, 0, 0], [1, 2, 3], False, (1, 1, 1, 1), 0),
        ((0, 5, 0, 0), [0, 1, 0, 0], [1, 2, 3], False, (1, 1, 1, 1), None),
        ((0, 5, 15, 15), [0, 1, 0, 0], [1, 2, 3], True, (1, 1, 2, 1), None),
        ((0, 5, 15, 15), [0, -1, 0, 0], [1, 2, 3], True, (1, 1, 2, 1), None),
        ((0, 5, 30, 15), [0, 1, 0, 0], [1, 2, 3], True, (1, 1, 2, 1), 2),
        ((0, 5, 30, 15), [0, -1, 0, 0], [1, 2, 3], True, (1, 1, 2, 1), 0),
        ((0, 5, 0, 0), [0, 1, 0, 0], [1, 2, 3], True, (1, 1, 2, 1), None),
    ],
)
def test_value_3d(
    position, view_direction, dims_displayed, world, scale, expected
):
    """Test get_value in 3D with and without scale"""
    data = np.array([[0, 10, 15, 15], [0, 10, 5, 5], [0, 5, 15, 15]])
    layer = Points(data, size=5, scale=scale)
    layer._slice_dims(Dims(ndim=4, ndisplay=3))
    value = layer.get_value(
        position,
        view_direction=view_direction,
        dims_displayed=dims_displayed,
        world=world,
    )
    if expected is None:
        assert value is None
    else:
        assert value == expected


def test_message():
    """Test converting value and coords to message."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[-1] = [0, 0]
    layer = Points(data)
    msg = layer.get_status((0,) * 2)
    assert isinstance(msg, dict)


def test_message_3d():
    """Test converting values and coords to message in 3D."""
    shape = (10, 3)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    layer._slice_input = _SliceInput(
        ndisplay=3,
        world_slice=_ThickNDSlice.make_full(ndim=2),
        order=(0, 1, 2),
    )
    msg = layer.get_status(
        (0, 0, 0), view_direction=[1, 0, 0], dims_displayed=[0, 1, 2]
    )
    assert isinstance(msg, dict)


def test_thumbnail():
    """Test the image thumbnail for square data."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[0] = [0, 0]
    data[-1] = [20, 20]
    layer = Points(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_thumbnail_non_square_data():
    """Test the image thumbnail for non-square data.

    See: https://github.com/napari/napari/issues/1450
    """
    # The points coordinates are in a short and wide range.
    data_range = [1, 32]
    np.random.seed(0)
    data = np.random.random((10, 2)) * data_range
    # Make sure the random points span the range.
    data[0, :] = [0, 0]
    data[-1, :] = data_range
    layer = Points(data)

    layer._update_thumbnail()

    assert layer.thumbnail.shape == layer._thumbnail_shape
    # Check that the thumbnail only contains non-zero RGB values in the middle two rows.
    mid_row = layer.thumbnail.shape[0] // 2
    expected_zeros = np.zeros(shape=(mid_row - 1, 32, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        layer.thumbnail[: mid_row - 1, :, :3], expected_zeros
    )
    assert (
        np.count_nonzero(layer.thumbnail[mid_row - 1 : mid_row + 1, :, :3]) > 0
    )
    np.testing.assert_array_equal(
        layer.thumbnail[mid_row + 1 :, :, :3], expected_zeros
    )


def test_thumbnail_with_n_points_greater_than_max():
    """Test thumbnail generation with n_points > _max_points_thumbnail

    see: https://github.com/napari/napari/pull/934
    """
    # 2D
    max_points = Points._max_points_thumbnail * 2
    bigger_data = np.random.randint(10, 100, (max_points, 2))
    big_layer = Points(bigger_data)
    big_layer._update_thumbnail()
    assert big_layer.thumbnail.shape == big_layer._thumbnail_shape

    # #3D
    bigger_data_3d = np.random.randint(10, 100, (max_points, 3))
    bigger_layer_3d = Points(bigger_data_3d)
    bigger_layer_3d._slice_dims(Dims(ndim=3, ndisplay=3))
    bigger_layer_3d._update_thumbnail()
    assert bigger_layer_3d.thumbnail.shape == bigger_layer_3d._thumbnail_shape


def test_view_data():
    coords = np.array([[0, 1, 1], [0, 2, 2], [1, 3, 3], [3, 3, 3]])
    layer = Points(coords)

    layer._slice_dims(Dims(ndim=3, point=(0, 0, 0)))
    assert np.array_equal(layer._view_data, coords[np.ix_([0, 1], [1, 2])])

    layer._slice_dims(Dims(ndim=3, point=(1, 0, 0)))
    assert np.array_equal(layer._view_data, coords[np.ix_([2], [1, 2])])

    layer._slice_dims(Dims(ndim=3, point=(1, 0, 0), ndisplay=3))
    assert np.array_equal(layer._view_data, coords)


def test_view_size():
    """Test out of slice point rendering and slicing with no points."""
    coords = np.array([[0, 1, 1], [0, 2, 2], [1, 3, 3], [4, 3, 3]])
    sizes = np.array([5, 5, 3, 3])
    layer = Points(coords, size=sizes, out_of_slice_display=False)

    layer._slice_dims(Dims(ndim=3, point=(0, 0, 0)))
    assert np.array_equal(layer._view_size, sizes[[0, 1]])

    layer._slice_dims(Dims(ndim=3, point=(1, 0, 0)))
    assert np.array_equal(layer._view_size, sizes[[2]])

    layer.out_of_slice_display = True
    # NOTE: since a dims slice of thickness 0 defaults back to 1,
    # out_of_slice_display actually compares the half-size with
    # distance + 0.5, not just distance
    assert len(layer._view_size) == 3

    # test a slice with no points
    layer.out_of_slice_display = False
    layer._slice_dims(Dims(ndim=3, point=(2, 0, 0)))
    assert np.array_equal(layer._view_size, [])


def test_view_colors():
    coords = [[0, 1, 1], [0, 2, 2], [1, 3, 3], [3, 3, 3]]
    face_color = np.array(
        [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
    )
    border_color = np.array(
        [[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    )

    layer = Points(coords, face_color=face_color, border_color=border_color)
    layer._slice_dims(Dims(ndim=3, point=(0, 0, 0)))
    assert np.array_equal(layer._view_face_color, face_color[[0, 1]])
    assert np.array_equal(layer._view_border_color, border_color[[0, 1]])

    layer._slice_dims(Dims(ndim=3, point=(1, 0, 0)))
    assert np.array_equal(layer._view_face_color, face_color[[2]])
    assert np.array_equal(layer._view_border_color, border_color[[2]])

    # view colors should return empty array if there are no points
    layer._slice_dims(Dims(ndim=3, point=(2, 0, 0)))
    assert len(layer._view_face_color) == 0
    assert len(layer._view_border_color) == 0


def test_interaction_box():
    """Test the boxes calculated for selected points"""
    data = [[3, 3]]
    size = 2
    layer = Points(data, size=size)

    # get a box with no points selected
    index = []
    box = layer.interaction_box(index)
    assert box is None

    # get a box with a point selected
    index = [0]
    expected_box = points_to_squares(data, size)
    box = layer.interaction_box(index)
    np.all([np.isin(p, expected_box) for p in box])


def test_world_data_extent():
    """Test extent after applying transforms."""
    data = [(7, -5, 0), (-2, 0, 15), (4, 30, 12)]
    min_val = (-2, -5, 0)
    max_val = (7, 30, 15)
    layer = Points(data)
    extent = np.array((min_val, max_val))
    check_layer_world_data_extent(layer, extent, (3, 1, 1), (10, 20, 5))


def test_scale_init():
    layer = Points(None, scale=(1, 1, 1, 1))
    assert layer.ndim == 4
    layer1 = Points([], scale=(1, 1, 1, 1))
    assert layer1.ndim == 4
    layer2 = Points([])
    assert layer2.ndim == 2

    with pytest.raises(ValueError, match='dimensions must be equal to ndim'):
        Points([[1, 1, 1]], scale=(1, 1, 1, 1))


def test_update_none():
    layer = Points([(1, 2, 3), (1, 3, 2)])
    assert layer.ndim == 3
    assert layer.data.size == 6
    layer.data = None
    assert layer.ndim == 3
    assert layer.data.size == 0
    layer.data = [(1, 2, 3), (1, 3, 2)]
    assert layer.ndim == 3
    assert layer.data.size == 6


def test_set_face_color_mode_after_set_properties():
    # See GitHub issue for more details:
    # https://github.com/napari/napari/issues/2755
    np.random.seed(0)
    num_points = 3
    points = Points(np.random.random((num_points, 2)))

    points.properties = {
        'cat': np.random.randint(low=0, high=num_points, size=num_points),
        'cont': np.random.random(num_points),
    }

    # Initially the color_mode is DIRECT, which means that the face ColorManager
    # has no color_properties, so the first property is used with a warning.
    with pytest.warns(UserWarning):
        points.face_color_mode = 'cycle'

    first_property_key, first_property_values = next(
        iter(points.properties.items())
    )
    expected_properties = ColorProperties(
        name=first_property_key,
        values=first_property_values,
        current_value=first_property_values[-1],
    )
    assert points._face.color_properties == expected_properties


def test_to_mask_2d_with_size_1():
    points = Points([[1, 4]], size=1)

    mask = points.to_mask(shape=(5, 7))

    expected_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_2d_with_size_2():
    points = Points([[1, 4]], size=2)

    mask = points.to_mask(shape=(5, 7))

    expected_mask = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_2d_with_size_4():
    points = Points([[1, 4]], size=4)

    mask = points.to_mask(shape=(5, 7))

    expected_mask = np.array(
        [
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_2d_with_size_4_top_left():
    points = Points([[0, 0]], size=4)

    mask = points.to_mask(shape=(5, 7))

    expected_mask = np.array(
        [
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_2d_with_size_4_bottom_right():
    points = Points([[4, 6]], size=4)

    mask = points.to_mask(shape=(5, 7))

    expected_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_2d_with_diff_sizes():
    points = Points([[2, 2], [1, 4]], size=[1, 2])

    mask = points.to_mask(shape=(5, 7))

    expected_mask = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_2d_with_overlap():
    points = Points([[1, 3], [1, 4]], size=2)

    mask = points.to_mask(shape=(5, 7))

    expected_mask = np.array(
        [
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_2d_with_translate():
    points = Points([[1, 4]], size=2)

    mask = points.to_mask(
        shape=(5, 7), data_to_world=CompositeAffine(translate=(-1, 2))
    )

    expected_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_2d_with_rotate():
    # Make the size just over 2, instead of exactly 2, to ensure that all expected pixels are
    # included, despite floating point imprecision caused by applying the rotation.
    points = Points([[-4, 1]], size=2.1)

    mask = points.to_mask(
        shape=(5, 7), data_to_world=CompositeAffine(rotate=90)
    )

    # The point [-4, 1] is defined in world coordinates, so after applying
    # the inverse data_to_world transform will become [1, 4].
    expected_mask = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_2d_with_isotropic_scale():
    points = Points([[2, 8]], size=4)

    mask = points.to_mask(
        shape=(5, 7), data_to_world=CompositeAffine(scale=(2, 2))
    )

    expected_mask = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_2d_with_negative_isotropic_scale():
    points = Points([[2, -8]], size=4)

    mask = points.to_mask(
        shape=(5, 7), data_to_world=CompositeAffine(scale=(2, -2))
    )

    expected_mask = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_2d_with_anisotropic_scale_isotropic_output():
    # With isotropic output, the size of the output ball is determined
    # by the geometric mean of the scale which is sqrt(2), so absorb that
    # into the size to keep the math simple.
    points = Points([[2, 4]], size=2 * np.sqrt(2))

    mask = points.to_mask(
        shape=(5, 7),
        data_to_world=CompositeAffine(scale=(2, 1)),
        isotropic_output=True,
    )

    expected_mask = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_2d_with_anisotropic_scale_anisotropic_output():
    points = Points([[2, 4]], size=4)

    mask = points.to_mask(
        shape=(5, 7),
        data_to_world=CompositeAffine(scale=(2, 1)),
        isotropic_output=False,
    )

    # With anisotropic output, the output ball will be squashed
    # in the dimension with scaling, so that after adding it back as an image
    # with the same scaling, it should be roughly isotropic.
    expected_mask = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_2d_with_points_scale_but_no_mask_scale():
    points = Points([[1, 4]], size=2, scale=(2, 2))

    mask = points.to_mask(shape=(5, 7))

    expected_mask = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_2d_with_same_points_and_mask_scale():
    scale = (2, 2)
    points = Points([[1, 4]], size=2, scale=scale)

    mask = points.to_mask(
        shape=(5, 7), data_to_world=CompositeAffine(scale=scale)
    )

    expected_mask = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_3d_with_size_1():
    points = Points([[1, 2, 3]], size=1)

    mask = points.to_mask(shape=(3, 4, 5))

    expected_mask = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_to_mask_3d_with_size_2():
    points = Points([[1, 2, 3]], size=2)

    mask = points.to_mask(shape=(3, 4, 5))

    expected_mask = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected_mask)


def test_set_properties_updates_text_values():
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Points(points, properties=properties, text='class')

    layer.properties = {'class': np.array(['D', 'E', 'F'])}

    np.testing.assert_array_equal(layer.text.values, ['D', 'E', 'F'])


def test_set_properties_with_invalid_shape_errors_safely():
    properties = {
        'class': np.array(['A', 'B', 'C']),
    }
    points = Points(np.random.rand(3, 2), text='class', properties=properties)
    np.testing.assert_equal(points.properties, properties)
    np.testing.assert_array_equal(points.text.values, ['A', 'B', 'C'])

    with pytest.raises(
        ValueError, match='(does not match length)|(indices imply)'
    ):
        points.properties = {'class': np.array(['D', 'E'])}

    np.testing.assert_equal(points.properties, properties)
    np.testing.assert_array_equal(points.text.values, ['A', 'B', 'C'])


def test_set_properties_with_missing_text_property_text_becomes_constant_empty_and_warns():
    properties = {
        'class': np.array(['A', 'B', 'C']),
    }
    points = Points(np.random.rand(3, 2), text='class', properties=properties)
    np.testing.assert_equal(points.properties, properties)
    np.testing.assert_array_equal(points.text.values, ['A', 'B', 'C'])

    with pytest.warns(RuntimeWarning):
        points.properties = {'not_class': np.array(['D', 'E', 'F'])}

    values = points.text.values
    np.testing.assert_array_equal(values, ['', '', ''])


def test_text_param_and_setter_are_consistent():
    """See https://github.com/napari/napari/issues/1833"""
    data = np.random.rand(5, 3) * 100
    properties = {
        'accepted': np.random.choice([True, False], (5,)),
    }
    text = {'string': 'accepted', 'color': 'black'}

    points_init = Points(data, properties=properties, text=text)

    points_set = Points(data, properties=properties)
    points_set.text = text

    np.testing.assert_array_equal(
        points_init.text.values,
        points_set.text.values,
    )
    np.testing.assert_array_equal(
        points_init.text.color, points_set.text.color
    )


def test_shown():
    """Test setting shown property"""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert len(layer.shown) == shape[0]
    assert np.all(layer.shown)

    # Hide the last point
    layer.shown[-1] = False
    assert np.all(layer.shown[:-1])
    assert layer.shown[-1] == False  # noqa

    # Add a new point, it should be shown but not affect the others
    coord = [17, 17]
    layer.add(coord)
    assert len(layer.shown) == shape[0] + 1
    assert np.all(layer.shown[:-2])
    assert layer.shown[-2] == False  # noqa
    assert layer.shown[-1] == True  # noqa


def test_shown_view_size_and_view_data_have_the_same_dimension():
    data = [[0, 0, 0], [1, 1, 1]]
    # Data with default settings
    layer = Points(
        data, out_of_slice_display=False, shown=[True, True], size=3
    )
    assert layer._view_size.shape[0] == layer._view_data.shape[0]
    assert layer._view_size.shape[0] == 1
    assert np.array_equal(layer._view_size, [3])

    # shown == [True, False]
    layer = Points(
        data, out_of_slice_display=False, shown=[True, False], size=3
    )
    assert layer._view_size.shape[0] == layer._view_data.shape[0]
    assert layer._view_size.shape[0] == 1
    assert np.array_equal(layer._view_size, [3])

    # shown == [False, True]
    layer = Points(
        data, out_of_slice_display=False, shown=[False, True], size=3
    )
    assert layer._view_size.shape[0] == layer._view_data.shape[0]
    assert layer._view_size.shape[0] == 0
    assert np.array_equal(layer._view_size, [])

    # shown == [False, False]
    layer = Points(
        data, out_of_slice_display=False, shown=[False, False], size=3
    )
    assert layer._view_size.shape[0] == layer._view_data.shape[0]
    assert layer._view_size.shape[0] == 0
    assert np.array_equal(layer._view_size, [])

    # Out of slice display == True
    layer = Points(data, out_of_slice_display=True, shown=[True, True], size=3)
    assert layer._view_size.shape[0] == layer._view_data.shape[0]
    assert layer._view_size.shape[0] == 2
    assert np.array_equiv(layer._view_size, [3, 2])

    # Out of slice display == True && shown == [True, False]
    layer = Points(
        data, out_of_slice_display=True, shown=[True, False], size=3
    )
    assert layer._view_size.shape[0] == layer._view_data.shape[0]
    assert layer._view_size.shape[0] == 1
    assert np.array_equal(layer._view_size, [3])

    # Out of slice display == True && shown == [False, True]
    layer = Points(
        data, out_of_slice_display=True, shown=[False, True], size=3
    )
    assert layer._view_size.shape[0] == layer._view_data.shape[0]
    assert layer._view_size.shape[0] == 1
    assert np.array_equal(layer._view_size, [2])

    # Out of slice display == True && shown == [False, False]
    layer = Points(
        data, out_of_slice_display=True, shown=[False, False], size=3
    )
    assert layer._view_size.shape[0] == layer._view_data.shape[0]
    assert layer._view_size.shape[0] == 0
    assert np.array_equal(layer._view_size, [])


def test_empty_data_from_tuple():
    """Test that empty data raises an error."""
    layer = Points(name='points')
    layer2 = Points.create(*layer.as_layer_data_tuple())
    assert layer2.data.size == 0


@pytest.mark.parametrize(
    ('attribute', 'new_value'),
    [
        ('size', 20),
        ('face_color', np.asarray([0.0, 0.0, 1.0, 1.0])),
        ('border_color', np.asarray([0.0, 0.0, 1.0, 1.0])),
        ('border_width', np.asarray([0.2])),
    ],
)
def test_new_point_size_editable(attribute, new_value):
    """tests the newly placed points may be edited without re-selecting"""
    layer = Points()
    layer.mode = Mode.ADD
    layer.add((0, 0))

    setattr(layer, f'current_{attribute}', new_value)
    np.testing.assert_allclose(getattr(layer, attribute)[0], new_value)


def test_antialiasing_setting_and_event_emission():
    """Antialiasing changing should cause event emission."""
    data = [[0, 0, 0], [1, 1, 1]]
    layer = Points(data)
    layer.events.antialiasing = Mock()
    layer.antialiasing = 5
    assert layer.antialiasing == 5
    layer.events.antialiasing.assert_called_once()


def test_antialiasing_value_clipping():
    """Antialiasing can only be set to positive values."""
    data = [[0, 0, 0], [1, 1, 1]]
    layer = Points(data)
    with pytest.warns(RuntimeWarning):
        layer.antialiasing = -1
    assert layer.antialiasing == 0


def test_set_drag_start():
    """Drag start should only change when currently None."""
    data = [[0, 0], [1, 1]]
    layer = Points(data)
    assert layer._drag_start is None
    position = (0, 1)
    layer._set_drag_start({0}, position=position)
    np.testing.assert_array_equal(layer._drag_start, position)
    layer._set_drag_start({0}, position=(1, 2))
    np.testing.assert_array_equal(layer._drag_start, position)


@pytest.mark.parametrize(
    ('dims_indices', 'target_indices'),
    [
        ((8, np.nan, np.nan), [2]),
        ((10, np.nan, np.nan), [0, 1, 3, 4]),
        ((10 + 2 * 1e-12, np.nan, np.nan), [0, 1, 3, 4]),
        ((10.1, np.nan, np.nan), [0, 1, 3, 4]),
    ],
)
def test_point_slice_request_response(dims_indices, target_indices):
    """Test points slicing with request and response."""
    data = [
        (10, 2, 4),
        (10 + 2 * 1e-7, 4, 6),
        (8, 1, 7),
        (10.1, 7, 2),
        (10 - 2 * 1e-7, 1, 6),
    ]

    layer = Points(data)

    data_slice = _ThickNDSlice.make_full(point=dims_indices)

    request = layer._make_slice_request_internal(
        layer._slice_input, data_slice
    )
    response = request()

    assert len(response.indices) == len(target_indices)
    assert all(
        a == b for a, b in zip(response.indices, target_indices, strict=False)
    )


def test_editable_and_visible_are_independent():
    """See https://github.com/napari/napari/issues/1346"""
    data = np.empty((0, 2))
    layer = Points(data)
    assert layer.editable
    assert layer.visible

    layer.editable = False
    layer.visible = False
    assert not layer.editable
    assert not layer.visible

    layer.visible = True

    assert not layer.editable


def test_point_selection_remains_evented_after_update():
    """Existing evented selection model should be updated rather than replaced."""
    data = np.empty((3, 2))
    layer = Points(data)
    assert isinstance(layer.selected_data, Selection)
    layer.selected_data = {0, 1}
    assert isinstance(layer.selected_data, Selection)


def test_points_data_setter_emits_event():
    data = np.random.random((5, 2))
    emitted_events = Mock()
    layer = Points(data)
    layer.events.data.connect(emitted_events)
    layer.data = np.random.random((5, 2))
    assert emitted_events.call_count == 2


def test_points_add_delete_only_emit_two_events():
    data = np.random.random((5, 2))
    emitted_events = Mock()
    layer = Points(data)
    layer.events.data.connect(emitted_events)
    layer.add(np.random.random(2))
    assert emitted_events.call_count == 2
    layer.selected_data = {3}
    layer.remove_selected()
    assert emitted_events.call_count == 4


def test_data_setter_events():
    data = np.random.random((5, 2))
    layer = Points(data)
    layer.events.data = Mock()

    layer.data = []
    assert layer.events.data.call_args_list[0][1] == {
        'value': data,
        'action': ActionType.REMOVING,
        'data_indices': tuple(i for i in range(len(data))),
        'vertex_indices': ((),),
    }

    # Avoid truth value of empty array error
    assert np.array_equal(
        layer.events.data.call_args_list[1][1]['value'], np.empty((0, 2))
    )
    assert (
        layer.events.data.call_args_list[1][1]['action'] == ActionType.REMOVED
    )
    assert layer.events.data.call_args_list[1][1]['data_indices'] == ()
    assert layer.events.data.call_args_list[1][1]['vertex_indices'] == ((),)

    layer.data = data
    assert np.array_equal(
        layer.events.data.call_args_list[2][1]['value'], np.empty((0, 2))
    )
    assert (
        layer.events.data.call_args_list[2][1]['action'] == ActionType.ADDING
    )
    assert layer.events.data.call_args_list[2][1]['data_indices'] == tuple(
        i for i in range(len(data))
    )
    assert layer.events.data.call_args_list[2][1]['vertex_indices'] == ((),)

    assert layer.events.data.call_args_list[3][1] == {
        'value': data,
        'action': ActionType.ADDED,
        'data_indices': tuple(i for i in range(len(data))),
        'vertex_indices': ((),),
    }

    layer.data = data
    assert layer.events.data.call_args_list[4][1] == {
        'value': data,
        'action': ActionType.CHANGING,
        'data_indices': tuple(i for i in range(len(layer.data))),
        'vertex_indices': ((),),
    }
    assert layer.events.data.call_args_list[5][1] == {
        'value': data,
        'action': ActionType.CHANGED,
        'data_indices': tuple(i for i in range(len(layer.data))),
        'vertex_indices': ((),),
    }


def test_thick_slice():
    data = np.array([[0, 0, 0], [10, 10, 10]])
    layer = Points(data)

    # only first point shown
    layer._slice_dims(Dims(ndim=3, point=(0, 0, 0)))
    np.testing.assert_array_equal(layer._view_data, data[:1, -2:])

    layer.projection_mode = 'all'
    np.testing.assert_array_equal(layer._view_data, data[:1, -2:])

    # if margin is thick enough and projection is `all`,
    # it will take in the other point
    layer._slice_dims(Dims(ndim=3, point=(0, 0, 0), margin_right=(10, 0, 0)))
    np.testing.assert_array_equal(layer._view_data, data[:, -2:])


@pytest.mark.parametrize(
    ('old_name', 'new_name', 'value'),
    [
        ('edge_width', 'border_width', 0.9),
        ('edge_width_is_relative', 'border_width_is_relative', False),
        ('current_edge_width', 'current_border_width', 0.9),
        ('edge_color', 'border_color', 'blue'),
        ('current_edge_color', 'current_border_color', 'pink'),
    ],
)
def test_events_callback(old_name, new_name, value):
    data = np.array([[0, 0, 0], [10, 10, 10]])
    layer = Points(data)
    old_name_callback = Mock()
    new_name_callback = Mock()
    with pytest.warns(FutureWarning):
        getattr(layer.events, old_name).connect(old_name_callback)
    getattr(layer.events, new_name).connect(new_name_callback)

    setattr(layer, new_name, value)

    new_name_callback.assert_called_once()
    old_name_callback.assert_called_once()


def test_changing_symbol():
    """Changing the symbol should update the UI"""
    layer = Points(np.random.rand(2, 2))

    assert layer.symbol[1].value == 'disc'
    assert layer.current_symbol.value == 'disc'

    # select a point and change its symbol
    layer.selected_data = {1}
    layer.current_symbol = 'square'
    assert layer.symbol[1].value == 'square'
    # add a point and check that it has the new symbol
    layer.add([1, 1])
    assert layer.symbol[2].value == 'square'
    assert layer.symbol[0].value == 'disc'


def test_docstring():
    validate_all_params_in_docstring(Points)
    validate_kwargs_sorted(Points)
    validate_docstring_parent_class_consistency(Points)


@pytest.mark.parametrize(
    'key',
    [
        'edge_width',
        'edge_width_is_relative',
        'edge_color',
        'edge_color_cycle',
        'edge_colormap',
        'edge_contrast_limits',
    ],
)
def test_as_layer_data_tuple_read_deprecated_attr(key: str):
    _, attrs, _ = Points().as_layer_data_tuple()
    with pytest.warns(FutureWarning, match='is deprecated since'):
        attrs[key]
