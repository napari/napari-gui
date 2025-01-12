from copy import copy
from itertools import cycle, islice
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from napari._pydantic_compat import ValidationError
from napari._tests.utils import (
    assert_colors_equal,
    check_layer_world_data_extent,
)
from napari.components import ViewerModel
from napari.components.dims import Dims
from napari.layers import Shapes
from napari.layers.base._base_constants import ActionType
from napari.layers.utils._text_constants import Anchor
from napari.layers.utils.color_encoding import ConstantColorEncoding
from napari.utils._test_utils import (
    validate_all_params_in_docstring,
    validate_kwargs_sorted,
)
from napari.utils.colormaps.standardize_color import transform_color


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


def test_empty_shapes():
    shp = Shapes()
    assert shp.ndim == 2


def test_update_thumbnail_empty_shapes():
    """Test updating the thumbnail with an empty shapes layer."""
    layer = Shapes()
    layer._allow_thumbnail_update = True
    layer._update_thumbnail()


def test_empty_shapes_with_features():
    """See the following for the points issues this covers:
    https://github.com/napari/napari/issues/5632
    https://github.com/napari/napari/issues/5634
    """
    shapes = Shapes(
        features={'a': np.empty(0, int)},
        feature_defaults={'a': 0},
        face_color='a',
        face_color_cycle=list('rgb'),
    )

    shapes.add_rectangles([[0, 0], [1, 1]])
    shapes.feature_defaults['a'] = 1
    shapes.add_rectangles([[1, 1], [2, 2]])
    shapes.feature_defaults = {'a': 2}
    shapes.add_rectangles([[2, 2], [3, 3]])

    assert_colors_equal(shapes.face_color, list('rgb'))


properties_array = {'shape_type': _make_cycled_properties(['A', 'B'], 10)}
properties_list = {'shape_type': list(_make_cycled_properties(['A', 'B'], 10))}


@pytest.mark.parametrize('properties', [properties_array, properties_list])
def test_properties(properties):
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, properties=copy(properties))
    np.testing.assert_equal(layer.properties, properties)

    current_prop = {'shape_type': np.array(['B'])}
    assert layer.current_properties == current_prop

    # test removing shapes
    layer.selected_data = {0, 1}
    layer.remove_selected()
    remove_properties = properties['shape_type'][2::]
    assert len(layer.properties['shape_type']) == (shape[0] - 2)
    assert np.array_equal(layer.properties['shape_type'], remove_properties)

    # test selection of properties
    layer.selected_data = {0}
    selected_annotation = layer.current_properties['shape_type']
    assert len(selected_annotation) == 1
    assert selected_annotation[0] == 'A'

    # test adding shapes with properties
    new_data = np.random.random((1, 4, 2))
    new_shape_type = ['rectangle']
    layer.add(new_data, shape_type=new_shape_type)
    add_properties = np.concatenate((remove_properties, ['A']), axis=0)
    assert np.array_equal(layer.properties['shape_type'], add_properties)

    # test copy/paste
    layer.selected_data = {0, 1}
    layer._copy_data()
    assert np.array_equal(
        layer._clipboard['features']['shape_type'], ['A', 'B']
    )

    layer._paste_data()
    paste_properties = np.concatenate((add_properties, ['A', 'B']), axis=0)
    assert np.array_equal(layer.properties['shape_type'], paste_properties)

    # test updating a property
    layer.mode = 'select'
    layer.selected_data = {0}
    new_property = {'shape_type': np.array(['B'])}
    layer.current_properties = new_property
    updated_properties = layer.properties
    assert updated_properties['shape_type'][0] == 'B'


@pytest.mark.parametrize('attribute', ['edge', 'face'])
def test_adding_properties(attribute):
    """Test adding properties to an existing layer"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)

    # add properties
    properties = {'shape_type': _make_cycled_properties(['A', 'B'], shape[0])}
    layer.properties = properties
    np.testing.assert_equal(layer.properties, properties)

    # add properties as a dataframe
    properties_df = pd.DataFrame(properties)
    layer.properties = properties_df
    np.testing.assert_equal(layer.properties, properties)

    # add properties as a dictionary with list values
    properties_list = {
        'shape_type': list(_make_cycled_properties(['A', 'B'], shape[0]))
    }
    layer.properties = properties_list
    assert isinstance(layer.properties['shape_type'], np.ndarray)

    # removing a property that was the _*_color_property should give a warning
    setattr(layer, f'_{attribute}_color_property', 'shape_type')
    properties_2 = {
        'not_shape_type': _make_cycled_properties(['A', 'B'], shape[0])
    }
    with pytest.warns(RuntimeWarning):
        layer.properties = properties_2


def test_colormap_scale_change():
    data = 20 * np.random.random((10, 4, 2))
    properties = {'a': np.linspace(0, 1, 10), 'b': np.linspace(0, 100000, 10)}
    layer = Shapes(data, properties=properties, edge_color='b')

    assert not np.allclose(
        layer.edge_color[0], layer.edge_color[1], atol=0.001
    )

    layer.edge_color = 'a'

    # note that VisPy colormaps linearly interpolate by default, so
    # non-rescaled colors are not identical, but they are closer than 24-bit
    # color precision can distinguish!
    assert not np.allclose(
        layer.edge_color[0], layer.edge_color[1], atol=0.001
    )


def test_data_setter_with_properties():
    """Test layer data on a layer with properties via the data setter"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'shape_type': _make_cycled_properties(['A', 'B'], shape[0])}
    layer = Shapes(data, properties=properties)
    layer.events.data = Mock()

    # test setting to data with fewer shapes
    n_new_shapes = 4
    new_data = 20 * np.random.random((n_new_shapes, 4, 2))
    layer.data = new_data
    assert len(layer.properties['shape_type']) == n_new_shapes

    # test setting to data with more shapes
    n_new_shapes_2 = 6
    new_data_2 = 20 * np.random.random((n_new_shapes_2, 4, 2))
    layer.data = new_data_2
    assert len(layer.properties['shape_type']) == n_new_shapes_2

    # test setting to data with same shapes
    new_data_3 = 20 * np.random.random((n_new_shapes_2, 4, 2))
    layer.data = new_data_3
    assert len(layer.properties['shape_type']) == n_new_shapes_2


def test_properties_dataframe():
    """Test if properties can be provided as a DataFrame"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'shape_type': _make_cycled_properties(['A', 'B'], shape[0])}
    properties_df = pd.DataFrame(properties)
    properties_df = properties_df.astype(properties['shape_type'].dtype)
    layer = Shapes(data, properties=properties_df)
    np.testing.assert_equal(layer.properties, properties)


def test_setting_current_properties():
    shape = (2, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {
        'annotation': ['paw', 'leg'],
        'confidence': [0.5, 0.75],
        'annotator': ['jane', 'ash'],
        'model': ['worst', 'best'],
    }
    layer = Shapes(data, properties=copy(properties))
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


def test_empty_layer_with_text_property_choices():
    """Test initializing an empty layer with text defined"""
    default_properties = {'shape_type': np.array([1.5], dtype=float)}
    text_kwargs = {'string': 'shape_type', 'color': 'red'}
    layer = Shapes(
        property_choices=default_properties,
        text=text_kwargs,
    )
    assert layer.text.values.size == 0
    np.testing.assert_allclose(layer.text.color.constant, [1, 0, 0, 1])

    # add a shape and check that the appropriate text value was added
    layer.add(np.random.random((1, 4, 2)))
    np.testing.assert_equal(layer.text.values, ['1.5'])
    np.testing.assert_allclose(layer.text.color.constant, [1, 0, 0, 1])


def test_empty_layer_with_text_formatted():
    """Test initializing an empty layer with text defined"""
    default_properties = {'shape_type': np.array([1.5], dtype=float)}
    layer = Shapes(
        property_choices=default_properties,
        text='shape_type: {shape_type:.2f}',
    )
    assert layer.text.values.size == 0

    # add a shape and check that the appropriate text value was added
    layer.add(np.random.random((1, 4, 2)))
    np.testing.assert_equal(layer.text.values, ['shape_type: 1.50'])


@pytest.mark.parametrize('properties', [properties_array, properties_list])
def test_text_from_property_value(properties):
    """Test setting text from a property value"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, properties=copy(properties), text='shape_type')

    np.testing.assert_equal(layer.text.values, properties['shape_type'])


@pytest.mark.parametrize('properties', [properties_array, properties_list])
def test_text_from_property_fstring(properties):
    """Test setting text with an f-string from the property value"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(
        data, properties=copy(properties), text='type: {shape_type}'
    )

    expected_text = ['type: ' + v for v in properties['shape_type']]
    np.testing.assert_equal(layer.text.values, expected_text)

    # test updating the text
    layer.text = 'type-ish: {shape_type}'
    expected_text_2 = ['type-ish: ' + v for v in properties['shape_type']]
    np.testing.assert_equal(layer.text.values, expected_text_2)

    # copy/paste
    layer.selected_data = {0}
    layer._copy_data()
    layer._paste_data()
    expected_text_3 = [*expected_text_2, 'type-ish: A']
    np.testing.assert_equal(layer.text.values, expected_text_3)

    # add shape
    layer.selected_data = {0}
    new_shape = np.random.random((1, 4, 2))
    layer.add(new_shape)
    expected_text_4 = [*expected_text_3, 'type-ish: A']
    np.testing.assert_equal(layer.text.values, expected_text_4)


@pytest.mark.parametrize('properties', [properties_array, properties_list])
def test_set_text_with_kwarg_dict(properties):
    text_kwargs = {
        'string': 'type: {shape_type}',
        'color': ConstantColorEncoding(constant=[0, 0, 0, 1]),
        'rotation': 10,
        'translation': [5, 5],
        'anchor': Anchor.UPPER_LEFT,
        'size': 10,
        'visible': True,
    }
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, properties=copy(properties), text=text_kwargs)

    expected_text = ['type: ' + v for v in properties['shape_type']]
    np.testing.assert_equal(layer.text.values, expected_text)

    for property_, value in text_kwargs.items():
        if property_ == 'string':
            continue
        layer_value = getattr(layer._text, property_)
        np.testing.assert_equal(layer_value, value)


@pytest.mark.parametrize('properties', [properties_array, properties_list])
def test_text_error(properties):
    """creating a layer with text as the wrong type should raise an error"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    # try adding text as the wrong type
    with pytest.raises(ValidationError):
        Shapes(data, properties=copy(properties), text=123)


def test_select_properties_object_dtype():
    """selecting points when they have a property of object dtype should not fail"""
    # pandas uses object as dtype for strings by default
    properties = pd.DataFrame({'color': ['red', 'green']})
    pl = Shapes(np.ones((2, 2, 2)), properties=properties)
    selection = {0, 1}
    pl.selected_data = selection
    assert pl.selected_data == selection


def test_refresh_text():
    """Test refreshing the text after setting new properties"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'shape_type': ['A'] * shape[0]}
    layer = Shapes(data, properties=copy(properties), text='shape_type')

    new_properties = {'shape_type': ['B'] * shape[0]}
    layer.properties = new_properties
    np.testing.assert_equal(layer.text.values, new_properties['shape_type'])


@pytest.mark.parametrize('prepend', [(), (7,), (8, 9)])
def test_nd_text(prepend):
    """Test slicing of text coords with nD shapes

    We can prepend as many dimensions as we want it should not change the result
    """
    shapes_data = [
        [
            prepend + (0, 10, 10, 10),
            prepend + (0, 10, 20, 20),
            prepend + (0, 10, 10, 20),
            prepend + (0, 10, 20, 10),
        ],
        [
            prepend + (1, 20, 30, 30),
            prepend + (1, 20, 50, 50),
            prepend + (1, 20, 50, 30),
            prepend + (1, 20, 30, 50),
        ],
    ]
    layer = Shapes(shapes_data)
    assert layer.ndim == 4 + len(prepend)

    layer._slice_dims(
        Dims(
            ndim=layer.ndim,
            ndisplay=2,
            range=((0, 100, 1),) * layer.ndim,
            point=prepend + (0, 10, 0, 0),
        )
    )
    np.testing.assert_equal(layer._indices_view, [0])
    np.testing.assert_equal(layer._view_text_coords[0], [[15, 15]])

    # TODO: 1st bug #6205, ndisplay 3 is buggy in 5+ dimensions
    # may need to call _update_dims
    layer._slice_dims(
        Dims(
            ndim=layer.ndim,
            ndisplay=3,
            range=((0, 100, 1),) * layer.ndim,
            point=prepend + (1, 0, 0, 0),
        )
    )
    np.testing.assert_equal(layer._indices_view, [1])
    np.testing.assert_equal(layer._view_text_coords[0], [[20, 40, 40]])


@pytest.mark.parametrize('properties', [properties_array, properties_list])
def test_data_setter_with_text(properties):
    """Test layer data on a layer with text via the data setter"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, properties=copy(properties), text='shape_type')

    # test setting to data with fewer shapes
    n_new_shapes = 4
    new_data = 20 * np.random.random((n_new_shapes, 4, 2))
    layer.data = new_data
    assert len(layer.text.values) == n_new_shapes

    # test setting to data with more shapes
    n_new_shapes_2 = 6
    new_data_2 = 20 * np.random.random((n_new_shapes_2, 4, 2))
    layer.data = new_data_2
    assert len(layer.text.values) == n_new_shapes_2

    # test setting to data with same shapes
    new_data_3 = 20 * np.random.random((n_new_shapes_2, 4, 2))
    layer.data = new_data_3
    assert len(layer.text.values) == n_new_shapes_2


def test_rectangles(two_and_four_corners):
    """Test instantiating Shapes layer with a random 2D rectangles."""
    # Test instantiating with data
    layer = Shapes(two_and_four_corners)
    assert layer.nshapes == len(two_and_four_corners)
    # 4 corner rectangle(s) passed, assert vertices the same
    if two_and_four_corners[0].shape[0] == 4:
        assert np.all(
            [
                layer.data[i] == two_and_four_corners[i]
                for i in range(layer.nshapes)
            ]
        )
    # 2 corner rectangle(s) passed, assert 4 vertices in layer
    else:
        assert [len(rect) == 4 for rect in layer.data]
    assert layer.ndim == two_and_four_corners[0].shape[1]
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_rectangles_add_method(two_and_four_corners):
    # Test adding via add_rectangles method
    layer = Shapes(two_and_four_corners)
    layer2 = Shapes()
    layer2.add_rectangles(two_and_four_corners)
    assert layer.nshapes == layer2.nshapes
    assert np.allclose(layer2.data, layer.data)
    assert np.all([s == 'rectangle' for s in layer2.shape_type])


def test_add_rectangles_raises_errors():
    """Test input validation for add_rectangles method"""
    layer = Shapes()

    np.random.seed(0)
    # single rectangle, 3 vertices
    data = 20 * np.random.random((1, 3, 2))
    with pytest.raises(ValueError, match='invalid number of vertices'):
        layer.add_rectangles(data)
    # multiple rectangles, 5 vertices
    data = 20 * np.random.random((5, 5, 2))
    with pytest.raises(ValueError, match='invalid number of vertices'):
        layer.add_rectangles(data)


def test_rectangle_with_shape_type(single_four_corner: list[np.ndarray]):
    """Test instantiating rectangles with shape_type in data"""
    # Test (rectangle, shape_type) tuple
    data = (single_four_corner, 'rectangle')
    layer = Shapes(data)
    assert layer.nshapes == 1
    assert np.array_equiv(layer.data[0], data[0])
    assert layer.ndim == 2
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_rectangles_with_shape_type(ten_four_corner: list[np.ndarray]):
    # Test (list of rectangles, shape_type) tuple
    data = (ten_four_corner, 'rectangle')
    layer = Shapes(data)
    assert layer.nshapes == len(ten_four_corner)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, ten_four_corner, strict=False)
        ]
    )
    assert layer.ndim == ten_four_corner[0].shape[1]
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_rectangles_with_shape_type_per_element(
    ten_four_corner: list[np.ndarray],
):
    # Test list of (rectangle, shape_type) tuples
    data = [(el, 'rectangle') for el in ten_four_corner]
    layer = Shapes(data)
    assert layer.nshapes == len(ten_four_corner)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, ten_four_corner, strict=False)
        ]
    )
    assert layer.ndim == ten_four_corner[0].shape[1]
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_rectangles_roundtrip():
    """Test a full roundtrip with rectangles data."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    new_layer = Shapes(layer.data)
    assert np.all(
        [nd == d for nd, d in zip(new_layer.data, layer.data, strict=False)]
    )


def test_integer_rectangle():
    """Test instantiating rectangles with integer data."""
    shape = (10, 2, 2)
    np.random.seed(1)
    data = np.random.randint(20, size=shape)
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([len(ld) == 4 for ld in layer.data])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_negative_rectangle(ten_four_corner):
    """Test instantiating rectangles with negative data."""
    data = [x - 10 for x in ten_four_corner]
    layer = Shapes(data)
    assert layer.nshapes == len(data)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, data, strict=False)
        ]
    )
    assert layer.ndim == data[0].shape[1]
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_empty_rectangle():
    """Test instantiating rectangles with empty data."""
    shape = (0, 0, 2)
    data = np.empty(shape)
    layer = Shapes(data)
    assert layer.nshapes == 0
    assert len(layer.data) == 0
    assert layer.ndim == 2


def test_3D_rectangles():
    """Test instantiating Shapes layer with 3D planar rectangles."""
    # Test a single four corner rectangle
    np.random.seed(0)
    planes = np.tile(np.arange(10).reshape((10, 1, 1)), (1, 4, 1))
    corners = np.random.uniform(0, 10, size=(10, 4, 2))
    data = np.concatenate((planes, corners), axis=2, dtype=np.float32)
    layer = Shapes(data)
    assert layer.nshapes == len(data)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, data, strict=False)
        ]
    )
    assert layer.ndim == 3
    assert np.all([s == 'rectangle' for s in layer.shape_type])

    # test adding with add_rectangles
    layer2 = Shapes()
    layer2.add_rectangles(data)
    assert layer2.nshapes == layer.nshapes
    assert np.all(
        [
            np.array_equal(ld, ld2)
            for ld, ld2 in zip(layer.data, layer2.data, strict=False)
        ]
    )
    assert np.all([s == 'rectangle' for s in layer2.shape_type])


def test_ellipses(two_and_four_corners):
    """Test instantiating Shapes layer with 2D ellipses."""
    # Test instantiating with data
    layer = Shapes(two_and_four_corners, shape_type='ellipse')
    assert layer.nshapes == len(two_and_four_corners)
    # 4 corner bounding box passed, assert vertices the same
    if two_and_four_corners[0].shape[0] == 4:
        assert np.all(
            [
                layer.data[i] == two_and_four_corners[i]
                for i in range(layer.nshapes)
            ]
        )
    # (center, radii) passed, assert 4 vertices in layer
    else:
        assert [len(rect) == 4 for rect in layer.data]
    assert layer.ndim == two_and_four_corners[0].shape[1]
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_ellipses_add_method(two_and_four_corners):
    # Test adding via add_ellipses method
    layer = Shapes(two_and_four_corners, shape_type='ellipse')
    layer2 = Shapes()
    layer2.add_ellipses(two_and_four_corners)
    assert layer.nshapes == layer2.nshapes
    assert np.allclose(layer2.data, layer.data)
    assert np.all([s == 'ellipse' for s in layer2.shape_type])


def test_add_ellipses_raises_error():
    """Test input validation for add_ellipses method"""
    layer = Shapes()

    np.random.seed(0)
    # single ellipse, 3 vertices
    data = 20 * np.random.random((1, 3, 2))
    with pytest.raises(ValueError, match='invalid number of vertices'):
        layer.add_ellipses(data)
    # multiple ellipses, 5 vertices
    data = 20 * np.random.random((5, 5, 2))
    with pytest.raises(ValueError, match='invalid number of vertices'):
        layer.add_ellipses(data)


def test_single_ellipses_with_shape_type(single_four_corner):
    """Test instantiating ellipses with shape_type in data"""
    # Test single four corner (vertices, shape_type) tuple
    data = (single_four_corner, 'ellipse')
    layer = Shapes(data)
    assert layer.nshapes == len(single_four_corner)
    assert np.array_equiv(layer.data[0], data[0])
    assert layer.ndim == single_four_corner[0].shape[1]
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_ten_ellipses_with_shape_type(ten_four_corner):
    # Test multiple four corner (list of vertices, shape_type) tuple
    data = (ten_four_corner, 'ellipse')
    layer = Shapes(data)
    assert layer.nshapes == len(ten_four_corner)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, ten_four_corner, strict=False)
        ]
    )
    assert layer.ndim == ten_four_corner[0].shape[1]
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_ten_ellipses_with_shape_type_per_shape(ten_four_corner):
    # Test list of four corner (vertices, shape_type) tuples
    data = [(el, 'ellipse') for el in ten_four_corner]
    layer = Shapes(data)
    assert layer.nshapes == len(ten_four_corner)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, ten_four_corner, strict=False)
        ]
    )
    assert layer.ndim == ten_four_corner[0].shape[1]
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_single_ellipses_two_corner_with_shape_type(single_two_corners):
    # Test single (center-radii, shape_type) ellipse
    data = (single_two_corners, 'ellipse')
    layer = Shapes(data)
    assert layer.nshapes == 1
    assert len(layer.data[0]) == 4
    assert layer.ndim == single_two_corners[0].shape[1]
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_ellipses_two_corner_with_shape_type(ten_two_corners):
    # Test (list of center-radii, shape_type) tuple
    data = (ten_two_corners, 'ellipse')
    layer = Shapes(data)
    assert layer.nshapes == len(ten_two_corners)
    assert np.all([len(ld) == 4 for ld in layer.data])
    assert layer.ndim == ten_two_corners[0].shape[1]
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_ellipses_two_corner_with_shape_type_per_shape(ten_two_corners):
    # Test list of (center-radii, shape_type) tuples
    data = [(el, 'ellipse') for el in ten_two_corners]
    layer = Shapes(data)
    assert layer.nshapes == len(ten_two_corners)
    assert np.all([len(ld) == 4 for ld in layer.data])
    assert layer.ndim == ten_two_corners[0].shape[1]
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_4D_ellispse():
    """Test instantiating Shapes layer with 4D planar ellipse."""
    # Test a single 4D ellipse
    np.random.seed(0)
    data = [
        [
            [3, 5, 108, 108],
            [3, 5, 108, 148],
            [3, 5, 148, 148],
            [3, 5, 148, 108],
        ]
    ]
    layer = Shapes(data, shape_type='ellipse')
    assert layer.nshapes == len(data)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, data, strict=False)
        ]
    )
    assert layer.ndim == 4
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    # test adding via add_ellipses
    layer2 = Shapes(ndim=4)
    layer2.add_ellipses(data)
    assert layer.nshapes == layer2.nshapes
    assert np.all(
        [
            np.array_equal(ld, ld2)
            for ld, ld2 in zip(layer.data, layer2.data, strict=False)
        ]
    )
    assert layer.ndim == 4
    assert np.all([s == 'ellipse' for s in layer2.shape_type])


def test_ellipses_roundtrip():
    """Test a full roundtrip with ellipss data."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='ellipse')
    new_layer = Shapes(layer.data, shape_type='ellipse')
    assert np.all(
        [nd == d for nd, d in zip(new_layer.data, layer.data, strict=False)]
    )


def test_lines(two_corners):
    """Test instantiating Shapes layer with a random 2D lines."""
    # Test instantiating with data
    layer = Shapes(two_corners, shape_type='line')
    assert layer.nshapes == len(two_corners)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, two_corners, strict=False)
        ]
    )
    assert layer.ndim == two_corners[0].shape[1]
    assert np.all([s == 'line' for s in layer.shape_type])


def test_lines_add_method(two_corners):
    # Test adding using add_lines
    layer = Shapes(two_corners, shape_type='line')
    layer2 = Shapes()
    layer2.add_lines(two_corners)
    assert layer.nshapes == layer2.nshapes
    assert np.allclose(layer2.data, layer.data)
    assert np.all([s == 'line' for s in layer2.shape_type])


def test_add_lines_raises_error():
    """Test adding lines with incorrect vertices raise error"""

    # single line
    shape = (1, 3, 2)
    data = 20 * np.random.random(shape)
    layer = Shapes()
    with pytest.raises(ValueError, match='invalid number of vertices'):
        layer.add_lines(data)

    # multiple lines
    data = [
        20 * np.random.random((np.random.randint(3, 10), 2)) for _ in range(10)
    ]
    with pytest.raises(ValueError, match='invalid number of vertices'):
        layer.add_lines(data)


def test_single_lines_with_shape_type(single_two_corners):
    """Test instantiating lines with shape_type"""
    # Test (single line, shape_type) tuple
    data = (single_two_corners, 'line')
    layer = Shapes(data)
    assert layer.nshapes == len(single_two_corners)
    assert np.array_equal(layer.data[0], single_two_corners[0])
    assert layer.ndim == single_two_corners[0].shape[1]
    assert np.all([s == 'line' for s in layer.shape_type])


def test_ten_lines_with_shape_type(ten_two_corners):
    # Test (multiple lines, shape_type) tuple
    data = (ten_two_corners, 'line')
    layer = Shapes(data)
    assert layer.nshapes == len(ten_two_corners)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, ten_two_corners, strict=False)
        ]
    )
    assert layer.ndim == ten_two_corners[0].shape[1]
    assert np.all([s == 'line' for s in layer.shape_type])


def test_ten_lines_with_shape_type_per_shape(ten_two_corners):
    # Test list of (line, shape_type) tuples
    data = [(el, 'line') for el in ten_two_corners]
    layer = Shapes(data)
    assert layer.nshapes == len(ten_two_corners)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, ten_two_corners, strict=False)
        ]
    )
    assert layer.ndim == ten_two_corners[0].shape[1]
    assert np.all([s == 'line' for s in layer.shape_type])


def test_lines_roundtrip():
    """Test a full roundtrip with line data."""
    shape = (10, 2, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='line')
    new_layer = Shapes(layer.data, shape_type='line')
    assert np.all(
        [nd == d for nd, d in zip(new_layer.data, layer.data, strict=False)]
    )


@pytest.mark.parametrize(
    'shape',
    [
        # single path, six points
        (6, 2),
    ]
    + [
        # multiple 2D paths with different numbers of points
        (np.random.randint(2, 12), 2)
        for _ in range(10)
    ],
)
def test_paths(shape):
    """Test instantiating Shapes layer with random 2D paths."""

    # Test instantiating with data
    data = [20 * np.random.random(shape).astype(np.float32)]
    layer = Shapes(data, shape_type='path')
    assert layer.nshapes == len(data)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, data, strict=False)
        ]
    )
    assert layer.ndim == 2
    assert np.all([s == 'path' for s in layer.shape_type])

    # Test adding to layer via add_paths
    layer2 = Shapes()
    layer2.add_paths(data)
    assert layer.nshapes == layer2.nshapes
    assert np.allclose(layer2.data, layer.data)
    assert np.all([s == 'path' for s in layer2.shape_type])


def test_add_paths_raises_error():
    """Test adding paths with incorrect vertices raise error"""

    # single path
    shape = (1, 1, 2)
    data = 20 * np.random.random(shape)
    layer = Shapes()
    with pytest.raises(ValueError, match='invalid number of vertices'):
        layer.add_paths(data)

    # multiple paths
    data = 20 * np.random.random((10, 1, 2))
    with pytest.raises(ValueError, match='invalid number of vertices'):
        layer.add_paths(data)


def test_paths_with_shape_type():
    """Test instantiating paths with shape_type in data"""
    # Test (single path, shape_type) tuple
    shape = (1, 6, 2)
    np.random.seed(0)
    path_points = 20 * np.random.random(shape).astype(np.float32)
    data = (path_points, 'path')
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.array_equal(layer.data[0], path_points[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'path' for s in layer.shape_type])

    # Test (list of paths, shape_type) tuple
    path_points = [
        20 * np.random.random((np.random.randint(2, 12), 2)).astype(np.float32)
        for i in range(10)
    ]
    data = (path_points, 'path')
    layer = Shapes(data)
    assert layer.nshapes == len(path_points)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, path_points, strict=False)
        ]
    )
    assert layer.ndim == 2
    assert np.all([s == 'path' for s in layer.shape_type])

    # Test list of  (path, shape_type) tuples
    data = [(path_points[i], 'path') for i in range(len(path_points))]
    layer = Shapes(data)
    assert layer.nshapes == len(data)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, path_points, strict=False)
        ]
    )
    assert layer.ndim == 2
    assert np.all([s == 'path' for s in layer.shape_type])


def test_paths_roundtrip():
    """Test a full roundtrip with path data."""
    np.random.seed(0)
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(10)
    ]
    layer = Shapes(data, shape_type='path')
    new_layer = Shapes(layer.data, shape_type='path')
    assert np.all(
        [
            np.array_equal(nd, d)
            for nd, d in zip(new_layer.data, layer.data, strict=False)
        ]
    )


@pytest.mark.parametrize(
    'shape',
    [
        # single 2D polygon, six points
        (6, 2),
    ]
    + [
        # multiple 2D polygons with different numbers of points
        (np.random.randint(3, 12), 2)
        for _ in range(10)
    ],
)
def test_polygons(shape):
    """Test instantiating Shapes layer with a random 2D polygons."""

    # Test instantiating with data
    data = [20 * np.random.random(shape).astype(np.float32)]
    layer = Shapes(data, shape_type='polygon')
    assert layer.nshapes == len(data)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, data, strict=False)
        ]
    )
    assert layer.ndim == 2
    assert np.all([s == 'polygon' for s in layer.shape_type])

    # Test adding via add_polygons
    layer2 = Shapes()
    layer2.events.data = Mock()

    layer2.add_polygons(data)
    assert layer.nshapes == layer2.nshapes
    assert np.allclose(layer2.data, layer.data)
    assert np.all([s == 'polygon' for s in layer2.shape_type])

    # Avoid a.any(), a.all()
    assert layer2.events.data.call_args_list[0][1] == {
        'value': [],
        'action': ActionType.ADDING,
        'data_indices': (-1,),
        'vertex_indices': ((),),
    }

    assert np.array_equal(
        layer2.events.data.call_args_list[1][1]['value'], layer.data
    )
    assert (
        layer2.events.data.call_args_list[0][1]['action'] == ActionType.ADDING
    )
    assert layer2.events.data.call_args_list[0][1]['data_indices'] == (-1,)
    assert layer2.events.data.call_args_list[0][1]['vertex_indices'] == ((),)


def test_add_polygons_raises_error():
    """Test input validation for add_polygons method"""
    layer = Shapes()

    np.random.seed(0)
    # single polygon, 2 vertices
    data = 20 * np.random.random((1, 2, 2))
    with pytest.raises(ValueError, match='invalid number of vertices'):
        layer.add_polygons(data)
    # multiple polygons, only some with 2 vertices
    data = [20 * np.random.random((5, 2)) for _ in range(5)] + [
        20 * np.random.random((2, 2)) for _ in range(2)
    ]
    with pytest.raises(ValueError, match='invalid number of vertices'):
        layer.add_polygons(data)


def test_polygons_with_shape_type():
    """Test 2D polygons with shape_type in data"""

    # Test single (polygon, shape_type) tuple
    shape = (1, 6, 2)
    np.random.seed(0)
    vertices = 20 * np.random.random(shape).astype(np.float32)
    data = (vertices, 'polygon')
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.array_equal(layer.data[0], vertices[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'polygon' for s in layer.shape_type])

    # Test (list of polygons, shape_type) tuple
    polygons = [
        20 * np.random.random((np.random.randint(2, 12), 2)).astype(np.float32)
        for i in range(10)
    ]
    data = (polygons, 'polygon')
    layer = Shapes(data)
    assert layer.nshapes == len(polygons)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, polygons, strict=False)
        ]
    )
    assert layer.ndim == 2
    assert np.all([s == 'polygon' for s in layer.shape_type])

    # Test list of (polygon, shape_type) tuples
    data = [(polygons[i], 'polygon') for i in range(len(polygons))]
    layer = Shapes(data)
    assert layer.nshapes == len(polygons)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, polygons, strict=False)
        ]
    )
    assert layer.ndim == 2
    assert np.all([s == 'polygon' for s in layer.shape_type])


def test_polygon_roundtrip():
    """Test a full roundtrip with polygon data."""
    np.random.seed(0)
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(10)
    ]
    layer = Shapes(data, shape_type='polygon')
    new_layer = Shapes(layer.data, shape_type='polygon')
    assert np.all(
        [
            np.array_equal(nd, d)
            for nd, d in zip(new_layer.data, layer.data, strict=False)
        ]
    )


def test_mixed_shapes():
    """Test instantiating Shapes layer with a mix of random 2D shapes."""
    # Test multiple polygons with different numbers of points
    np.random.seed(0)
    shape_vertices = [
        20 * np.random.random((np.random.randint(2, 12), 2)).astype(np.float32)
        for i in range(5)
    ] + list(np.random.random((5, 4, 2)).astype(np.float32))
    shape_type = ['polygon'] * 5 + ['rectangle'] * 3 + ['ellipse'] * 2
    layer = Shapes(shape_vertices, shape_type=shape_type)
    assert layer.nshapes == len(shape_vertices)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, shape_vertices, strict=False)
        ]
    )
    assert layer.ndim == 2
    assert np.all(
        [s == so for s, so in zip(layer.shape_type, shape_type, strict=False)]
    )

    # Test roundtrip with mixed data
    new_layer = Shapes(layer.data, shape_type=layer.shape_type)
    assert np.all(
        [
            np.array_equal(nd, d)
            for nd, d in zip(new_layer.data, layer.data, strict=False)
        ]
    )
    assert np.all(
        [
            ns == s
            for ns, s in zip(
                new_layer.shape_type, layer.shape_type, strict=False
            )
        ]
    )


def test_mixed_shapes_with_shape_type():
    """Test adding mixed shapes with shape_type in data"""
    np.random.seed(0)
    shape_vertices = [
        20 * np.random.random((np.random.randint(2, 12), 2)).astype(np.float32)
        for i in range(5)
    ] + list(np.random.random((5, 4, 2)).astype(np.float32))
    shape_type = ['polygon'] * 5 + ['rectangle'] * 3 + ['ellipse'] * 2

    # Test multiple (shape, shape_type) tuples
    data = list(zip(shape_vertices, shape_type, strict=False))
    layer = Shapes(data)
    assert layer.nshapes == len(shape_vertices)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, shape_vertices, strict=False)
        ]
    )
    assert layer.ndim == 2
    assert np.all(
        [s == so for s, so in zip(layer.shape_type, shape_type, strict=False)]
    )


def test_data_shape_type_overwrites_meta():
    """Test shape type passed through data property overwrites metadata shape type"""
    shape = (10, 4, 2)
    np.random.seed(0)
    vertices = 20 * np.random.random(shape)
    data = (vertices, 'ellipse')
    layer = Shapes(data, shape_type='rectangle')
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    data = [(vertices[i], 'ellipse') for i in range(shape[0])]
    layer = Shapes(data, shape_type='rectangle')
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_changing_shapes(ten_four_corner, twenty_four_corner):
    """Test changing Shapes data."""
    layer = Shapes(ten_four_corner)
    assert layer.nshapes == len(ten_four_corner)
    layer.data = twenty_four_corner
    assert layer.nshapes == len(twenty_four_corner)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, twenty_four_corner, strict=False)
        ]
    )
    assert layer.ndim == twenty_four_corner[0].shape[1]
    assert np.all([s == 'rectangle' for s in layer.shape_type])

    # setting data with shape type
    data_a = (ten_four_corner, 'ellipse')
    layer.data = data_a
    assert layer.nshapes == len(ten_four_corner)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, ten_four_corner, strict=False)
        ]
    )
    assert layer.ndim == ten_four_corner[0].shape[1]
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    # setting data with fewer shapes
    smaller_data = ten_four_corner[:5]
    current_edge_color = layer._data_view.edge_color
    current_edge_width = layer._data_view.edge_widths
    current_face_color = layer._data_view.face_color
    current_z = layer._data_view.z_indices

    layer.data = smaller_data
    assert layer.nshapes == len(smaller_data)
    assert np.allclose(layer._data_view.edge_color, current_edge_color[:5])
    assert np.allclose(layer._data_view.face_color, current_face_color[:5])
    assert np.allclose(layer._data_view.edge_widths, current_edge_width[:5])
    assert np.allclose(layer._data_view.z_indices, current_z[:5])

    # setting data with added shapes
    current_edge_color = layer._data_view.edge_color
    current_edge_width = layer._data_view.edge_widths
    current_face_color = layer._data_view.face_color
    current_z = layer._data_view.z_indices

    bigger_data = twenty_four_corner
    layer.data = bigger_data
    assert layer.nshapes == len(bigger_data)
    assert np.allclose(layer._data_view.edge_color[:5], current_edge_color)
    assert np.allclose(layer._data_view.face_color[:5], current_face_color)
    assert np.allclose(layer._data_view.edge_widths[:5], current_edge_width)
    assert np.allclose(layer._data_view.z_indices[:5], current_z)


def test_changing_shape_type():
    """Test changing shape type"""
    np.random.seed(0)
    rectangles = 20 * np.random.random((10, 4, 2))
    layer = Shapes(rectangles, shape_type='rectangle')
    layer.shape_type = 'ellipse'
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_adding_shapes(polygons, ten_four_corner):
    """Test adding shapes."""
    # Start with polygons with different numbers of points
    # shape_type = ['polygon'] * 5 + ['rectangle'] * 3 + ['ellipse'] * 2
    layer = Shapes(polygons, shape_type='polygon')
    new_shape_type = ['rectangle'] * 6 + ['ellipse'] * 4
    layer.add(ten_four_corner, shape_type=new_shape_type)
    all_data = polygons + ten_four_corner
    all_shape_type = ['polygon'] * len(polygons) + new_shape_type
    assert layer.nshapes == len(all_data)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, all_data, strict=False)
        ]
    )
    assert layer.ndim == 2
    assert np.all(
        [
            s == so
            for s, so in zip(layer.shape_type, all_shape_type, strict=False)
        ]
    )


def test_adding_shapes_per_shape(polygons, ten_four_corner):
    # test adding data with shape_type
    layer = Shapes(polygons, shape_type='polygon')
    new_shape_type = ['ellipse'] * 6 + ['rectangle'] * 4
    new_data = list(zip(ten_four_corner, new_shape_type, strict=False))
    layer.add(new_data)
    all_vertices = polygons + ten_four_corner
    all_shape_type = ['polygon'] * len(polygons) + new_shape_type
    assert layer.nshapes == len(all_vertices)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, all_vertices, strict=False)
        ]
    )
    assert layer.ndim == 2
    assert np.all(
        [
            s == so
            for s, so in zip(layer.shape_type, all_shape_type, strict=False)
        ]
    )


def test_adding_shapes_to_empty():
    """Test adding shapes to empty."""
    data = np.empty((0, 0, 2))
    np.random.seed(0)
    layer = Shapes(np.empty((0, 0, 2)))
    assert len(layer.data) == 0

    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)).astype(np.float32)
        for i in range(5)
    ] + list(np.random.random((5, 4, 2)).astype(np.float32))
    shape_type = ['path'] * 5 + ['rectangle'] * 3 + ['ellipse'] * 2

    layer.add(data, shape_type=shape_type)
    assert layer.nshapes == len(data)
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, data, strict=False)
        ]
    )
    assert layer.ndim == 2
    assert np.all(
        [s == so for s, so in zip(layer.shape_type, shape_type, strict=False)]
    )


def test_selecting_shapes():
    """Test selecting shapes."""
    data = 20 * np.random.random((10, 4, 2))
    np.random.seed(0)
    layer = Shapes(data)
    layer.selected_data = {0, 1}
    assert layer.selected_data == {0, 1}

    layer.selected_data = {9}
    assert layer.selected_data == {9}

    layer.selected_data = set()
    assert layer.selected_data == set()


def test_removing_all_shapes_empty_list():
    """Test removing all shapes with an empty list."""
    data = 20 * np.random.random((10, 4, 2))
    np.random.seed(0)
    layer = Shapes(data)
    layer.events.data = Mock()
    old_data = layer.data
    assert layer.nshapes == 10

    layer.data = []
    assert layer.nshapes == 0
    assert layer.events.data.call_args_list[0][1] == {
        'value': old_data,
        'action': ActionType.REMOVING,
        'data_indices': tuple(i for i in range(len(old_data))),
        'vertex_indices': ((),),
    }
    assert layer.events.data.call_args_list[1][1] == {
        'value': layer.data,
        'action': ActionType.REMOVED,
        'data_indices': (),
        'vertex_indices': ((),),
    }


def test_removing_all_shapes_empty_array():
    """Test removing all shapes with an empty list."""
    data = 20 * np.random.random((10, 4, 2))
    np.random.seed(0)
    layer = Shapes(data)
    layer.events.data = Mock()
    old_data = layer.data
    assert layer.nshapes == 10

    layer.data = np.empty((0, 2))
    assert layer.nshapes == 0
    assert layer.events.data.call_args_list[0][1] == {
        'value': old_data,
        'action': ActionType.REMOVING,
        'data_indices': tuple(i for i in range(len(old_data))),
        'vertex_indices': ((),),
    }
    assert layer.events.data.call_args_list[1][1] == {
        'value': layer.data,
        'action': ActionType.REMOVED,
        'data_indices': (),
        'vertex_indices': ((),),
    }


def test_removing_selected_shapes():
    """Test removing selected shapes."""
    np.random.seed(0)
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)).astype(np.float32)
        for i in range(5)
    ] + list(np.random.random((5, 4, 2)).astype(np.float32))
    shape_type = ['polygon'] * 5 + ['rectangle'] * 3 + ['ellipse'] * 2
    layer = Shapes(data, shape_type=shape_type)
    layer.events.data = Mock()
    old_data = layer.data
    # With nothing selected no points should be removed
    layer.remove_selected()
    layer.events.data.assert_not_called()
    assert len(layer.data) == len(data)

    # Select three shapes and remove them
    selection = {1, 7, 8}
    layer.selected_data = selection
    layer.remove_selected()
    assert layer.events.data.call_args_list[0][1] == {
        'value': old_data,
        'action': ActionType.REMOVING,
        'data_indices': tuple(
            selection,
        ),
        'vertex_indices': ((),),
    }
    assert layer.events.data.call_args_list[1][1] == {
        'value': layer.data,
        'action': ActionType.REMOVED,
        'data_indices': tuple(
            selection,
        ),
        'vertex_indices': ((),),
    }

    keep = [0, *range(2, 7)] + [9]
    data_keep = [data[i] for i in keep]
    shape_type_keep = [shape_type[i] for i in keep]
    assert len(layer.data) == len(data_keep)
    assert len(layer.selected_data) == 0
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, data_keep, strict=False)
        ]
    )
    assert layer.ndim == 2
    assert np.all(
        [
            s == so
            for s, so in zip(layer.shape_type, shape_type_keep, strict=False)
        ]
    )


def test_changing_modes():
    """Test changing modes."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    layer = Shapes(data)
    assert layer.mode == 'pan_zoom'
    assert layer.mouse_pan is True

    layer.mode = 'select'
    assert layer.mode == 'select'
    assert layer.mouse_pan is False

    layer.mode = 'direct'
    assert layer.mode == 'direct'
    assert layer.mouse_pan is False

    layer.mode = 'vertex_insert'
    assert layer.mode == 'vertex_insert'
    assert layer.mouse_pan is False

    layer.mode = 'vertex_remove'
    assert layer.mode == 'vertex_remove'
    assert layer.mouse_pan is False

    layer.mode = 'add_rectangle'
    assert layer.mode == 'add_rectangle'
    assert layer.mouse_pan is False

    layer.mode = 'add_ellipse'
    assert layer.mode == 'add_ellipse'
    assert layer.mouse_pan is False

    layer.mode = 'add_line'
    assert layer.mode == 'add_line'
    assert layer.mouse_pan is False

    layer.mode = 'add_path'
    assert layer.mode == 'add_path'
    assert layer.mouse_pan is False

    layer.mode = 'add_polygon'
    assert layer.mode == 'add_polygon'
    assert layer.mouse_pan is False

    layer.mode = 'pan_zoom'
    assert layer.mode == 'pan_zoom'
    assert layer.mouse_pan is True


def test_name():
    """Test setting layer name."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    layer = Shapes(data)
    assert layer.name == 'Shapes'

    layer = Shapes(data, name='random')
    assert layer.name == 'random'

    layer.name = 'shps'
    assert layer.name == 'shps'


def test_visiblity():
    """Test setting layer visibility."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    layer = Shapes(data)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer = Shapes(data, visible=False)
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_opacity():
    """Test setting opacity."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    # Check default opacity value of 0.7
    assert layer.opacity == 0.7

    # Select data and change opacity of selection
    layer.selected_data = {0, 1}
    assert layer.opacity == 0.7
    layer.opacity = 0.5
    assert layer.opacity == 0.5

    # Add new shape and test its width
    new_shape = np.random.random((1, 4, 2))
    layer.selected_data = set()
    layer.add(new_shape)
    assert layer.opacity == 0.5

    # Instantiate with custom opacity
    layer2 = Shapes(data, opacity=0.2)
    assert layer2.opacity == 0.2

    # Check removing data shouldn't change opacity
    layer2.selected_data = {0, 2}
    layer2.remove_selected()
    assert len(layer2.data) == shape[0] - 2
    assert layer2.opacity == 0.2


def test_blending():
    """Test setting layer blending."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    layer = Shapes(data)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Shapes(data, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


@pytest.mark.parametrize('attribute', ['edge', 'face'])
def test_switch_color_mode(attribute):
    """Test switching between color modes"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    # create a continuous property with a known value in the last element
    continuous_prop = np.random.random((shape[0],))
    continuous_prop[-1] = 1
    properties = {
        'shape_truthiness': continuous_prop,
        'shape_type': _make_cycled_properties(['A', 'B'], shape[0]),
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
    layer = Shapes(data, properties=properties, **args)

    layer_color_mode = getattr(layer, f'{attribute}_color_mode')
    layer_color = getattr(layer, f'{attribute}_color')
    assert layer_color_mode == 'direct'
    np.testing.assert_allclose(
        layer_color, np.repeat([initial_color], shape[0], axis=0)
    )

    # there should not be an edge_color_property
    color_property = getattr(layer, f'_{attribute}_color_property')
    assert color_property == ''

    # transitioning to colormap should raise a warning
    # because there isn't an edge color property yet and
    # the first property in shapes.properties is being automatically selected
    with pytest.warns(UserWarning):
        setattr(layer, f'{attribute}_color_mode', 'colormap')
    color_property = getattr(layer, f'_{attribute}_color_property')
    assert color_property == next(iter(properties))
    layer_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(layer_color[-1], [1, 1, 1, 1])

    # switch to color cycle
    setattr(layer, f'{attribute}_color_mode', 'cycle')
    setattr(layer, f'{attribute}_color', 'shape_type')
    color = getattr(layer, f'{attribute}_color')
    layer_color = transform_color(color_cycle * int(shape[0] / 2))
    np.testing.assert_allclose(color, layer_color)

    # switch back to direct, edge_colors shouldn't change
    setattr(layer, f'{attribute}_color_mode', 'direct')
    new_edge_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(new_edge_color, color)


@pytest.mark.parametrize('attribute', ['edge', 'face'])
def test_color_direct(attribute: str):
    """Test setting face/edge color directly."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer_kwargs = {f'{attribute}_color': 'black'}
    layer = Shapes(data, **layer_kwargs)
    color_array = transform_color(['black'] * shape[0])

    current_color = getattr(layer, f'current_{attribute}_color')
    layer_color = getattr(layer, f'{attribute}_color')
    assert current_color == 'black'
    assert len(layer.edge_color) == shape[0]
    np.testing.assert_allclose(color_array, layer_color)

    # With no data selected changing color has no effect
    setattr(layer, f'current_{attribute}_color', 'blue')
    current_color = getattr(layer, f'current_{attribute}_color')
    assert current_color == 'blue'
    np.testing.assert_allclose(color_array, layer_color)

    # Select data and change edge color of selection
    selected_data = {0, 1}
    layer.selected_data = {0, 1}
    current_color = getattr(layer, f'current_{attribute}_color')
    assert current_color == 'black'
    setattr(layer, f'current_{attribute}_color', 'green')
    colorarray_green = transform_color(['green'] * len(layer.selected_data))
    color_array[list(selected_data)] = colorarray_green
    layer_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(color_array, layer_color)
    # Add new shape and test its color
    new_shape = np.random.random((1, 4, 2))
    layer.selected_data = set()
    setattr(layer, f'current_{attribute}_color', 'blue')
    layer.add(new_shape)
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

    # set the color directly
    setattr(layer, f'{attribute}_color', 'black')
    color_array = np.tile([[0, 0, 0, 1]], (len(layer.data), 1))
    layer_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(color_array, layer_color)


@pytest.mark.parametrize('attribute', ['edge', 'face'])
def test_single_shape_properties(attribute):
    """Test creating single shape with properties"""
    shape = (4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer_kwargs = {f'{attribute}_color': 'red'}
    layer = Shapes(data, **layer_kwargs)
    layer_color = getattr(layer, f'{attribute}_color')
    assert len(layer_color) == 1
    np.testing.assert_allclose([1, 0, 0, 1], layer_color[0])


color_cycle_str = ['red', 'blue']
color_cycle_rgb = [[1, 0, 0], [0, 0, 1]]
color_cycle_rgba = [[1, 0, 0, 1], [0, 0, 1, 1]]


@pytest.mark.parametrize('attribute', ['edge', 'face'])
@pytest.mark.parametrize(
    'color_cycle',
    [color_cycle_str, color_cycle_rgb, color_cycle_rgba],
)
def test_color_cycle(attribute, color_cycle):
    """Test setting edge/face color with a color cycle list"""
    # create Shapes using list color cycle
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'shape_type': _make_cycled_properties(['A', 'B'], shape[0])}
    shapes_kwargs = {
        'properties': properties,
        f'{attribute}_color': 'shape_type',
        f'{attribute}_color_cycle': color_cycle,
    }
    layer = Shapes(data, **shapes_kwargs)

    np.testing.assert_equal(layer.properties, properties)
    color_array = transform_color(
        list(islice(cycle(color_cycle), 0, shape[0]))
    )
    layer_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(layer_color, color_array)

    # Add new shape and test its color
    new_shape = np.random.random((1, 4, 2))
    layer.selected_data = {0}
    layer.add(new_shape)
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

    # refresh colors
    layer.refresh_colors(update_color_mapping=True)

    # test adding a shape with a new property value
    layer.selected_data = {}
    current_properties = layer.current_properties
    current_properties['shape_type'] = np.array(['new'])
    layer.current_properties = current_properties
    new_shape_2 = np.random.random((1, 4, 2))
    layer.add(new_shape_2)
    color_cycle_map = getattr(layer, f'{attribute}_color_cycle_map')

    assert 'new' in color_cycle_map
    np.testing.assert_allclose(
        color_cycle_map['new'], np.squeeze(transform_color(color_cycle[0]))
    )


@pytest.mark.parametrize('attribute', ['edge', 'face'])
def test_add_color_cycle_to_empty_layer(attribute):
    """Test adding a shape to an empty layer when edge/face color is a color cycle

    See: https://github.com/napari/napari/pull/1069
    """
    default_properties = {'shape_type': np.array(['A'])}
    color_cycle = ['red', 'blue']
    shapes_kwargs = {
        'property_choices': default_properties,
        f'{attribute}_color': 'shape_type',
        f'{attribute}_color_cycle': color_cycle,
    }
    layer = Shapes(**shapes_kwargs)

    # verify the current_edge_color is correct
    expected_color = transform_color(color_cycle[0])
    current_color = getattr(layer, f'_current_{attribute}_color')
    np.testing.assert_allclose(current_color, expected_color)

    # add a shape
    np.random.seed(0)
    new_shape = 20 * np.random.random((1, 4, 2))
    layer.add(new_shape)
    props = {'shape_type': np.array(['A'])}
    expected_color = np.array([[1, 0, 0, 1]])
    np.testing.assert_equal(layer.properties, props)
    attribute_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(attribute_color, expected_color)

    # add a shape with a new property
    layer.selected_data = []
    layer.current_properties = {'shape_type': np.array(['B'])}
    new_shape_2 = 20 * np.random.random((1, 4, 2))
    layer.add(new_shape_2)
    new_color = np.array([0, 0, 1, 1])
    expected_color = np.vstack((expected_color, new_color))
    new_properties = {'shape_type': np.array(['A', 'B'])}
    attribute_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(attribute_color, expected_color)
    np.testing.assert_equal(layer.properties, new_properties)


@pytest.mark.parametrize('attribute', ['edge', 'face'])
def test_adding_value_color_cycle(attribute):
    """Test that adding values to properties used to set a color cycle
    and then calling Shapes.refresh_colors() performs the update and adds the
    new value to the face/edge_color_cycle_map.
    """
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'shape_type': _make_cycled_properties(['A', 'B'], shape[0])}
    color_cycle = ['red', 'blue']
    shapes_kwargs = {
        'properties': properties,
        f'{attribute}_color': 'shape_type',
        f'{attribute}_color_cycle': color_cycle,
    }
    layer = Shapes(data, **shapes_kwargs)

    # make shape 0 shape_type C
    shape_types = layer.properties['shape_type']
    shape_types[0] = 'C'
    layer.properties['shape_type'] = shape_types
    layer.refresh_colors(update_color_mapping=False)

    color_cycle_map = getattr(layer, f'{attribute}_color_cycle_map')
    color_map_keys = [*color_cycle_map]
    assert 'C' in color_map_keys


@pytest.mark.parametrize('attribute', ['edge', 'face'])
def test_color_colormap(attribute):
    """Test setting edge/face color with a colormap"""
    # create Shapes using with a colormap
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'shape_type': _make_cycled_properties([0, 1.5], shape[0])}
    shapes_kwargs = {
        'properties': properties,
        f'{attribute}_color': 'shape_type',
        f'{attribute}_colormap': 'gray',
    }
    layer = Shapes(data, **shapes_kwargs)
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

    # Add new shape and test its color
    new_shape = np.random.random((1, 4, 2))
    layer.selected_data = {0}
    layer.add(new_shape)
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
    layer.refresh_colors(update_color_mapping=False)
    attribute_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(attribute_color[-2], [0.5, 0.5, 0.5, 1])

    # change the colormap
    new_colormap = 'viridis'
    setattr(layer, f'{attribute}_colormap', new_colormap)
    attribute_colormap = getattr(layer, f'{attribute}_colormap')
    assert attribute_colormap.name == new_colormap


@pytest.mark.parametrize('attribute', ['edge', 'face'])
def test_colormap_without_properties(attribute):
    """Setting the colormode to colormap should raise an exception"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)

    with pytest.raises(ValueError, match='must be a valid Shapes.properties'):
        setattr(layer, f'{attribute}_color_mode', 'colormap')


@pytest.mark.parametrize('attribute', ['edge', 'face'])
def test_colormap_with_categorical_properties(attribute):
    """Setting the colormode to colormap should raise an exception"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'shape_type': _make_cycled_properties(['A', 'B'], shape[0])}
    layer = Shapes(data, properties=properties)

    with pytest.raises(TypeError), pytest.warns(UserWarning):
        setattr(layer, f'{attribute}_color_mode', 'colormap')


@pytest.mark.parametrize('attribute', ['edge', 'face'])
def test_add_colormap(attribute):
    """Test  directly adding a vispy Colormap object"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    annotations = {'shape_type': _make_cycled_properties([0, 1.5], shape[0])}
    color_kwarg = f'{attribute}_color'
    colormap_kwarg = f'{attribute}_colormap'
    args = {color_kwarg: 'shape_type', colormap_kwarg: 'viridis'}
    layer = Shapes(data, properties=annotations, **args)

    setattr(layer, f'{attribute}_colormap', 'gray')
    layer_colormap = getattr(layer, f'{attribute}_colormap')
    assert layer_colormap.name == 'gray'


def test_edge_width():
    """Test setting edge width."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.current_edge_width == 1
    assert len(layer.edge_width) == shape[0]
    assert layer.edge_width == [1] * shape[0]

    # With no data selected changing edge width has no effect
    layer.current_edge_width = 2
    assert layer.current_edge_width == 2
    assert layer.edge_width == [1] * shape[0]

    # Select data and change edge color of selection
    layer.selected_data = {0, 1}
    assert layer.current_edge_width == 1
    layer.current_edge_width = 3
    assert layer.edge_width == [3] * 2 + [1] * (shape[0] - 2)

    # Add new shape and test its width
    new_shape = np.random.random((1, 4, 2))
    layer.selected_data = set()
    layer.current_edge_width = 4
    layer.add(new_shape)
    assert layer.edge_width == [3] * 2 + [1] * (shape[0] - 2) + [4]

    # Instantiate with custom edge width
    layer = Shapes(data, edge_width=5)
    assert layer.current_edge_width == 5

    # Instantiate with custom edge width list
    width_list = [2, 3] * 5
    layer = Shapes(data, edge_width=width_list)
    assert layer.current_edge_width == 1
    assert layer.edge_width == width_list

    # Add new shape and test its color
    layer.current_edge_width = 4
    layer.add(new_shape)
    assert len(layer.edge_width) == shape[0] + 1
    assert layer.edge_width == [*width_list, 4]

    # Check removing data adjusts colors correctly
    layer.selected_data = {0, 2}
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    assert len(layer.edge_width) == shape[0] - 1
    assert layer.edge_width == [width_list[1]] + width_list[3:] + [4]

    # Test setting edge width with number
    layer.edge_width = 4
    assert all(width == 4 for width in layer.edge_width)

    # Test setting edge width with list
    new_widths = [2] * 5 + [3] * 4
    layer.edge_width = new_widths
    assert layer.edge_width == new_widths

    # Test setting with incorrect size list throws error
    new_widths = [2, 3]
    with pytest.raises(ValueError, match='does not match number of shapes'):
        layer.edge_width = new_widths


def test_z_index():
    """Test setting z-index during instantiation."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.z_index == [0] * shape[0]

    # Instantiate with custom z-index
    layer = Shapes(data, z_index=4)
    assert layer.z_index == [4] * shape[0]

    # Instantiate with custom z-index list
    z_index_list = [2, 3] * 5
    layer = Shapes(data, z_index=z_index_list)
    assert layer.z_index == z_index_list

    # Add new shape and its z-index
    new_shape = np.random.random((1, 4, 2))
    layer.add(new_shape)
    assert len(layer.z_index) == shape[0] + 1
    assert layer.z_index == [*z_index_list, 4]

    # Check removing data adjusts colors correctly
    layer.selected_data = {0, 2}
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    assert len(layer.z_index) == shape[0] - 1
    assert layer.z_index == [z_index_list[1]] + z_index_list[3:] + [4]

    # Test setting index with number
    layer.z_index = 4
    assert all(idx == 4 for idx in layer.z_index)

    # Test setting index with list
    new_z_indices = [2] * 5 + [3] * 4
    layer.z_index = new_z_indices
    assert layer.z_index == new_z_indices

    # Test setting with incorrect size list throws error
    new_z_indices = [2, 3]
    with pytest.raises(ValueError, match='does not match number of shapes'):
        layer.z_index = new_z_indices


def test_move_to_front():
    """Test moving shapes to front."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    z_index_list = [2, 3] * 5
    layer = Shapes(data, z_index=z_index_list)
    assert layer.z_index == z_index_list

    # Move selected shapes to front
    layer.selected_data = {0, 2}
    layer.move_to_front()
    assert layer.z_index == [4] + [z_index_list[1]] + [4] + z_index_list[3:]


def test_move_to_back():
    """Test moving shapes to back."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    z_index_list = [2, 3] * 5
    layer = Shapes(data, z_index=z_index_list)
    assert layer.z_index == z_index_list

    # Move selected shapes to front
    layer.selected_data = {0, 2}
    layer.move_to_back()
    assert layer.z_index == [1] + [z_index_list[1]] + [1] + z_index_list[3:]


def test_interaction_box():
    """Test the creation of the interaction box."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer._selected_box is None

    layer.selected_data = {0}
    assert len(layer._selected_box) == 10

    layer.selected_data = {0, 1}
    assert len(layer._selected_box) == 10

    layer.selected_data = set()
    assert layer._selected_box is None


def test_copy_and_paste():
    """Test copying and pasting selected shapes."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    # Clipboard starts empty
    assert layer._clipboard == {}

    # Pasting empty clipboard doesn't change data
    layer._paste_data()
    assert len(layer.data) == 10

    # Copying with nothing selected leave clipboard empty
    layer._copy_data()
    assert layer._clipboard == {}

    # Copying and pasting with two shapes selected adds to clipboard and data
    layer.selected_data = {0, 1}
    layer._copy_data()
    layer._paste_data()
    assert len(layer._clipboard) > 0
    assert len(layer.data) == shape[0] + 2
    assert np.all(
        [
            np.array_equal(a, b)
            for a, b in zip(layer.data[:2], layer.data[-2:], strict=False)
        ]
    )

    # Pasting again adds two more shapes to data
    layer._paste_data()
    assert len(layer.data) == shape[0] + 4
    assert np.all(
        [
            np.array_equal(a, b)
            for a, b in zip(layer.data[:2], layer.data[-2:], strict=False)
        ]
    )

    # Unselecting everything and copying and pasting will empty the clipboard
    # and add no new data
    layer.selected_data = set()
    layer._copy_data()
    layer._paste_data()
    assert layer._clipboard == {}
    assert len(layer.data) == shape[0] + 4


def test_value():
    """Test getting the value of the data at the current coordinates."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[-1, :] = [[0, 0], [0, 10], [10, 0], [10, 10]]
    assert Shapes([]).get_value((0,) * 2) == (None, None)
    layer = Shapes(data)
    value = layer.get_value((0,) * 2)
    assert value == (9, None)

    layer.mode = 'select'
    layer.selected_data = {9}
    value = layer.get_value((0,) * 2)
    assert value == (9, 7)

    layer = Shapes(data + 5)
    value = layer.get_value((0,) * 2)
    assert value == (None, None)


def test_value_non_convex():
    """Test getting the value of the data at the current coordinates."""
    data = [
        [[0, 0], [10, 10], [20, 0], [10, 5]],
    ]
    layer = Shapes(data, shape_type='polygon')
    assert layer.get_value((1,) * 2) == (0, None)
    assert layer.get_value((10, 3)) == (None, None)


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
        ((0, 5, 21, 15), [0, 1, 0, 0], [1, 2, 3], True, (1, 1, 2, 1), 2),
        ((0, 5, 21, 15), [0, -1, 0, 0], [1, 2, 3], True, (1, 1, 2, 1), 0),
        ((0, 5, 0, 0), [0, 1, 0, 0], [1, 2, 3], True, (1, 1, 2, 1), None),
    ],
)
def test_value_3d(
    position, view_direction, dims_displayed, world, scale, expected
):
    """Test get_value in 3D with and without scale"""
    data = np.array(
        [
            [
                [0, 10, 10, 10],
                [0, 10, 10, 30],
                [0, 10, 30, 30],
                [0, 10, 30, 10],
            ],
            [[0, 7, 10, 10], [0, 7, 10, 30], [0, 7, 30, 30], [0, 7, 30, 10]],
            [[0, 5, 10, 10], [0, 5, 10, 30], [0, 5, 30, 30], [0, 5, 30, 10]],
        ]
    )
    layer = Shapes(data, scale=scale)
    layer._slice_dims(Dims(ndim=4, ndisplay=3, point=(0, 0, 0, 0)))
    value, _ = layer.get_value(
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
    """Test converting values and coords to message."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    msg = layer.get_status((0,) * 2)
    assert isinstance(msg, dict)


def test_message_3d():
    """Test converting values and coords to message in 3D."""
    shape = (10, 4, 3)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    msg = layer.get_status(
        (0, 0, 0), view_direction=[1, 0, 0], dims_displayed=[0, 1, 2]
    )
    assert isinstance(msg, dict)


def test_thumbnail():
    """Test the image thumbnail for square data."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[-1, :] = [[0, 0], [0, 20], [20, 0], [20, 20]]
    layer = Shapes(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_to_masks():
    """Test the mask generation."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    masks = layer.to_masks()
    assert masks.ndim == 3
    assert len(masks) == shape[0]

    masks = layer.to_masks(mask_shape=[20, 20])
    assert masks.shape == (shape[0], 20, 20)


def test_to_masks_default_shape():
    """Test that labels data generation preserves origin at (0, 0).

    See https://github.com/napari/napari/issues/3401
    """
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape) + [50, 100]
    layer = Shapes(data)
    masks = layer.to_masks()
    assert len(masks) == 10
    assert 50 <= masks[0].shape[0] <= 71
    assert 100 <= masks[0].shape[1] <= 121


def test_to_labels():
    """Test the labels generation."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    labels = layer.to_labels()
    assert labels.ndim == 2
    assert len(np.unique(labels)) <= 11

    labels = layer.to_labels(labels_shape=[20, 20])
    assert labels.shape == (20, 20)
    assert len(np.unique(labels)) <= 11


def test_to_labels_default_shape():
    """Test that labels data generation preserves origin at (0, 0).

    See https://github.com/napari/napari/issues/3401
    """
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape) + [50, 100]
    layer = Shapes(data)
    labels = layer.to_labels()
    assert labels.ndim == 2
    assert 1 < len(np.unique(labels)) <= 11
    assert 50 <= labels.shape[0] <= 71
    assert 100 <= labels.shape[1] <= 121


def test_to_labels_3D():
    """Test label generation for 3D data"""
    data = [
        [[0, 100, 100], [0, 100, 200], [0, 200, 200], [0, 200, 100]],
        [[1, 125, 125], [1, 125, 175], [1, 175, 175], [1, 175, 125]],
        [[2, 100, 100], [2, 100, 200], [2, 200, 200], [2, 200, 100]],
    ]
    labels_shape = (3, 300, 300)
    layer = Shapes(np.array(data), shape_type='polygon')
    labels = layer.to_labels(labels_shape=labels_shape)
    assert np.array_equal(labels.shape, labels_shape)
    assert np.array_equal(np.unique(labels), [0, 1, 2, 3])


def test_add_single_shape_consistent_properties():
    """Test adding a single shape ensures correct number of added properties"""
    data = [
        np.array([[100, 200], [200, 300]]),
        np.array([[300, 400], [400, 500]]),
    ]
    properties = {'index': [1, 2]}
    layer = Shapes(
        np.array(data), shape_type='rectangle', properties=properties
    )

    layer.add(np.array([[500, 600], [700, 800]]))
    assert len(layer.properties['index']) == 3
    assert layer.properties['index'][2] == 2


def test_add_shapes_consistent_properties():
    """Test adding multiple shapes ensures correct number of added properties"""
    data = [
        np.array([[100, 200], [200, 300]]),
        np.array([[300, 400], [400, 500]]),
    ]
    properties = {'index': [1, 2]}
    layer = Shapes(
        np.array(data), shape_type='rectangle', properties=properties
    )

    layer.add(
        [
            np.array([[500, 600], [700, 800]]),
            np.array([[700, 800], [800, 900]]),
        ]
    )
    assert len(layer.properties['index']) == 4
    assert layer.properties['index'][2] == 2
    assert layer.properties['index'][3] == 2


def test_world_data_extent():
    """Test extent after applying transforms."""
    data = [(7, -5, 0), (-2, 0, 15), (4, 30, 12)]
    layer = Shapes([data, np.add(data, [2, -3, 0])], shape_type='polygon')
    min_val = (-2, -8, 0)
    max_val = (9, 30, 15)
    extent = np.array((min_val, max_val))
    check_layer_world_data_extent(layer, extent, (3, 1, 1), (10, 20, 5))


def test_set_data_3d():
    """Test to reproduce https://github.com/napari/napari/issues/4527"""
    lines = [
        np.array([[0, 0, 0], [500, 0, 0]]),
        np.array([[0, 0, 0], [0, 300, 0]]),
        np.array([[0, 0, 0], [0, 0, 200]]),
    ]
    shapes = Shapes(lines, shape_type='line')
    shapes._slice_dims(Dims(ndim=3, ndisplay=3))
    shapes.data = lines


def test_editing_4d():
    viewer = ViewerModel()
    viewer.add_shapes(
        ndim=4,
        name='rois',
        edge_color='red',
        face_color=np.array([0, 0, 0, 0]),
        edge_width=1,
    )

    viewer.layers['rois'].add(
        [
            np.array(
                [
                    [1, 4, 1.7, 4.9],
                    [1, 4, 1.7, 13.1],
                    [1, 4, 13.5, 13.1],
                    [1, 4, 13.5, 4.9],
                ]
            )
        ]
    )
    # check if set data doe not end with an exception
    # https://github.com/napari/napari/issues/5379
    viewer.layers['rois'].data = [
        np.around(x) for x in viewer.layers['rois'].data
    ]


def test_shapes_data_setter_emits_event():
    data = np.random.random((4, 2))
    emitted_events = Mock()
    layer = Shapes(data)
    layer.events.data.connect(emitted_events)
    layer.data = np.random.random((4, 2))
    assert emitted_events.call_count == 2


def test_shapes_add_delete_only_emit_two_events():
    data = np.random.random((4, 2))
    emitted_events = Mock()
    layer = Shapes(data)
    layer.events.data.connect(emitted_events)
    layer.add(np.random.random((4, 2)))
    assert emitted_events.call_count == 2
    layer.selected_data = {1}
    layer.remove_selected()
    assert emitted_events.call_count == 4


def test_clean_selection_on_set_data():
    data = [[[0, 0], (10, 10)], [[0, 15], [10, 25]]]
    layer = Shapes(data)
    layer.selected_data = {0}
    layer.data = [[[0, 0], (10, 10)]]
    assert layer.selected_data == set()


def test_docstring():
    validate_all_params_in_docstring(Shapes)
    validate_kwargs_sorted(Shapes)
