import copy

import numpy as np
import pandas as pd
import pytest

from napari._tests.utils import check_layer_world_data_extent
from napari.components.dims import Dims
from napari.layers import Surface
from napari.layers.surface.normals import SurfaceNormals
from napari.layers.surface.wireframe import SurfaceWireframe
from napari.utils._test_utils import (
    validate_all_params_in_docstring,
    validate_kwargs_sorted,
)


def test_random_surface():
    """Test instantiating Surface layer with random 2D data."""
    np.random.seed(0)
    vertices = np.random.random((10, 2))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)
    assert layer.ndim == 2
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, data, strict=False)
        ]
    )
    assert np.array_equal(layer.vertices, vertices)
    assert np.array_equal(layer.faces, faces)
    assert np.array_equal(layer.vertex_values, values)
    assert layer._data_view.shape[1] == 2
    assert layer._view_vertex_values.ndim == 1


def test_random_surface_features():
    """Test instantiating surface layer with features."""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    features = pd.DataFrame({'feature': np.random.random(10)})

    data = (vertices, faces, values)
    layer = Surface(data, features=features)
    assert 'feature' in layer.features.columns


def test_set_features_and_defaults():
    """Test setting features and defaults."""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)

    data = (vertices, faces, values)
    layer = Surface(data)

    assert layer.features.shape[1] == layer.feature_defaults.shape[1] == 0

    features = pd.DataFrame(
        {
            'str': ('a', 'b') * 5,
            'float': np.random.random(10),
        }
    )
    feature_defaults = pd.DataFrame(
        {
            'str': ('b',),
            'float': (0.5,),
        }
    )

    layer.features = features
    layer.feature_defaults = feature_defaults

    pd.testing.assert_frame_equal(layer.features, features)
    pd.testing.assert_frame_equal(layer.feature_defaults, feature_defaults)


def test_random_surface_no_values():
    """Test instantiating Surface layer with random 2D data but no vertex values."""
    np.random.seed(0)
    vertices = np.random.random((10, 2))
    faces = np.random.randint(10, size=(6, 3))
    data = (vertices, faces)
    layer = Surface(data)
    assert layer.ndim == 2
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, data, strict=False)
        ]
    )
    assert np.array_equal(layer.vertices, vertices)
    assert np.array_equal(layer.faces, faces)
    assert np.array_equal(layer.vertex_values, np.ones(len(vertices)))
    assert layer._data_view.shape[1] == 2
    assert layer._view_vertex_values.ndim == 1


def test_random_surface_clearing_vertex_values():
    """Test setting `vertex_values=None` resets values to uniform ones."""
    np.random.seed(0)
    vertices = np.random.random((10, 2))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)
    assert np.array_equal(layer.vertex_values, values)
    layer.vertex_values = None
    assert np.array_equal(layer.vertex_values, np.ones(len(vertices)))


def test_random_3D_surface():
    """Test instantiating Surface layer with random 3D data."""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)
    assert layer.ndim == 3
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, data, strict=False)
        ]
    )
    assert layer._data_view.shape[1] == 2
    assert layer._view_vertex_values.ndim == 1

    layer._slice_dims(Dims(ndim=3, ndisplay=3))
    assert layer._data_view.shape[1] == 3
    assert layer._view_vertex_values.ndim == 1


def test_random_4D_surface():
    """Test instantiating Surface layer with random 4D data."""
    np.random.seed(0)
    vertices = np.random.random((10, 4))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)
    assert layer.ndim == 4
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, data, strict=False)
        ]
    )
    assert layer._data_view.shape[1] == 2
    assert layer._view_vertex_values.ndim == 1

    layer._slice_dims(Dims(ndim=4, ndisplay=3))
    assert layer._data_view.shape[1] == 3
    assert layer._view_vertex_values.ndim == 1


def test_random_3D_timeseries_surface():
    """Test instantiating Surface layer with random 3D timeseries data."""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random((22, 10))
    data = (vertices, faces, values)
    layer = Surface(data)
    assert layer.ndim == 4
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, data, strict=False)
        ]
    )
    assert layer._data_view.shape[1] == 2
    assert layer._view_vertex_values.ndim == 1
    assert layer.extent.data[1][0] == 21

    layer._slice_dims(Dims(ndim=4, ndisplay=3))
    assert layer._data_view.shape[1] == 3
    assert layer._view_vertex_values.ndim == 1

    # If a values axis is made to be a displayed axis then no data should be
    # shown
    with pytest.warns(UserWarning):
        layer._slice_dims(Dims(ndim=4, ndisplay=3, order=(3, 0, 1, 2)))
        assert len(layer._data_view) == 0


def test_random_3D_multitimeseries_surface():
    """Test instantiating Surface layer with random 3D multitimeseries data."""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random((16, 22, 10))
    data = (vertices, faces, values)
    layer = Surface(data)
    assert layer.ndim == 5
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, data, strict=False)
        ]
    )
    assert layer._data_view.shape[1] == 2
    assert layer._view_vertex_values.ndim == 1
    assert layer.extent.data[1][0] == 15
    assert layer.extent.data[1][1] == 21

    layer._slice_dims(Dims(ndim=5, ndisplay=3))
    assert layer._data_view.shape[1] == 3
    assert layer._view_vertex_values.ndim == 1


def test_changing_surface():
    """Test changing surface layer data"""
    np.random.seed(0)
    vertices = np.random.random((10, 2))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)

    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer.data = data
    assert layer.ndim == 3
    assert np.all(
        [
            np.array_equal(ld, d)
            for ld, d in zip(layer.data, data, strict=False)
        ]
    )
    assert layer._data_view.shape[1] == 2
    assert layer._view_vertex_values.ndim == 1

    layer._slice_dims(Dims(ndim=3, ndisplay=3))
    assert layer._data_view.shape[1] == 3
    assert layer._view_vertex_values.ndim == 1


def test_visiblity():
    """Test setting layer visibility."""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer = Surface(data, visible=False)
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_surface_gamma():
    """Test setting gamma."""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)
    assert layer.gamma == 1

    # Change gamma property
    gamma = 0.7
    layer.gamma = gamma
    assert layer.gamma == gamma

    # Set gamma as keyword argument
    layer = Surface(data, gamma=gamma)
    assert layer.gamma == gamma


def test_world_data_extent():
    """Test extent after applying transforms."""
    data = [(-5, 0), (0, 15), (30, 12)]
    min_val = (-5, 0)
    max_val = (30, 15)
    layer = Surface((np.array(data), np.array((0, 1, 2)), np.array((0, 0, 0))))
    extent = np.array((min_val, max_val))
    check_layer_world_data_extent(layer, extent, (3, 1), (20, 5))


def test_shading():
    """Test setting shading"""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)

    # change shading property
    shading = 'flat'
    layer.shading = shading
    assert layer.shading == shading

    # set shading as keyword argument
    layer = Surface(data, shading=shading)
    assert layer.shading == shading


def test_texture():
    """Test setting texture"""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)

    texture = np.random.random((32, 32, 3)).astype(np.float32)
    texcoords = vertices[:, :2]
    layer = Surface(data, texture=texture, texcoords=texcoords)

    np.testing.assert_allclose(layer.texture, texture)
    np.testing.assert_allclose(layer.texcoords, texcoords)
    assert layer._has_texture

    layer.texture, layer.texcoords = None, texcoords
    assert not layer._has_texture

    layer.texture, layer.texcoords = texture, None
    assert not layer._has_texture

    layer.texture, layer.texcoords = None, None
    assert not layer._has_texture

    layer.texture, layer.texcoords = texture, texcoords
    assert layer._has_texture


def test_vertex_colors():
    """Test setting vertex colors"""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)

    vertex_colors = np.random.random((len(vertices), 3))
    layer = Surface(data, vertex_colors=vertex_colors)
    np.testing.assert_allclose(layer.vertex_colors, vertex_colors)

    layer.vertex_colors = vertex_colors**2
    np.testing.assert_allclose(layer.vertex_colors, vertex_colors**2)


@pytest.mark.parametrize(
    ('ray_start', 'ray_direction', 'expected_value', 'expected_index'),
    [
        ([0, 1, 1], [1, 0, 0], 2, 0),
        ([10, 1, 1], [-1, 0, 0], 2, 1),
    ],
)
def test_get_value_3d(
    ray_start, ray_direction, expected_value, expected_index
):
    vertices = np.array(
        [
            [3, 0, 0],
            [3, 0, 3],
            [3, 3, 0],
            [5, 0, 0],
            [5, 0, 3],
            [5, 3, 0],
            [2, 50, 50],
            [2, 50, 100],
            [2, 100, 50],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    values = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    surface_layer = Surface((vertices, faces, values))

    surface_layer._slice_dims(Dims(ndim=3, ndisplay=3))
    value, index = surface_layer.get_value(
        position=ray_start,
        view_direction=ray_direction,
        dims_displayed=[0, 1, 2],
        world=False,
    )
    assert index == expected_index
    np.testing.assert_allclose(value, expected_value)


@pytest.mark.parametrize(
    ('ray_start', 'ray_direction', 'expected_value', 'expected_index'),
    [
        ([0, 0, 1, 1], [0, 1, 0, 0], 2, 0),
        ([0, 10, 1, 1], [0, -1, 0, 0], 2, 1),
    ],
)
def test_get_value_3d_nd(
    ray_start, ray_direction, expected_value, expected_index
):
    vertices = np.array(
        [
            [0, 3, 0, 0],
            [0, 3, 0, 3],
            [0, 3, 3, 0],
            [0, 5, 0, 0],
            [0, 5, 0, 3],
            [0, 5, 3, 0],
            [0, 2, 50, 50],
            [0, 2, 50, 100],
            [0, 2, 100, 50],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    values = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    surface_layer = Surface((vertices, faces, values))

    surface_layer._slice_dims(Dims(ndim=4, ndisplay=3))
    value, index = surface_layer.get_value(
        position=ray_start,
        view_direction=ray_direction,
        dims_displayed=[1, 2, 3],
        world=False,
    )
    assert index == expected_index
    np.testing.assert_allclose(value, expected_value)


def test_surface_normals():
    """Ensure that normals can be set both with dict and SurfaceNormals.

    The model should internally always use SurfaceNormals.
    """
    vertices = np.array(
        [
            [3, 0, 0],
            [3, 0, 3],
            [3, 3, 0],
            [5, 0, 0],
            [5, 0, 3],
            [5, 3, 0],
            [2, 50, 50],
            [2, 50, 100],
            [2, 100, 50],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    values = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])

    normals = {'face': {'visible': True, 'color': 'red'}}
    surface_layer = Surface((vertices, faces, values), normals=normals)
    assert isinstance(surface_layer.normals, SurfaceNormals)
    assert surface_layer.normals.face.visible is True
    assert np.array_equal(surface_layer.normals.face.color, (1, 0, 0, 1))

    surface_layer = Surface(
        (vertices, faces, values), normals=SurfaceNormals(**normals)
    )
    assert isinstance(surface_layer.normals, SurfaceNormals)
    assert surface_layer.normals.face.visible is True
    assert np.array_equal(surface_layer.normals.face.color, (1, 0, 0, 1))


def test_surface_wireframe():
    """Ensure that wireframe can be set both with dict and SurfaceWireframe.

    The model should internally always use SurfaceWireframe.
    """
    vertices = np.array(
        [
            [3, 0, 0],
            [3, 0, 3],
            [3, 3, 0],
            [5, 0, 0],
            [5, 0, 3],
            [5, 3, 0],
            [2, 50, 50],
            [2, 50, 100],
            [2, 100, 50],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    values = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])

    wireframe = {'visible': True, 'color': 'red'}
    surface_layer = Surface((vertices, faces, values), wireframe=wireframe)
    assert isinstance(surface_layer.wireframe, SurfaceWireframe)
    assert surface_layer.wireframe.visible is True
    assert np.array_equal(surface_layer.wireframe.color, (1, 0, 0, 1))

    surface_layer = Surface(
        (vertices, faces, values), wireframe=SurfaceWireframe(**wireframe)
    )
    assert isinstance(surface_layer.wireframe, SurfaceWireframe)
    assert surface_layer.wireframe.visible is True
    assert np.array_equal(surface_layer.wireframe.color, (1, 0, 0, 1))


def test_surface_copy():
    vertices = np.array(
        [
            [3, 0, 0],
            [3, 0, 3],
            [3, 3, 0],
            [5, 0, 0],
            [5, 0, 3],
            [5, 3, 0],
            [2, 50, 50],
            [2, 50, 100],
            [2, 100, 50],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    values = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])

    l1 = Surface((vertices, faces, values))
    l2 = copy.copy(l1)
    assert l1.data[0] is not l2.data[0]


def test_surface_with_no_visible_faces():
    points = np.array([[0, 0.0, 0.0, 0.0], [0, 1.0, 0, 0], [0, 1, 1, 0]])
    faces = np.array([[0, 1, 2]])
    layer = Surface((points, faces))
    # the following with throw an exception when _view_faces
    # is non-integer values.
    with pytest.raises(
        ValueError, match='operands could not be broadcast together'
    ):
        layer._get_value_3d(
            np.array([1, 0, 0, 0]), np.array([1, 1, 0, 0]), [1, 2, 3]
        )


def test_docstring():
    validate_all_params_in_docstring(Surface)
    validate_kwargs_sorted(Surface)
