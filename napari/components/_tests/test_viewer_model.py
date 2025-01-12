import time

import numpy as np
import pytest
from npe2 import DynamicPlugin

from napari._tests.utils import (
    count_warning_events,
    good_layer_data,
    layer_test_data,
)
from napari.components import ViewerModel
from napari.errors import MultipleReaderError, ReaderPluginError
from napari.errors.reader_errors import NoAvailableReaderError
from napari.layers import Image
from napari.layers.shapes._tests.conftest import (
    ten_four_corner,  # noqa: F401
)  # import to not put this data in top level conftest.py
from napari.settings import get_settings
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from napari.utils.events.event import WarningEmitter


def test_viewer_model():
    """Test instantiating viewer model."""
    viewer = ViewerModel()
    assert viewer.title == 'napari'
    assert len(viewer.layers) == 0
    assert viewer.dims.ndim == 2

    # Create viewer model with custom title
    viewer = ViewerModel(title='testing')
    assert viewer.title == 'testing'


def test_add_image():
    """Test adding image."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    assert len(viewer.layers) == 1
    assert np.array_equal(viewer.layers[0].data, data)
    assert viewer.dims.ndim == 2


def test_add_image_multichannel_share_memory():
    viewer = ViewerModel()
    image = np.random.random((10, 5, 64, 64))
    layers = viewer.add_image(image, channel_axis=1)
    for layer in layers:
        assert np.may_share_memory(image, layer.data)


def test_add_image_colormap_variants():
    """Test adding image with all valid colormap argument types."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    # as string
    assert viewer.add_image(data, colormap='green')

    # as string that is valid, but not a default colormap
    assert viewer.add_image(data, colormap='fire')

    # as tuple
    cmap_tuple = ('my_colormap', Colormap(['g', 'm', 'y']))
    assert viewer.add_image(data, colormap=cmap_tuple)

    # as dict
    cmap_dict = {'your_colormap': Colormap(['g', 'r', 'y'])}
    assert viewer.add_image(data, colormap=cmap_dict)

    # as Colormap instance
    blue_cmap = AVAILABLE_COLORMAPS['blue']
    assert viewer.add_image(data, colormap=blue_cmap)

    # string values must be known colormap types
    with pytest.raises(KeyError) as err:
        viewer.add_image(data, colormap='nonsense')

    assert 'Colormap "nonsense" not found' in str(err.value)

    # lists are only valid with channel_axis
    with pytest.raises(TypeError) as err:
        viewer.add_image(data, colormap=['green', 'red'])

    assert "did you mean to specify a 'channel_axis'" in str(err.value)


def test_add_image_accepts_all_arguments_as_sequence():
    """See https://github.com/napari/napari/pull/7089."""
    viewer = ViewerModel(ndisplay=3)
    img = viewer.add_image(np.random.rand(2, 2))
    viewer.add_image(**img._get_state())


def test_add_volume():
    """Test adding volume."""
    viewer = ViewerModel(ndisplay=3)
    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    viewer.add_image(data)
    assert len(viewer.layers) == 1
    assert np.array_equal(viewer.layers[0].data, data)
    assert viewer.dims.ndim == 3


def test_add_multiscale():
    """Test adding image multiscale."""
    viewer = ViewerModel()
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    viewer.add_image(data, multiscale=True)
    assert len(viewer.layers) == 1
    # this is not an nd array but a list of ndarray.
    # I think that might be a edge case of MultiScaleData.
    assert viewer.layers[0].data == data
    assert viewer.dims.ndim == 2


def test_add_multiscale_image_with_negative_floats():
    """See https://github.com/napari/napari/issues/5257"""
    viewer = ViewerModel()
    shapes = [(20, 10), (10, 5)]
    data = [np.zeros(s, dtype=np.float64) for s in shapes]
    data[0][-4:, -2:] = -1
    data[1][-2:, -1:] = -1

    viewer.add_image(data, multiscale=True)

    assert len(viewer.layers) == 1
    # this is not an nd array but a list of ndarray.
    # I think that might be a edge case of MultiScaleData.
    assert viewer.layers[0].data == data
    assert viewer.dims.ndim == 2


def test_add_labels():
    """Test adding labels image."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    viewer.add_labels(data)
    assert len(viewer.layers) == 1
    assert np.array_equal(viewer.layers[0].data, data)
    assert viewer.dims.ndim == 2


def test_add_points():
    """Test adding points."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    viewer.add_points(data)
    assert len(viewer.layers) == 1
    assert np.array_equal(viewer.layers[0].data, data)
    assert viewer.dims.ndim == 2


def test_single_point_dims():
    """Test dims of a Points layer with a single 3D point."""
    viewer = ViewerModel()
    shape = (1, 3)
    data = np.zeros(shape)
    viewer.add_points(data)
    assert all(r == (0.0, 0.0, 1.0) for r in viewer.dims.range)


def test_add_empty_points_to_empty_viewer():
    viewer = ViewerModel()
    layer = viewer.add_points(name='empty points')
    assert layer.ndim == 2
    layer.add([1000.0, 27.0])
    assert layer.data.shape == (1, 2)


def test_add_empty_points_on_top_of_image():
    viewer = ViewerModel()
    image = np.random.random((8, 64, 64))
    # add_image always returns the corresponding layer
    _ = viewer.add_image(image)
    layer = viewer.add_points(ndim=3)
    assert layer.ndim == 3
    layer.add([5.0, 32.0, 61.0])
    assert layer.data.shape == (1, 3)


def test_add_empty_shapes_layer():
    viewer = ViewerModel()
    image = np.random.random((8, 64, 64))
    # add_image always returns the corresponding layer
    _ = viewer.add_image(image)
    layer = viewer.add_shapes(ndim=3)
    assert layer.ndim == 3


def test_add_vectors():
    """Test adding vectors."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = 20 * np.random.random((10, 2, 2))
    viewer.add_vectors(data)
    assert len(viewer.layers) == 1
    assert np.array_equal(viewer.layers[0].data, data)
    assert viewer.dims.ndim == 2


def test_add_shapes(ten_four_corner):  # noqa: F811
    """Test adding shapes."""
    viewer = ViewerModel()
    viewer.add_shapes(ten_four_corner)
    assert len(viewer.layers) == 1
    assert np.array_equal(viewer.layers[0].data, ten_four_corner)
    assert viewer.dims.ndim == 2


def test_add_surface():
    """Test adding 3D surface."""
    viewer = ViewerModel()
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    viewer.add_surface(data)
    assert len(viewer.layers) == 1
    assert np.all(
        [
            np.array_equal(vd, d)
            for vd, d in zip(viewer.layers[0].data, data, strict=False)
        ]
    )
    assert viewer.dims.ndim == 3


def test_mix_dims():
    """Test adding images of mixed dimensionality."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    assert len(viewer.layers) == 1
    assert np.array_equal(viewer.layers[0].data, data)
    assert viewer.dims.ndim == 2

    data = np.random.random((6, 10, 15))
    viewer.add_image(data)
    assert len(viewer.layers) == 2
    assert np.array_equal(viewer.layers[1].data, data)
    assert viewer.dims.ndim == 3


def test_new_labels_empty():
    """Test adding new labels layer to empty viewer."""
    viewer = ViewerModel()
    viewer._new_labels()
    assert len(viewer.layers) == 1
    assert np.max(viewer.layers[0].data) == 0
    assert viewer.dims.ndim == 2
    # Default shape when no data is present is 512x512
    np.testing.assert_equal(viewer.layers[0].data.shape, (512, 512))


def test_new_labels_image():
    """Test adding new labels layer with image present."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer._new_labels()
    assert len(viewer.layers) == 2
    assert np.max(viewer.layers[1].data) == 0
    assert viewer.dims.ndim == 2
    np.testing.assert_equal(viewer.layers[1].data.shape, (10, 15))
    np.testing.assert_equal(viewer.layers[1].scale, (1, 1))
    np.testing.assert_equal(viewer.layers[1].translate, (0, 0))


def test_new_labels_scaled_image():
    """Test adding new labels layer with scaled image present."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data, scale=(3, 3))
    viewer._new_labels()
    assert len(viewer.layers) == 2
    assert np.max(viewer.layers[1].data) == 0
    assert viewer.dims.ndim == 2
    np.testing.assert_equal(viewer.layers[1].data.shape, (10, 15))
    np.testing.assert_equal(viewer.layers[1].scale, (3, 3))
    np.testing.assert_equal(viewer.layers[1].translate, (0, 0))


def test_new_labels_scaled_translated_image():
    """Test adding new labels layer with transformed image present."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data, scale=(3, 3), translate=(20, -5))
    viewer._new_labels()
    assert len(viewer.layers) == 2
    assert np.max(viewer.layers[1].data) == 0
    assert viewer.dims.ndim == 2
    np.testing.assert_almost_equal(viewer.layers[1].data.shape, (10, 15))
    np.testing.assert_almost_equal(viewer.layers[1].scale, (3, 3))
    np.testing.assert_almost_equal(viewer.layers[1].translate, (20, -5))


def test_new_points():
    """Test adding new points layer."""
    # Add labels to empty viewer
    viewer = ViewerModel()
    viewer.add_points()
    assert len(viewer.layers) == 1
    assert len(viewer.layers[0].data) == 0
    assert viewer.dims.ndim == 2

    # Add points with image already present
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer.add_points()
    assert len(viewer.layers) == 2
    assert len(viewer.layers[1].data) == 0
    assert viewer.dims.ndim == 2


def test_view_centering_with_points_add():
    """Test if the viewer is only centered when the first
    points were added
    Regression test for issue  #3803
    """
    image = np.zeros((5, 10, 10))

    viewer = ViewerModel()
    viewer.add_image(image)
    assert tuple(viewer.dims.point) == (2, 4, 4)

    viewer.dims.set_point(0, 0)
    # viewer point shouldn't change after this
    assert tuple(viewer.dims.point) == (0, 4, 4)

    pts_layer = viewer.add_points(ndim=3)
    assert tuple(viewer.dims.point) == (0, 4, 4)

    pts_layer.add([(0, 8, 8)])
    assert tuple(viewer.dims.point) == (0, 4, 4)


def test_view_centering_with_scale():
    """Regression test for issue #5735"""
    image = np.zeros((5, 10, 10))

    viewer = ViewerModel()
    viewer.add_image(image, scale=(1, 1, 1))
    assert tuple(viewer.dims.point) == (2, 4, 4)

    viewer.layers.pop()
    viewer.add_image(image, scale=(2, 1, 1))
    assert tuple(viewer.dims.point) == (4, 4, 4)


def test_new_shapes():
    """Test adding new shapes layer."""
    # Add labels to empty viewer
    viewer = ViewerModel()
    viewer.add_shapes()
    assert len(viewer.layers) == 1
    assert len(viewer.layers[0].data) == 0
    assert viewer.dims.ndim == 2

    # Add points with image already present
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer.add_shapes()
    assert len(viewer.layers) == 2
    assert len(viewer.layers[1].data) == 0
    assert viewer.dims.ndim == 2


def test_swappable_dims():
    """Test swapping dims after adding layers."""
    viewer = ViewerModel()
    np.random.seed(0)
    image_data = np.random.random((7, 12, 10, 15))
    image_name = viewer.add_image(image_data).name
    assert np.array_equal(
        viewer.layers[image_name]._data_view, image_data[3, 5, :, :]
    )

    points_data = np.random.randint(6, size=(10, 4))
    viewer.add_points(points_data)

    vectors_data = np.random.randint(6, size=(10, 2, 4))
    viewer.add_vectors(vectors_data)

    labels_data = np.random.randint(20, size=(7, 12, 10, 15))
    labels_name = viewer.add_labels(labels_data).name
    # midpoints indices into the data below depend on the data range.
    # This depends on the values in vectors_data and thus the random seed.
    assert np.array_equal(
        viewer.layers[labels_name]._slice.image.raw, labels_data[3, 5, :, :]
    )

    # Swap dims
    viewer.dims.order = [0, 2, 1, 3]
    assert viewer.dims.order == (0, 2, 1, 3)
    assert np.array_equal(
        viewer.layers[image_name]._data_view, image_data[3, :, 4, :]
    )
    assert np.array_equal(
        viewer.layers[labels_name]._slice.image.raw, labels_data[3, :, 4, :]
    )


def test_grid():
    "Test grid_view"
    viewer = ViewerModel()

    np.random.seed(0)
    # Add image
    for _i in range(6):
        data = np.random.random((15, 15))
        viewer.add_image(data)
    assert not viewer.grid.enabled
    assert viewer.grid.actual_shape(6) == (1, 1)
    assert viewer.grid.stride == 1
    translations = [layer._translate_grid for layer in viewer.layers]
    expected_translations = np.zeros((6, 2))
    np.testing.assert_allclose(translations, expected_translations)

    # enter grid view
    viewer.grid.enabled = True
    assert viewer.grid.enabled
    assert viewer.grid.actual_shape(6) == (2, 3)
    assert viewer.grid.stride == 1
    translations = [layer._translate_grid for layer in viewer.layers]
    expected_translations = [
        [0, 0],
        [0, 15],
        [0, 30],
        [15, 0],
        [15, 15],
        [15, 30],
    ]
    np.testing.assert_allclose(translations, expected_translations[::-1])

    # return to stack view
    viewer.grid.enabled = False
    assert not viewer.grid.enabled
    assert viewer.grid.actual_shape(6) == (1, 1)
    assert viewer.grid.stride == 1
    translations = [layer._translate_grid for layer in viewer.layers]
    expected_translations = np.zeros((6, 2))
    np.testing.assert_allclose(translations, expected_translations)

    # reenter grid view with new stride
    viewer.grid.stride = -2
    viewer.grid.enabled = True
    assert viewer.grid.enabled
    assert viewer.grid.actual_shape(6) == (2, 2)
    assert viewer.grid.stride == -2
    translations = [layer._translate_grid for layer in viewer.layers]
    expected_translations = [
        [0, 0],
        [0, 0],
        [0, 15],
        [0, 15],
        [15, 0],
        [15, 0],
    ]
    np.testing.assert_allclose(translations, expected_translations)


def test_add_remove_layer_dims_change():
    """Test dims change appropriately when adding and removing layers."""
    np.random.seed(0)
    viewer = ViewerModel()

    # Check ndim starts at 2
    assert viewer.dims.ndim == 2

    # Check ndim increase to 3 when 3D data added
    data = np.random.random((10, 15, 20))
    layer = viewer.add_image(data)
    assert len(viewer.layers) == 1
    assert np.array_equal(viewer.layers[0].data, data)
    assert viewer.dims.ndim == 3

    # Remove layer and check ndim returns to 2
    viewer.layers.remove(layer)
    assert len(viewer.layers) == 0
    assert viewer.dims.ndim == 2


@pytest.mark.parametrize('data', good_layer_data)
def test_add_layer_from_data(data):
    # make sure adding valid layer data calls the proper corresponding add_*
    # method for all layer types
    viewer = ViewerModel()
    viewer._add_layer_from_data(*data)

    # make sure a layer of the correct type got added
    assert len(viewer.layers) == 1
    expected_layer_type = data[2] if len(data) > 2 else 'image'
    assert viewer.layers[0]._type_string == expected_layer_type


def test_add_layer_from_data_raises():
    # make sure that adding invalid data or kwargs raises the right errors
    viewer = ViewerModel()
    # unrecognized layer type raises Value Error
    with pytest.raises(ValueError, match='Unrecognized layer_type'):
        # (even though there is an add_layer method)
        viewer._add_layer_from_data(
            np.random.random((10, 10)), layer_type='layer'
        )

    # even with the correct meta kwargs, the underlying add_* method may raise
    with pytest.raises(
        ValueError, match='data does not have suitable dimensions'
    ):
        viewer._add_layer_from_data(
            np.random.random((10, 10, 6)), {'rgb': True}
        )

    # using a kwarg in the meta dict that is invalid for the corresponding
    # add_* method raises a TypeError
    with pytest.raises(TypeError):
        viewer._add_layer_from_data(
            np.random.random((10, 2, 2)) * 20,
            {'rgb': True},  # vectors do not have an 'rgb' kwarg
            layer_type='vectors',
        )


def test_naming():
    """Test unique naming in LayerList."""
    viewer = ViewerModel()
    viewer.add_image(np.random.random((10, 10)), name='img')
    viewer.add_image(np.random.random((10, 10)), name='img')

    assert [lay.name for lay in viewer.layers] == ['img', 'img [1]']

    viewer.layers[1].name = 'chg'
    assert [lay.name for lay in viewer.layers] == ['img', 'chg']

    viewer.layers[0].name = 'chg'
    assert [lay.name for lay in viewer.layers] == ['chg [1]', 'chg']


def test_selection():
    """Test only last added is selected."""
    viewer = ViewerModel()
    viewer.add_image(np.random.random((10, 10)))
    assert viewer.layers[0] in viewer.layers.selection

    viewer.add_image(np.random.random((10, 10)))
    assert viewer.layers.selection == {viewer.layers[-1]}

    viewer.add_image(np.random.random((10, 10)))
    assert viewer.layers.selection == {viewer.layers[-1]}

    viewer.layers.selection.update(viewer.layers)
    viewer.add_image(np.random.random((10, 10)))
    assert viewer.layers.selection == {viewer.layers[-1]}


def test_add_delete_layers():
    """Test adding and deleting layers with different dims."""
    viewer = ViewerModel()
    np.random.seed(0)
    viewer.add_image(np.random.random((5, 5, 10, 15)))
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 4
    viewer.add_image(np.random.random((5, 6, 5, 10, 15)))
    assert len(viewer.layers) == 2
    assert viewer.dims.ndim == 5
    viewer.layers.remove_selected()
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 4


def test_active_layer():
    """Test active layer is correct as layer selections change."""
    viewer = ViewerModel()
    np.random.seed(0)
    # Check no active layer present
    assert viewer.layers.selection.active is None

    # Check added layer is active
    viewer.add_image(np.random.random((5, 5, 10, 15)))
    assert len(viewer.layers) == 1
    assert viewer.layers.selection.active == viewer.layers[0]
    assert viewer.layers[0]._highlight_visible

    # Check newly added layer is active
    viewer.add_image(np.random.random((5, 6, 5, 10, 15)))
    assert len(viewer.layers) == 2
    assert viewer.layers.selection.active == viewer.layers[1]
    assert not viewer.layers[0]._highlight_visible
    assert viewer.layers[1]._highlight_visible

    # Check no active layer after unselecting all
    viewer.layers.selection.clear()
    assert viewer.layers.selection.active is None
    assert not viewer.layers[0]._highlight_visible
    assert not viewer.layers[1]._highlight_visible

    # Check selected layer is active
    viewer.layers.selection.add(viewer.layers[0])
    assert viewer.layers.selection.active == viewer.layers[0]
    assert viewer.layers[0]._highlight_visible
    assert not viewer.layers[1]._highlight_visible

    # Check no layer is active if both layers are selected
    viewer.layers.selection.add(viewer.layers[1])
    assert viewer.layers.selection.active is None
    assert not viewer.layers[0]._highlight_visible
    assert not viewer.layers[1]._highlight_visible


def test_active_layer_status_update():
    """Test status updates from active layer on cursor move."""
    viewer = ViewerModel()
    np.random.seed(0)
    viewer.add_image(np.random.random((5, 5, 10, 15)))
    viewer.add_image(np.random.random((5, 6, 5, 10, 15)))
    assert len(viewer.layers) == 2
    assert viewer.layers.selection.active == viewer.layers[1]

    # wait 1 s to avoid the cursor event throttling
    time.sleep(1)
    viewer.mouse_over_canvas = True
    viewer.cursor.position = [1, 1, 1, 1, 1]
    assert viewer._calc_status_from_cursor()[
        0
    ] == viewer.layers.selection.active.get_status(
        viewer.cursor.position, world=True
    )


def test_active_layer_cursor_size():
    """Test cursor size update on active layer."""
    viewer = ViewerModel()
    np.random.seed(0)
    viewer.add_image(np.random.random((10, 10)))
    # Base layer has a default cursor size of 1
    assert viewer.cursor.size == 1

    viewer.add_labels(np.random.randint(0, 10, size=(10, 10)))
    assert len(viewer.layers) == 2
    assert viewer.layers.selection.active == viewer.layers[1]

    viewer.layers[1].mode = 'paint'
    # Labels layer has a default cursor size of 10
    # due to paintbrush
    assert viewer.cursor.size == 10


def test_cursor_ndim_matches_layer():
    """Test cursor position ndim matches viewer ndim after update."""
    viewer = ViewerModel()
    np.random.seed(0)
    im = viewer.add_image(np.random.random((10, 10)))
    assert viewer.dims.ndim == 2
    assert len(viewer.cursor.position) == 2

    im.data = np.random.random((10, 10, 10))
    assert viewer.dims.ndim == 3
    assert len(viewer.cursor.position) == 3

    im.data = np.random.random((10, 10))
    assert viewer.dims.ndim == 2
    assert len(viewer.cursor.position) == 2


def test_sliced_world_extent():
    """Test world extent after adding layers and slicing."""
    np.random.seed(0)
    viewer = ViewerModel()

    # Empty data is taken to be 512 x 512
    np.testing.assert_allclose(
        viewer._sliced_extent_world_augmented[0], (-0.5, -0.5)
    )
    np.testing.assert_allclose(
        viewer._sliced_extent_world_augmented[1], (511.5, 511.5)
    )

    # Add one layer
    viewer.add_image(
        np.random.random((6, 10, 15)), scale=(3, 1, 1), translate=(10, 20, 5)
    )
    np.testing.assert_allclose(
        viewer.layers._extent_world_augmented[0], (8.5, 19.5, 4.5)
    )
    np.testing.assert_allclose(
        viewer.layers._extent_world_augmented[1], (26.5, 29.5, 19.5)
    )
    np.testing.assert_allclose(
        viewer._sliced_extent_world_augmented[0], (19.5, 4.5)
    )
    np.testing.assert_allclose(
        viewer._sliced_extent_world_augmented[1], (29.5, 19.5)
    )

    # Change displayed dims order
    viewer.dims.order = (1, 2, 0)
    np.testing.assert_allclose(
        viewer._sliced_extent_world_augmented[0], (4.5, 8.5)
    )
    np.testing.assert_allclose(
        viewer._sliced_extent_world_augmented[1], (19.5, 26.5)
    )


def test_camera():
    """Test camera."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    viewer.add_image(data)
    assert len(viewer.layers) == 1
    assert np.array_equal(viewer.layers[0].data, data)
    assert viewer.dims.ndim == 3

    assert viewer.dims.ndisplay == 2
    assert viewer.camera.center == (0, 7, 9.5)
    assert viewer.camera.angles == (0, 0, 90)

    viewer.dims.ndisplay = 3
    assert viewer.dims.ndisplay == 3
    assert viewer.camera.center == (4.5, 7, 9.5)
    assert viewer.camera.angles == (0, 0, 90)

    viewer.dims.ndisplay = 2
    assert viewer.dims.ndisplay == 2
    assert viewer.camera.center == (0, 7, 9.5)
    assert viewer.camera.angles == (0, 0, 90)


def test_update_scale():
    viewer = ViewerModel()
    np.random.seed(0)
    shape = (10, 15, 20)
    data = np.random.random(shape)
    viewer.add_image(data)
    assert viewer.dims.range == tuple((0.0, x - 1, 1.0) for x in shape)
    scale = (3.0, 2.0, 1.0)
    viewer.layers[0].scale = scale
    assert viewer.dims.range == tuple(
        (0.0, (x - 1) * s, s) for x, s in zip(shape, scale, strict=False)
    )


@pytest.mark.parametrize(('Layer', 'data', 'ndim'), layer_test_data)
def test_add_remove_layer_no_callbacks(Layer, data, ndim):
    """Test all callbacks for layer emmitters removed."""
    viewer = ViewerModel()

    layer = Layer(data)
    # Check layer has been correctly created
    assert layer.ndim == ndim

    # Check that no internal callbacks have been registered
    assert len(layer.events.callbacks) == 0
    for em in layer.events.emitters.values():
        assert len(em.callbacks) == count_warning_events(em.callbacks)

    viewer.layers.append(layer)
    # Check layer added correctly
    assert len(viewer.layers) == 1

    # check that adding a layer created new callbacks
    assert any(len(em.callbacks) > 0 for em in layer.events.emitters.values())

    viewer.layers.remove(layer)
    # Check layer added correctly
    assert len(viewer.layers) == 0

    # Check that all callbacks have been removed
    assert len(layer.events.callbacks) == 0
    for em in layer.events.emitters.values():
        assert len(em.callbacks) == count_warning_events(em.callbacks)


@pytest.mark.parametrize(('Layer', 'data', 'ndim'), layer_test_data)
def test_add_remove_layer_external_callbacks(Layer, data, ndim):
    """Test external callbacks for layer emmitters preserved."""
    viewer = ViewerModel()

    layer = Layer(data)
    # Check layer has been correctly created
    assert layer.ndim == ndim

    # Connect a custom callback
    def my_custom_callback():
        return

    layer.events.connect(my_custom_callback)

    # Check that no internal callbacks have been registered
    assert len(layer.events.callbacks) == 1
    for em in layer.events.emitters.values():
        if not isinstance(em, WarningEmitter):
            assert len(em.callbacks) == count_warning_events(em.callbacks) + 1

    viewer.layers.append(layer)
    # Check layer added correctly
    assert len(viewer.layers) == 1

    # check that adding a layer created new callbacks
    assert any(
        len(em.callbacks) > count_warning_events(em.callbacks)
        for em in layer.events.emitters.values()
    )

    viewer.layers.remove(layer)
    # Check layer added correctly
    assert len(viewer.layers) == 0

    # Check that all internal callbacks have been removed
    assert len(layer.events.callbacks) == 1
    for em in layer.events.emitters.values():
        if not isinstance(em, WarningEmitter):
            assert len(em.callbacks) == count_warning_events(em.callbacks) + 1


@pytest.mark.parametrize(
    'field', ['camera', 'cursor', 'dims', 'grid', 'layers']
)
def test_not_mutable_fields(field):
    """Test appropriate fields are not mutable."""
    viewer = ViewerModel()

    # Check attribute lives on the viewer
    assert hasattr(viewer, field)
    # Check attribute does not have an event emitter
    assert not hasattr(viewer.events, field)

    # Check attribute is not settable
    with pytest.raises((TypeError, ValueError)) as err:
        setattr(viewer, field, 'test')

    assert 'has allow_mutation set to False and cannot be assigned' in str(
        err.value
    )


@pytest.mark.parametrize(('Layer', 'data', 'ndim'), layer_test_data)
def test_status_tooltip(Layer, data, ndim):
    viewer = ViewerModel()
    viewer.tooltip.visible = True
    layer = Layer(data)
    viewer.layers.append(layer)
    viewer.cursor.position = (1,) * ndim


def test_viewer_object_event_sources():
    viewer = ViewerModel()
    assert viewer.cursor.events.source is viewer.cursor
    assert viewer.camera.events.source is viewer.camera


def test_open_or_get_error_multiple_readers(tmp_plugin: DynamicPlugin):
    """Assert error is returned when multiple plugins are available to read."""
    viewer = ViewerModel()
    tmp2 = tmp_plugin.spawn(register=True)

    @tmp_plugin.contribute.reader(filename_patterns=['*.fake'])
    def _(path): ...

    @tmp2.contribute.reader(filename_patterns=['*.fake'])
    def _(path): ...

    with pytest.raises(
        MultipleReaderError, match='Multiple plugins found capable'
    ):
        viewer._open_or_raise_error(['my_file.fake'])


def test_open_or_get_error_no_plugin():
    """Assert error is raised when no plugin is available."""
    viewer = ViewerModel()

    with pytest.raises(
        NoAvailableReaderError, match='No plugin found capable of reading'
    ):
        viewer._open_or_raise_error(['my_file.fake'])


def test_open_or_get_error_builtins(builtins: DynamicPlugin, tmp_path):
    """Test builtins is available to read npy files."""
    viewer = ViewerModel()

    f_pth = tmp_path / 'my-file.npy'
    data = np.random.random((10, 10))
    np.save(f_pth, data)

    added = viewer._open_or_raise_error([str(f_pth)])
    assert len(added) == 1
    layer = added[0]
    assert isinstance(layer, Image)
    np.testing.assert_allclose(layer.data, data)
    assert layer.source.reader_plugin == builtins.name


def test_open_or_get_error_prefered_plugin(
    tmp_path, builtins: DynamicPlugin, tmp_plugin: DynamicPlugin
):
    """Test plugin preference is respected."""
    viewer = ViewerModel()
    pth = tmp_path / 'my-file.npy'
    np.save(pth, np.random.random((10, 10)))

    @tmp_plugin.contribute.reader(filename_patterns=['*.npy'])
    def _(path): ...

    get_settings().plugins.extension2reader = {'*.npy': builtins.name}

    added = viewer._open_or_raise_error([str(pth)])
    assert len(added) == 1
    assert added[0].source.reader_plugin == builtins.name


def test_open_or_get_error_cant_find_plugin(tmp_path, builtins: DynamicPlugin):
    """Test user is warned and only plugin used if preferred plugin missing."""
    viewer = ViewerModel()
    pth = tmp_path / 'my-file.npy'
    np.save(pth, np.random.random((10, 10)))

    get_settings().plugins.extension2reader = {'*.npy': 'fake-reader'}

    with pytest.warns(RuntimeWarning, match="Can't find fake-reader plugin"):
        added = viewer._open_or_raise_error([str(pth)])
    assert len(added) == 1
    assert added[0].source.reader_plugin == builtins.name


def test_open_or_get_error_no_prefered_plugin_many_available(
    tmp_plugin: DynamicPlugin,
):
    """Test MultipleReaderError raised if preferred plugin missing."""
    viewer = ViewerModel()
    tmp2 = tmp_plugin.spawn(register=True)

    @tmp_plugin.contribute.reader(filename_patterns=['*.fake'])
    def _(path): ...

    @tmp2.contribute.reader(filename_patterns=['*.fake'])
    def _(path): ...

    get_settings().plugins.extension2reader = {'*.fake': 'not-a-plugin'}

    with pytest.warns(RuntimeWarning, match="Can't find not-a-plugin plugin"):
        with pytest.raises(
            MultipleReaderError, match='Multiple plugins found capable'
        ):
            viewer._open_or_raise_error(['my_file.fake'])


def test_open_or_get_error_preferred_fails(builtins, tmp_path):
    viewer = ViewerModel()
    pth = tmp_path / 'my-file.npy'

    get_settings().plugins.extension2reader = {'*.npy': builtins.name}

    with pytest.raises(
        ReaderPluginError, match='Tried opening with napari, but failed.'
    ):
        viewer._open_or_raise_error([str(pth)])


def test_slice_order_with_mixed_dims():
    viewer = ViewerModel(ndisplay=2)
    image_2d = viewer.add_image(np.zeros((4, 5)))
    image_3d = viewer.add_image(np.zeros((3, 4, 5)))
    image_4d = viewer.add_image(np.zeros((2, 3, 4, 5)))

    # With standard ordering, the shapes of the slices match,
    # so are trivially numpy-broadcastable.
    assert image_2d._slice.image.view.shape == (4, 5)
    assert image_3d._slice.image.view.shape == (4, 5)
    assert image_4d._slice.image.view.shape == (4, 5)

    viewer.dims.order = (2, 1, 0, 3)

    # With non-standard ordering, the shapes of the slices do not match,
    # and are not numpy-broadcastable.
    assert image_2d._slice.image.view.shape == (4, 5)
    assert image_3d._slice.image.view.shape == (3, 5)
    assert image_4d._slice.image.view.shape == (2, 5)


def test_make_layer_visible_after_slicing():
    """See https://github.com/napari/napari/issues/6760"""
    viewer = ViewerModel(ndisplay=2)
    data = np.array([np.ones((2, 2)) * i for i in range(3)])
    layer: Image = viewer.add_image(data)
    layer.visible = False
    assert viewer.dims.current_step[0] != 0
    assert not np.array_equal(layer._slice.image.raw, data[0])

    viewer.dims.current_step = (0, 0, 0)
    layer.visible = True

    np.testing.assert_array_equal(layer._slice.image.raw, data[0])


def test_get_status_text():
    viewer = ViewerModel(ndisplay=2)
    viewer.mouse_over_canvas = False
    assert viewer._calc_status_from_cursor() is None
    viewer.mouse_over_canvas = True
    assert viewer._calc_status_from_cursor() == ('Ready', '')
    viewer.cursor.position = (1, 2)
    viewer.add_labels(
        np.zeros((10, 10), dtype='uint8'), features={'a': [1, 2]}
    )
    viewer.tooltip.visible = False
    assert viewer._calc_status_from_cursor() == (
        {
            'coordinates': ' [1 2]: 0; a: 1',
            'layer_base': 'Labels',
            'layer_name': 'Labels',
            'plugin': '',
            'source_type': '',
        },
        '',
    )
    viewer.tooltip.visible = True
    assert viewer._calc_status_from_cursor() == (
        {
            'coordinates': ' [1 2]: 0; a: 1',
            'layer_base': 'Labels',
            'layer_name': 'Labels',
            'plugin': '',
            'source_type': '',
        },
        'a: 1',
    )
    viewer.update_status_from_cursor()
    assert viewer.status == {
        'coordinates': ' [1 2]: 0; a: 1',
        'layer_base': 'Labels',
        'layer_name': 'Labels',
        'plugin': '',
        'source_type': '',
    }
    assert viewer.tooltip.text == 'a: 1'
