import os

import npe2
import numpy as np
import pytest

from napari.components import LayerList
from napari.layers import Image
from napari.layers.layergroup import LayerGroup


@pytest.fixture(params=[LayerList, LayerGroup])
def LayersClass(request):
    return request.param


def test_empty_layers_list(LayersClass):
    """
    Test instantiating an empty LayersClass object
    """
    layers = LayersClass()

    assert len(layers) == 0


def test_initialize_from_list(LayersClass):
    layers = LayersClass(
        [Image(np.random.random((10, 10))), Image(np.random.random((10, 10)))]
    )
    assert len(layers) == 2


def test_adding_layer(LayersClass):
    """
    Test adding a Layer
    """
    layers = LayersClass()
    layer = Image(np.random.random((10, 10)))
    layers.append(layer)

    # LayersClass should err if you add anything other than a layer
    with pytest.raises(TypeError):
        layers.append('something')

    assert len(layers) == 1


def test_removing_layer(LayersClass):
    """
    Test removing a Layer
    """
    layers = LayersClass()
    layer = Image(np.random.random((10, 10)))
    layers.append(layer)
    layers.remove(layer)

    assert len(layers) == 0


def test_popping_layer(LayersClass):
    """Test popping a layer off LayersClass."""
    layers = LayersClass()
    layer = Image(np.random.random((10, 10)))
    layers.append(layer)
    assert len(layers) == 1
    layers.pop(0)
    assert len(layers) == 0


def test_indexing(LayersClass):
    """
    Test indexing into a LayersClass
    """
    layers = LayersClass()
    layer = Image(np.random.random((10, 10)), name='image')
    layers.append(layer)

    assert layers[0] == layer
    assert layers['image'] == layer


def test_insert(LayersClass):
    """
    Test inserting into a LayersClass
    """
    layers = LayersClass()
    layer_a = Image(np.random.random((10, 10)), name='image_a')
    layer_b = Image(np.random.random((15, 15)), name='image_b')
    layers.append(layer_a)
    layers.insert(0, layer_b)

    assert list(layers) == [layer_b, layer_a]


def test_get_index(LayersClass):
    """
    Test getting indexing from LayersClass
    """
    layers = LayersClass()
    layer_a = Image(np.random.random((10, 10)), name='image_a')
    layer_b = Image(np.random.random((15, 15)), name='image_b')
    layers.append(layer_a)
    layers.append(layer_b)

    assert layers.index(layer_a) == 0
    assert layers.index('image_a') == 0
    assert layers.index(layer_b) == 1
    assert layers.index('image_b') == 1


def test_reordering(LayersClass):
    """
    Test indexing into a LayersClass by name
    """
    layers = LayersClass()
    layer_a = Image(np.random.random((10, 10)), name='image_a')
    layer_b = Image(np.random.random((15, 15)), name='image_b')
    layer_c = Image(np.random.random((15, 15)), name='image_c')
    layers.append(layer_a)
    layers.append(layer_b)
    layers.append(layer_c)

    # Rearrange layers by tuple
    layers[:] = [layers[i] for i in (1, 0, 2)]
    assert list(layers) == [layer_b, layer_a, layer_c]

    # Reverse layers
    layers.reverse()
    assert list(layers) == [layer_c, layer_a, layer_b]


def test_clearing_LayersClass(LayersClass):
    """Test clearing layer list."""
    layers = LayersClass()
    layer = Image(np.random.random((10, 10)))
    layer2 = Image(np.random.random((10, 10)))
    layers.append(layer)
    layers.append(layer2)
    assert len(layers) == 2

    layers.clear()
    assert len(layers) == 0


def test_remove_selected(LayersClass):
    """Test removing selected layers."""
    layers = LayersClass()
    layer_a = Image(np.random.random((10, 10)))
    layer_b = Image(np.random.random((15, 15)))
    layer_c = Image(np.random.random((15, 15)))
    layers.append(layer_a)
    layers.append(layer_b)
    layers.append(layer_c)

    # remove last added layer as only one selected
    layers.selection.clear()
    layers.selection.add(layer_c)
    layers.remove_selected()
    assert list(layers) == [layer_a, layer_b]

    # select and remove all layers
    layers.select_all()
    layers.remove_selected()
    assert len(layers) == 0


@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_move_selected():
    """
    Test removing selected layers
    """
    layers = LayerList()
    layer_a = Image(np.random.random((10, 10)))
    layer_b = Image(np.random.random((15, 15)))
    layer_c = Image(np.random.random((15, 15)))
    layer_d = Image(np.random.random((15, 15)))
    layers.append(layer_a)
    layers.append(layer_b)
    layers.append(layer_c)
    layers.append(layer_d)

    # Check nothing moves if given same insert and origin
    layers.selection.clear()
    layers.move_selected(2, 2)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]
    assert layers.selection == {layer_c}

    # Move middle element to front of list and back
    layers.selection.clear()
    layers.move_selected(2, 0)
    assert list(layers) == [layer_c, layer_a, layer_b, layer_d]
    assert layers.selection == {layer_c}

    layers.selection.clear()
    layers.move_selected(0, 2)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]
    assert layers.selection == {layer_c}

    # Move middle element to end of list and back
    layers.selection.clear()
    layers.move_selected(2, 3)
    assert list(layers) == [layer_a, layer_b, layer_d, layer_c]
    assert layers.selection == {layer_c}

    layers.selection.clear()
    layers.move_selected(3, 2)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]
    assert layers.selection == {layer_c}

    # Select first two layers only
    layers.selection = layers[:2]

    # Move unselected middle element to front of list even if others selected
    layers.move_selected(2, 0)
    assert list(layers) == [layer_c, layer_a, layer_b, layer_d]
    # Move selected first element back to middle of list
    layers.move_selected(0, 2)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]

    # Select first two layers only
    layers.selection = layers[:2]
    # Check nothing moves if given same insert and origin and multiple selected
    layers.move_selected(0, 0)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]
    assert layers.selection == {layer_a, layer_b}

    # Check nothing moves if given same insert and origin and multiple selected
    layers.move_selected(1, 1)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]
    assert layers.selection == {layer_a, layer_b}

    # Move first two selected to middle of list
    layers.move_selected(0, 2)
    assert list(layers) == [layer_c, layer_a, layer_b, layer_d]
    assert layers.selection == {layer_a, layer_b}

    # Move middle selected to front of list
    layers.move_selected(2, 0)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]
    assert layers.selection == {layer_a, layer_b}

    # Move first two selected to middle of list
    layers.move_selected(1, 2)
    assert list(layers) == [layer_c, layer_a, layer_b, layer_d]
    assert layers.selection == {layer_a, layer_b}

    # Move middle selected to front of list
    layers.move_selected(1, 0)
    assert list(layers) == [layer_a, layer_b, layer_c, layer_d]
    assert layers.selection == {layer_a, layer_b}

    # Select first and third layers only
    layers.selection = layers[::2]
    # Move selection together to middle
    layers.move_selected(2, 2)
    assert list(layers) == [layer_b, layer_a, layer_c, layer_d]
    assert layers.selection == {layer_a, layer_c}
    layers.move_multiple((1, 0, 2, 3), 0)

    # Move selection together to middle
    layers.move_selected(0, 1)
    assert list(layers) == [layer_b, layer_a, layer_c, layer_d]
    assert layers.selection == {layer_a, layer_c}
    layers.move_multiple((1, 0, 2, 3), 0)

    # Move selection together to end
    layers.move_selected(2, 3)
    assert list(layers) == [layer_b, layer_d, layer_a, layer_c]
    assert layers.selection == {layer_a, layer_c}
    layers.move_multiple((2, 0, 3, 1), 0)

    # Move selection together to end
    layers.move_selected(0, 3)
    assert list(layers) == [layer_b, layer_d, layer_a, layer_c]
    assert layers.selection == {layer_a, layer_c}
    layers.move_multiple((2, 0, 3, 1), 0)

    layer_e = Image(np.random.random((15, 15)))
    layer_f = Image(np.random.random((15, 15)))
    layers.append(layer_e)
    layers.append(layer_f)
    # Check current order is correct
    assert list(layers) == [
        layer_a,
        layer_b,
        layer_c,
        layer_d,
        layer_e,
        layer_f,
    ]
    # Select second and firth layers only
    layers.selection = {layers[1], layers[4]}

    # Move selection together to middle
    layers.move_selected(1, 2)
    assert list(layers) == [
        layer_a,
        layer_c,
        layer_b,
        layer_e,
        layer_d,
        layer_f,
    ]
    assert layers.selection == {layer_b, layer_e}


def test_toggle_visibility(LayersClass):
    """
    Test toggling layer visibility
    """
    layers = LayersClass()
    layer_a = Image(np.random.random((10, 10)))
    layer_b = Image(np.random.random((15, 15)))
    layer_c = Image(np.random.random((15, 15)))
    layer_d = Image(np.random.random((15, 15)))
    layers.append(layer_a)
    layers.append(layer_b)
    layers.append(layer_c)
    layers.append(layer_d)

    layers[0].visible = False
    layers[1].visible = True
    layers[2].visible = False
    layers[3].visible = True

    layers.select_all()
    layers.selection.remove(layers[0])

    layers.toggle_selected_visibility()

    assert [lay.visible for lay in layers] == [False, False, True, False]

    layers.toggle_selected_visibility()

    assert [lay.visible for lay in layers] == [False, True, False, True]


# the layer_data_and_types fixture is defined in napari/conftest.py
@pytest.mark.filterwarnings('ignore:distutils Version classes are deprecated')
def test_layers_save(builtins, tmpdir, layer_data_and_types, LayersClass):
    """Test saving all layer data."""
    list_of_layers, _, _, filenames = layer_data_and_types
    layers = LayersClass(list_of_layers)

    path = os.path.join(tmpdir, 'layers_folder')

    # Check folder does not exist
    assert not os.path.isdir(path)

    # Write data
    layers.save(path, plugin=builtins.name)

    # Check folder now exists
    assert os.path.isdir(path)

    # Check individual files now exist
    for f in filenames:
        assert os.path.isfile(os.path.join(path, f))

    # Check no additional files exist
    assert set(os.listdir(path)) == set(filenames)
    assert set(os.listdir(tmpdir)) == {'layers_folder'}


# the layer_data_and_types fixture is defined in napari/conftest.py
def test_layers_save_none_selected(
    builtins, tmpdir, layer_data_and_types, LayersClass
):
    """Test saving all layer data."""
    list_of_layers, _, _, filenames = layer_data_and_types
    layers = LayersClass(list_of_layers)
    layers.selection.clear()

    path = os.path.join(tmpdir, 'layers_folder')

    # Check folder does not exist
    assert not os.path.isdir(path)

    # Write data (will get a warning that nothing is selected)
    with pytest.warns(UserWarning):
        layers.save(path, selected=True, plugin=builtins.name)

    # Check folder still does not exist
    assert not os.path.isdir(path)

    # Check individual files still do not exist
    for f in filenames:
        assert not os.path.isfile(os.path.join(path, f))

    # Check no additional files exist
    assert set(os.listdir(tmpdir)) == set('')


# the layer_data_and_types fixture is defined in napari/conftest.py
def test_layers_save_selected(
    builtins, tmpdir, layer_data_and_types, LayersClass
):
    """Test saving all layer data."""
    list_of_layers, _, _, filenames = layer_data_and_types
    layers = LayersClass(list_of_layers)
    layers.selection.clear()
    layers.selection.update({layers[0], layers[2]})

    path = os.path.join(tmpdir, 'layers_folder')

    # Check folder does not exist
    assert not os.path.isdir(path)

    # Write data
    layers.save(path, selected=True, plugin=builtins.name)

    # Check folder exists
    assert os.path.isdir(path)

    # Check only appropriate files exist
    assert os.path.isfile(os.path.join(path, filenames[0]))
    assert not os.path.isfile(os.path.join(path, filenames[1]))
    assert os.path.isfile(os.path.join(path, filenames[2]))
    assert not os.path.isfile(os.path.join(path, filenames[1]))

    # Check no additional files exist
    assert set(os.listdir(path)) == {filenames[0], filenames[2]}
    assert set(os.listdir(tmpdir)) == {'layers_folder'}


# the layers fixture is defined in napari/conftest.py
@pytest.mark.filterwarnings('ignore:`np.int` is a deprecated alias for')
def test_layers_save_svg(tmpdir, layers, napari_svg_name, LayersClass):
    """Test saving all layer data to an svg."""
    pm = npe2.PluginManager.instance()
    pm.register(npe2.PluginManifest.from_distribution('napari-svg'))

    path = os.path.join(tmpdir, 'layers_file.svg')

    # Check file does not exist
    assert not os.path.isfile(path)

    # Write data
    layers.save(path, plugin=napari_svg_name)

    # Check file now exists
    assert os.path.isfile(path)


def test_world_extent(LayersClass):
    """Test world extent after adding layers."""
    np.random.seed(0)
    layers = LayersClass()

    # Empty data is taken to be 512 x 512
    np.testing.assert_allclose(layers.extent.world[0], (-0.5, -0.5))
    np.testing.assert_allclose(layers.extent.world[1], (511.5, 511.5))
    np.testing.assert_allclose(layers.extent.step, (1, 1))

    # Add one layer
    layer_a = Image(
        np.random.random((6, 10, 15)), scale=(3, 1, 1), translate=(10, 20, 5)
    )
    layers.append(layer_a)
    np.testing.assert_allclose(layer_a.extent.world[0], (8.5, 19.5, 4.5))
    np.testing.assert_allclose(layer_a.extent.world[1], (26.5, 29.5, 19.5))
    np.testing.assert_allclose(layers.extent.world[0], (8.5, 19.5, 4.5))
    np.testing.assert_allclose(layers.extent.world[1], (26.5, 29.5, 19.5))
    np.testing.assert_allclose(layers.extent.step, (3, 1, 1))

    # Add another layer
    layer_b = Image(
        np.random.random((8, 6, 15)), scale=(6, 2, 1), translate=(-5, -10, 10)
    )
    layers.append(layer_b)
    np.testing.assert_allclose(layer_b.extent.world[0], (-8, -11, 9.5))
    np.testing.assert_allclose(layer_b.extent.world[1], (40, 1, 24.5))
    np.testing.assert_allclose(layers.extent.world[0], (-8, -11, 4.5))
    np.testing.assert_allclose(layers.extent.world[1], (40, 29.5, 24.5))
    np.testing.assert_allclose(layers.extent.step, (3, 1, 1))


def test_world_extent_mixed_ndim(LayersClass):
    """Test world extent after adding layers of different dimensionality."""
    np.random.seed(0)
    layers = LayersClass()

    # Add 3D layer
    layer_a = Image(np.random.random((15, 15, 15)), scale=(4, 12, 2))
    layers.append(layer_a)
    np.testing.assert_allclose(layers.extent.world[1], (58, 174, 29))
    np.testing.assert_allclose(
        layers.extent.world[1] - layers.extent.world[0], (60, 180, 30)
    )

    # Add 2D layer
    layer_b = Image(np.random.random((10, 10)), scale=(6, 4))
    layers.append(layer_b)
    np.testing.assert_allclose(layers.extent.world[1], (58, 174, 38))
    np.testing.assert_allclose(
        layers.extent.world[1] - layers.extent.world[0], (60, 180, 40)
    )
    np.testing.assert_allclose(layers.extent.step, (4, 6, 2))


def test_world_extent_mixed_flipped(LayersClass):
    """Test world extent after adding data with a flip."""
    # Flipped data results in a negative scale value which should be
    # made positive when taking into consideration for the step size
    # calculation
    np.random.seed(0)
    layers = LayersClass()

    layer = Image(
        np.random.random((15, 15)), affine=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    )
    layers.append(layer)
    np.testing.assert_allclose(layer._data_to_world.scale, (1, -1))
    np.testing.assert_allclose(layers.extent.step, (1, 1))


def test_ndim(LayersClass):
    """Test world extent after adding layers."""
    np.random.seed(0)
    layers = LayersClass()

    assert layers.ndim == 2

    # Add one layer
    layer_a = Image(np.random.random((10, 15)))
    layers.append(layer_a)
    assert layers.ndim == 2

    # Add another layer
    layer_b = Image(np.random.random((8, 6, 15)))
    layers.append(layer_b)
    assert layers.ndim == 3

    # Remove layer
    layers.remove(layer_b)
    assert layers.ndim == 2


def test_name_uniqueness(LayersClass):
    layers = LayersClass()
    layers.append(Image(np.random.random((10, 15)), name="Image [1]"))
    layers.append(Image(np.random.random((10, 15)), name="Image"))
    layers.append(Image(np.random.random((10, 15)), name="Image"))
    assert [x.name for x in layers] == ['Image [1]', 'Image', 'Image [2]']


def test_readd_layers():
    layers = LayerList()
    imgs = []
    for _i in range(5):
        img = Image(np.random.rand(10, 10, 10))
        layers.append(img)
        imgs.append(img)

    assert layers == imgs

    with pytest.raises(ValueError):
        layers.append(imgs[1])
    assert layers == imgs

    layers[1] = layers[1]
    assert layers == imgs

    with pytest.raises(ValueError):
        layers[1] = layers[2]
    assert layers == imgs

    layers[:3] = layers[:3]
    assert layers == imgs

    # invert a section
    layers[:3] = layers[2::-1]
    assert set(layers) == set(imgs)

    with pytest.raises(ValueError):
        layers[:3] = layers[:]
    assert set(layers) == set(imgs)
