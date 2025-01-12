import numpy as np
import pytest

from napari._tests.utils import skip_local_popups, skip_on_win_ci
from napari.layers import Shapes
from napari.utils._test_utils import read_only_mouse_event
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)


@pytest.fixture
def qt_viewer(qt_viewer_):
    # show the qt_viewer and hide its welcome widget
    qt_viewer_.show()
    qt_viewer_.set_welcome_visible(False)
    return qt_viewer_


@skip_on_win_ci
@skip_local_popups
def test_z_order_adding_removing_images(make_napari_viewer):
    """Test z order is correct after adding/ removing images."""
    data = np.ones((11, 11))

    viewer = make_napari_viewer(show=True)
    viewer.add_image(data, colormap='red', name='red')
    viewer.add_image(data, colormap='green', name='green')
    viewer.add_image(data, colormap='blue', name='blue')

    # Check that blue is visible
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    # Remove and re-add image
    viewer.layers.remove('red')
    viewer.add_image(data, colormap='red', name='red')
    # Check that red is visible
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    np.testing.assert_almost_equal(screenshot[center], [255, 0, 0, 255])

    # Remove two other images
    viewer.layers.remove('green')
    viewer.layers.remove('blue')
    # Check that red is still visible
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    np.testing.assert_almost_equal(screenshot[center], [255, 0, 0, 255])

    # Add two other layers back
    viewer.add_image(data, colormap='green', name='green')
    viewer.add_image(data, colormap='blue', name='blue')
    # Check that blue is visible
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    # Hide blue
    viewer.layers['blue'].visible = False
    # Check that green is visible. Note this assert was failing before #1463
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    np.testing.assert_almost_equal(screenshot[center], [0, 255, 0, 255])


@skip_on_win_ci
@skip_local_popups
def test_z_order_images(make_napari_viewer):
    """Test changing order of images changes z order in display."""
    data = np.ones((11, 11))

    viewer = make_napari_viewer(show=True)
    viewer.add_image(data, colormap='red')
    viewer.add_image(data, colormap='blue')
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    viewer.layers.move(1, 0)
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that red is now visible
    np.testing.assert_almost_equal(screenshot[center], [255, 0, 0, 255])


@skip_on_win_ci
@skip_local_popups
def test_z_order_image_points(make_napari_viewer):
    """Test changing order of image and points changes z order in display."""
    data = np.ones((11, 11))

    viewer = make_napari_viewer(show=True)
    viewer.add_image(data, colormap='red')
    viewer.add_points([5, 5], face_color='blue', size=10)
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    viewer.layers.move(1, 0)
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that red is now visible
    np.testing.assert_almost_equal(screenshot[center], [255, 0, 0, 255])


@skip_on_win_ci
@skip_local_popups
def test_z_order_images_after_ndisplay(make_napari_viewer):
    """Test z order of images remanins constant after chaning ndisplay."""
    data = np.ones((11, 11))

    viewer = make_napari_viewer(show=True)
    viewer.add_image(data, colormap='red')
    viewer.add_image(data, colormap='blue')
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    # Switch to 3D rendering
    viewer.dims.ndisplay = 3
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is still visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    # Switch back to 2D rendering
    viewer.dims.ndisplay = 2
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is still visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])


@pytest.mark.skip('Image is 1 pixel thick in 3D, so point is swallowed')
@skip_on_win_ci
@skip_local_popups
def test_z_order_image_points_after_ndisplay(make_napari_viewer):
    """Test z order of image and points remanins constant after chaning ndisplay."""
    data = np.ones((11, 11))

    viewer = make_napari_viewer(show=True)
    viewer.add_image(data, colormap='red')
    viewer.add_points([5, 5], face_color='blue', size=5)
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    # Switch to 3D rendering
    viewer.dims.ndisplay = 3
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is still visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    # Switch back to 2D rendering
    viewer.dims.ndisplay = 2
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is still visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])


@skip_on_win_ci
@skip_local_popups
def test_changing_image_colormap(make_napari_viewer):
    """Test changing colormap changes rendering."""
    viewer = make_napari_viewer(show=True)

    data = np.ones((20, 20, 20))
    layer = viewer.add_image(data, contrast_limits=[0, 1])

    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    np.testing.assert_almost_equal(screenshot[center], [255, 255, 255, 255])

    layer.colormap = 'red'
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    np.testing.assert_almost_equal(screenshot[center], [255, 0, 0, 255])

    viewer.dims.ndisplay = 3
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    np.testing.assert_almost_equal(screenshot[center], [255, 0, 0, 255])

    layer.colormap = 'blue'
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    viewer.dims.ndisplay = 2
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])


@skip_on_win_ci
@skip_local_popups
def test_changing_image_gamma(make_napari_viewer):
    """Test changing gamma changes rendering."""
    viewer = make_napari_viewer(show=True)

    data = np.ones((20, 20, 20))
    layer = viewer.add_image(data, contrast_limits=[0, 2])

    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    assert 127 <= screenshot[(*center, 0)] <= 129

    layer.gamma = 0.1
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    assert screenshot[(*center, 0)] > 230

    viewer.dims.ndisplay = 3
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    assert screenshot[(*center, 0)] > 230

    layer.gamma = 1.9
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    assert screenshot[(*center, 0)] < 80

    viewer.dims.ndisplay = 2
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    assert screenshot[(*center, 0)] < 80


@skip_on_win_ci
@skip_local_popups
def test_grid_mode(make_napari_viewer):
    """Test changing gamma changes rendering."""
    viewer = make_napari_viewer(show=True)

    # Add images
    data = np.ones((6, 15, 15))
    viewer.add_image(data, channel_axis=0, blending='translucent')

    assert not viewer.grid.enabled
    assert viewer.grid.actual_shape(6) == (1, 1)
    assert viewer.grid.stride == 1
    translations = [layer._translate_grid for layer in viewer.layers]
    expected_translations = np.zeros((6, 2))
    np.testing.assert_allclose(translations, expected_translations)

    # check screenshot
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

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

    # check screenshot
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    # sample 6 squares of the grid and check they have right colors
    pos = [
        (1 / 3, 1 / 4),
        (1 / 3, 1 / 2),
        (1 / 3, 3 / 4),
        (2 / 3, 1 / 4),
        (2 / 3, 1 / 2),
        (2 / 3, 3 / 4),
    ]
    # BGRMYC color order
    color = [
        [0, 0, 255, 255],
        [0, 255, 0, 255],
        [255, 0, 0, 255],
        [255, 0, 255, 255],
        [255, 255, 0, 255],
        [0, 255, 255, 255],
    ]
    for c, p in zip(color, pos, strict=False):
        coord = tuple(
            np.round(np.multiply(screenshot.shape[:2], p)).astype(int)
        )
        np.testing.assert_almost_equal(screenshot[coord], c)

    # reorder layers, swapping 0 and 5
    viewer.layers.move(5, 0)
    viewer.layers.move(1, 6)

    # check screenshot
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    # CGRMYB color order
    color = [
        [0, 255, 255, 255],
        [0, 255, 0, 255],
        [255, 0, 0, 255],
        [255, 0, 255, 255],
        [255, 255, 0, 255],
        [0, 0, 255, 255],
    ]
    for c, p in zip(color, pos, strict=False):
        coord = tuple(
            np.round(np.multiply(screenshot.shape[:2], p)).astype(int)
        )
        np.testing.assert_almost_equal(screenshot[coord], c)

    # return to stack view
    viewer.grid.enabled = False
    assert not viewer.grid.enabled
    assert viewer.grid.actual_shape(6) == (1, 1)
    assert viewer.grid.stride == 1
    translations = [layer._translate_grid for layer in viewer.layers]
    expected_translations = np.zeros((6, 2))
    np.testing.assert_allclose(translations, expected_translations)

    # check screenshot
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    np.testing.assert_almost_equal(screenshot[center], [0, 255, 255, 255])


@skip_on_win_ci
@skip_local_popups
def test_changing_image_attenuation(make_napari_viewer):
    """Test changing attenuation value changes rendering."""
    data = np.zeros((100, 10, 10))
    data[0] = 1

    viewer = make_napari_viewer(show=True)
    viewer.dims.ndisplay = 3
    viewer.add_image(data, contrast_limits=[0, 1])

    # normal mip
    viewer.layers[0].rendering = 'mip'
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    mip_value = screenshot[center][0]

    # zero attenuation (still attenuated!)
    viewer.layers[0].rendering = 'attenuated_mip'
    viewer.layers[0].attenuation = 0.0
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    zero_att_value = screenshot[center][0]

    # increase attenuation
    viewer.layers[0].attenuation = 0.5
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    more_att_value = screenshot[center][0]
    # Check that rendering has been attenuated
    assert zero_att_value < more_att_value < mip_value


@skip_on_win_ci
@skip_local_popups
def test_labels_painting(make_napari_viewer):
    """Test painting labels updates image."""
    data = np.zeros((100, 100), dtype=np.int32)

    viewer = make_napari_viewer(show=True)
    viewer.add_labels(data)
    layer = viewer.layers[0]

    screenshot = viewer.screenshot(canvas_only=True, flash=False)

    # Check that no painting has occurred
    assert layer.data.max() == 0
    assert screenshot[:, :, :2].max() == 0

    # Enter paint mode
    viewer.cursor.position = (0, 0)
    layer.mode = 'paint'
    layer.selected_label = 3

    # Simulate click
    event = read_only_mouse_event(
        type='mouse_press',
        is_dragging=False,
        position=viewer.cursor.position,
    )
    mouse_press_callbacks(layer, event)

    viewer.cursor.position = (100, 100)

    # Simulate drag
    event = read_only_mouse_event(
        type='mouse_move',
        is_dragging=True,
        position=viewer.cursor.position,
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = read_only_mouse_event(
        type='mouse_release',
        is_dragging=False,
        position=viewer.cursor.position,
    )

    mouse_release_callbacks(layer, event)

    event = read_only_mouse_event(
        type='mouse_press',
        is_dragging=False,
        position=viewer.cursor.position,
    )
    mouse_press_callbacks(layer, event)

    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    # Check that painting has now occurred
    assert layer.data.max() > 0
    assert screenshot[:, :, :2].max() > 0


@pytest.mark.skip('Welcome visual temporarily disabled')
@skip_on_win_ci
@skip_local_popups
def test_welcome(make_napari_viewer):
    """Test that something visible on launch."""
    viewer = make_napari_viewer(show=True)

    # Check something is visible
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    assert len(viewer.layers) == 0
    assert screenshot[..., :-1].max() > 0

    # Check adding zeros image makes it go away
    viewer.add_image(np.zeros((1, 1)))
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    assert len(viewer.layers) == 1
    assert screenshot[..., :-1].max() == 0

    # Remove layer and check something is visible again
    viewer.layers.pop(0)
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    assert len(viewer.layers) == 0
    assert screenshot[..., :-1].max() > 0


@skip_on_win_ci
@skip_local_popups
def test_axes_visible(make_napari_viewer):
    """Test that something appears when axes become visible."""
    viewer = make_napari_viewer(show=True)
    viewer.window._qt_viewer.set_welcome_visible(False)

    # Check axes are not visible
    launch_screenshot = viewer.screenshot(canvas_only=True, flash=False)
    assert not viewer.axes.visible

    # Make axes visible and check something is seen
    viewer.axes.visible = True
    on_screenshot = viewer.screenshot(canvas_only=True, flash=False)
    assert viewer.axes.visible
    assert abs(on_screenshot - launch_screenshot).max() > 0

    # Make axes not visible and check they are gone
    viewer.axes.visible = False
    off_screenshot = viewer.screenshot(canvas_only=True, flash=False)
    assert not viewer.axes.visible
    np.testing.assert_almost_equal(launch_screenshot, off_screenshot)


@skip_on_win_ci
@skip_local_popups
def test_scale_bar_visible(make_napari_viewer):
    """Test that something appears when scale bar becomes visible."""
    viewer = make_napari_viewer(show=True)
    viewer.window._qt_viewer.set_welcome_visible(False)

    # Check scale bar is not visible
    launch_screenshot = viewer.screenshot(canvas_only=True, flash=False)
    assert not viewer.scale_bar.visible

    # Make scale bar visible and check something is seen
    viewer.scale_bar.visible = True
    on_screenshot = viewer.screenshot(canvas_only=True, flash=False)
    assert viewer.scale_bar.visible
    assert abs(on_screenshot - launch_screenshot).max() > 0

    # Make scale bar not visible and check it is gone
    viewer.scale_bar.visible = False
    off_screenshot = viewer.screenshot(canvas_only=True, flash=False)
    assert not viewer.scale_bar.visible
    np.testing.assert_almost_equal(launch_screenshot, off_screenshot)


@skip_on_win_ci
@skip_local_popups
def test_screenshot_has_no_border(make_napari_viewer):
    """See https://github.com/napari/napari/issues/3357"""
    viewer = make_napari_viewer(show=True)
    image_data = np.ones((60, 80))
    viewer.add_image(image_data, colormap='red')
    # Zoom in dramatically to make the screenshot all red.
    viewer.camera.zoom = 1000

    screenshot = viewer.screenshot(canvas_only=True, flash=False)

    expected = np.broadcast_to([255, 0, 0, 255], screenshot.shape)
    np.testing.assert_array_equal(screenshot, expected)


@skip_on_win_ci
@skip_local_popups
def test_blending_modes_with_canvas(make_napari_viewer):
    shape = (60, 80)
    viewer = make_napari_viewer(show=True)

    # add two images with different values
    img1 = np.full(shape, 20, np.uint8)
    img2 = np.full(shape, 50, np.uint8)
    img1_layer = viewer.add_image(img1)
    img2_layer = viewer.add_image(img2)

    viewer.window._qt_viewer.canvas.size = shape
    viewer.camera.zoom = 1

    # check that additive behaves correctly with black canvas
    img1_layer.blending = 'additive'
    img2_layer.blending = 'additive'
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    np.testing.assert_array_equal(screenshot[:, :, 0], img1 + img2)

    # minimum should not result in black background if canvas is black
    img1_layer.blending = 'minimum'
    img2_layer.blending = 'minimum'
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    np.testing.assert_array_equal(screenshot[:, :, 0], np.minimum(img1, img2))
    # toggle visibility of bottom layer
    img1_layer.visible = False
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    np.testing.assert_array_equal(screenshot[:, :, 0], img2)
    # and canvas should not affect the above results
    viewer.window._qt_viewer.canvas.bgcolor = 'white'

    # check that additive behaves correctly, despite white canvas
    img1_layer.visible = True
    img1_layer.blending = 'additive'
    img2_layer.blending = 'additive'
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    np.testing.assert_array_equal(screenshot[:, :, 0], img1 + img2)
    # toggle visibility of bottom layer
    img1_layer.visible = False
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    np.testing.assert_array_equal(screenshot[:, :, 0], img2)
    # minimum should always work with white canvas bgcolor
    img1_layer.visible = True
    img1_layer.blending = 'minimum'
    img2_layer.blending = 'minimum'
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    np.testing.assert_array_equal(screenshot[:, :, 0], np.minimum(img1, img2))


@skip_local_popups
def test_active_layer_highlight_visibility(qt_viewer):
    viewer = qt_viewer.viewer

    # take initial screenshot (full black/empty screenshot since welcome message is hidden)
    launch_screenshot = qt_viewer.screenshot(flash=False)
    # check screenshot ignoring alpha
    assert launch_screenshot[..., :-1].max() == 0

    # add shapes layer setting edge and face color to `black` (so shapes aren't
    # visible unless they're selected), create a rectangle and select the created shape
    shapes_layer: Shapes = viewer.add_shapes(
        edge_color='black', face_color='black'
    )
    shapes_layer.add_rectangles([[0, 0], [1, 1]])
    shapes_layer.selected_data = {0}

    # there should be a highlight so a screenshot should have something visible
    highlight_screenshot = qt_viewer.screenshot(flash=False)
    # check screenshot ignoring alpha
    assert highlight_screenshot[..., :-1].max() > 0

    # clear viewer layer selection
    viewer.layers.selection.clear()

    # there shouldn't be a highlight so a new screenshot shouldn't have something visible
    no_highlight_screenshot = qt_viewer.screenshot(flash=False)
    # check screenshot ignoring alpha
    assert no_highlight_screenshot[..., :-1].max() == 0

    # select again the layer with the rectangle shape
    viewer.layers.selection.add(shapes_layer)

    # there should be a highlight so a screenshot should have something visible
    reselection_highlight_screenshot = qt_viewer.screenshot(flash=False)
    # check screenshot ignoring alpha
    assert reselection_highlight_screenshot[..., :-1].max() > 0
