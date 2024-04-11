import os

import numpy as np
import pytest

from napari import Viewer, layers
from napari._tests.utils import (
    add_layer_by_type,
    check_view_transform_consistency,
    check_viewer_functioning,
    layer_test_data,
    skip_local_popups,
    skip_on_win_ci,
)
from napari.settings import get_settings
from napari.utils._tests.test_naming import eval_with_filename
from napari.utils.action_manager import action_manager


def _get_provider_actions(type_):
    actions = set()
    for superclass in type_.mro():
        actions.update(
            action.command
            for action in action_manager._get_provider_actions(
                superclass
            ).values()
        )
    return actions


def _assert_shortcuts_exist_for_each_action(type_):
    actions = _get_provider_actions(type_)
    shortcuts = {
        name.partition(':')[-1] for name in get_settings().shortcuts.shortcuts
    }
    shortcuts.update(func.__name__ for func in type_.class_keymap.values())
    for action in actions:
        assert (
            action.__name__ in shortcuts
        ), f"missing shortcut for action '{action.__name__}' on '{type_.__name__}' is missing"


viewer_actions = _get_provider_actions(Viewer)


def test_all_viewer_actions_are_accessible_via_shortcut(make_napari_viewer):
    """
    Make sure we do find all the actions attached to a viewer via keybindings
    """
    # instantiate to make sure everything is initialized correctly
    _ = make_napari_viewer()
    _assert_shortcuts_exist_for_each_action(Viewer)


@pytest.mark.xfail
def test_non_existing_bindings():
    """
    Those are condition tested in next unittest; but do not exists; this is
    likely due to an oversight somewhere.
    """
    assert 'play' in [func.__name__ for func in viewer_actions]
    assert 'toggle_fullscreen' in [func.__name__ for func in viewer_actions]


@pytest.mark.parametrize('func', viewer_actions)
def test_viewer_actions(make_napari_viewer, func):
    viewer = make_napari_viewer()

    if func.__name__ == 'toggle_fullscreen' and not os.getenv('CI'):
        pytest.skip('Fullscreen cannot be tested in CI')
    if func.__name__ == 'play':
        pytest.skip('Play cannot be tested with Pytest')
    func(viewer)


def test_viewer(make_napari_viewer):
    """Test instantiating viewer."""
    viewer = make_napari_viewer()
    view = viewer.window._qt_viewer

    assert viewer.title == 'napari'
    assert view.viewer == viewer

    assert len(viewer.layers) == 0
    assert view.layers.model().rowCount() == 0

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Switch to 3D rendering mode and back to 2D rendering mode
    viewer.dims.ndisplay = 3
    assert viewer.dims.ndisplay == 3
    viewer.dims.ndisplay = 2
    assert viewer.dims.ndisplay == 2


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
def test_add_layer(make_napari_viewer, layer_class, data, ndim):
    viewer = make_napari_viewer()
    layer = add_layer_by_type(viewer, layer_class, data, visible=True)
    check_viewer_functioning(viewer, viewer.window._qt_viewer, data, ndim)

    for func in layer.class_keymap.values():
        func(layer)


layer_types = (
    'Image',
    'Vectors',
    'Surface',
    'Tracks',
    'Points',
    'Labels',
    'Shapes',
)


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
def test_all_layer_actions_are_accessible_via_shortcut(
    layer_class, data, ndim
):
    """
    Make sure we do find all the actions attached to a layer via keybindings
    """
    # instantiate to make sure everything is initialized correctly
    _ = layer_class(data)
    _assert_shortcuts_exist_for_each_action(layer_class)


@pytest.mark.parametrize('layer_class, a_unique_name, ndim', layer_test_data)
def test_add_layer_magic_name(
    make_napari_viewer, layer_class, a_unique_name, ndim
):
    """Test magic_name works when using add_* for layers"""
    # Tests for issue #1709
    viewer = make_napari_viewer()  # noqa: F841
    layer = eval_with_filename(
        'add_layer_by_type(viewer, layer_class, a_unique_name)',
        'somefile.py',
    )
    assert layer.name == 'a_unique_name'


@skip_on_win_ci
def test_screenshot(make_napari_viewer, qtbot):
    """Test taking a screenshot."""
    viewer = make_napari_viewer()

    np.random.seed(0)
    # Add image
    data = np.random.random((10, 15))
    viewer.add_image(data)

    # Add labels
    data = np.random.randint(20, size=(10, 15))
    viewer.add_labels(data)

    # Add points
    data = 20 * np.random.random((10, 2))
    viewer.add_points(data)

    # Add vectors
    data = 20 * np.random.random((10, 2, 2))
    viewer.add_vectors(data)

    # Add shapes
    data = 20 * np.random.random((10, 4, 2))
    viewer.add_shapes(data)

    # Take screenshot of the image canvas only
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    assert screenshot.ndim == 3

    # Take screenshot with the viewer included
    screenshot = viewer.screenshot(canvas_only=False, flash=False)
    assert screenshot.ndim == 3

    # test size argument (and ensure it coerces to int)
    screenshot = viewer.screenshot(canvas_only=True, size=(20, 20.0))
    assert screenshot.shape == (20, 20, 4)
    # Here we wait until the flash animation will be over. We cannot wait on finished
    # signal as _flash_animation may be already removed when calling wait.
    qtbot.waitUntil(
        lambda: not hasattr(
            viewer.window._qt_viewer._welcome_widget, '_flash_animation'
        )
    )


@skip_on_win_ci
def test_changing_theme(make_napari_viewer):
    """Test changing the theme updates the full window."""
    viewer = make_napari_viewer(show=False)
    viewer.window._qt_viewer.set_welcome_visible(False)
    viewer.add_points(data=None)
    size = viewer.window._qt_viewer.size()
    viewer.window._qt_viewer.setFixedSize(size)

    assert viewer.theme == 'dark'
    screenshot_dark = viewer.screenshot(canvas_only=False, flash=False)

    viewer.theme = 'light'
    assert viewer.theme == 'light'
    screenshot_light = viewer.screenshot(canvas_only=False, flash=False)

    equal = (screenshot_dark == screenshot_light).min(-1)

    # more than 99.5% of the pixels have changed
    assert (np.count_nonzero(equal) / equal.size) < 0.05, 'Themes too similar'

    with pytest.raises(ValueError):
        viewer.theme = 'nonexistent_theme'


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
def test_roll_transpose_update(make_napari_viewer, layer_class, data, ndim):
    """Check that transpose and roll preserve correct transform sequence."""
    viewer = make_napari_viewer()

    np.random.seed(0)

    layer = add_layer_by_type(viewer, layer_class, data)

    # Set translations and scalings (match type of visual layer storing):
    transf_dict = {
        'translate': np.random.randint(0, 10, ndim).astype(np.float32),
        'scale': np.random.rand(ndim).astype(np.float32),
    }
    for k, val in transf_dict.items():
        setattr(layer, k, val)

    if layer_class in [layers.Image, layers.Labels]:
        transf_dict['translate'] -= transf_dict['scale'] / 2

    # Check consistency:
    check_view_transform_consistency(layer, viewer, transf_dict)

    # Roll dims and check again:
    viewer.dims.roll()
    check_view_transform_consistency(layer, viewer, transf_dict)

    # Transpose and check again:
    viewer.dims.transpose()
    check_view_transform_consistency(layer, viewer, transf_dict)


def test_toggling_axes(make_napari_viewer):
    """Test toggling axes."""
    viewer = make_napari_viewer()

    # Check axes are not visible
    assert not viewer.axes.visible

    # Make axes visible
    viewer.axes.visible = True
    assert viewer.axes.visible

    # Enter 3D rendering and check axes still visible
    viewer.dims.ndisplay = 3
    assert viewer.axes.visible

    # Make axes not visible
    viewer.axes.visible = False
    assert not viewer.axes.visible


@pytest.mark.parametrize('attribute', ['scale_bar', 'slice_text'])
def test_toggling_aux_overlay(make_napari_viewer, attribute):
    """Test toggling scale bar."""
    viewer = make_napari_viewer()

    aux_overlay = getattr(viewer, attribute)

    # Check scale bar is not visible
    assert not aux_overlay.visible

    # Make scale bar visible
    aux_overlay.visible = True
    assert aux_overlay.visible

    # Enter 3D rendering and check scale bar is still visible
    viewer.dims.ndisplay = 3
    assert aux_overlay.visible

    # Make scale bar not visible
    aux_overlay.visible = False
    assert not aux_overlay.visible


def test_removing_points_data(make_napari_viewer):
    viewer = make_napari_viewer()
    points = np.random.random((4, 2)) * 4

    pts_layer = viewer.add_points(points)
    pts_layer.data = np.zeros([0, 2])

    assert len(pts_layer.data) == 0


def test_deleting_points(make_napari_viewer):
    viewer = make_napari_viewer()
    points = np.random.random((4, 2)) * 4

    pts_layer = viewer.add_points(points)
    pts_layer.selected_data = {0}
    pts_layer.remove_selected()

    assert len(pts_layer.data) == 3


@skip_on_win_ci
@skip_local_popups
def test_custom_layer(make_napari_viewer):
    """Make sure that custom layers subclasses can be added to the viewer."""

    class NewLabels(layers.Labels):
        """'Empty' extension of napari Labels layer."""

    # Make a viewer and add the custom layer
    viewer = make_napari_viewer(show=True)
    viewer.add_layer(NewLabels(np.zeros((10, 10, 10), dtype=np.uint8)))


def test_emitting_data_doesnt_change_points_value(make_napari_viewer):
    """Test emitting data with no change doesn't change the layer _value."""
    viewer = make_napari_viewer()

    data = np.array([[0, 0], [10, 10], [20, 20]])
    layer = viewer.add_points(data, size=2)
    viewer.layers.selection.active = layer

    assert layer._value is None
    viewer.mouse_over_canvas = True
    viewer.cursor.position = tuple(layer.data[1])
    assert layer._value == 1

    layer.events.data(value=layer.data)
    assert layer._value == 1


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
def test_emitting_data_doesnt_change_cursor_position(
    make_napari_viewer, layer_class, data, ndim
):
    """Test emitting data event from layer doesn't change cursor position"""
    viewer = make_napari_viewer()
    layer = layer_class(data)
    viewer.add_layer(layer)

    new_position = (5,) * ndim
    viewer.cursor.position = new_position
    layer.events.data(value=layer.data)

    assert viewer.cursor.position == new_position


@skip_local_popups
@skip_on_win_ci
def test_empty_shapes_dims(make_napari_viewer):
    """make sure an empty shapes layer can render in 3D"""
    viewer = make_napari_viewer(show=True)
    viewer.add_shapes(None)
    viewer.dims.ndisplay = 3


def test_current_viewer(make_napari_viewer):
    """Test that the viewer made last is the "current_viewer()" until another is activated"""
    # Make two DIFFERENT viewers
    viewer1: Viewer = make_napari_viewer()
    viewer2: Viewer = make_napari_viewer()
    assert viewer2 is not viewer1
    # Ensure one is returned by napari.current_viewer()
    from napari import current_viewer

    assert current_viewer() is viewer2
    assert current_viewer() is not viewer1

    viewer1.window.activate()

    assert current_viewer() is viewer1
    assert current_viewer() is not viewer2


def test_reset_empty(make_napari_viewer):
    """
    Test that resetting an empty viewer doesn't crash
    https://github.com/napari/napari/issues/4867
    """
    viewer = make_napari_viewer()
    viewer.reset()


def test_reset_non_empty(make_napari_viewer):
    """
    Test that resetting a non-empty viewer doesn't crash
    https://github.com/napari/napari/issues/4867
    """
    viewer = make_napari_viewer()
    viewer.add_points([(0, 1), (2, 3)])
    viewer.reset()
