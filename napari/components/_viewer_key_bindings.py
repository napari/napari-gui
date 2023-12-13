from __future__ import annotations

from typing import TYPE_CHECKING, cast

from napari.components.viewer_model import ViewerModel
from napari.utils.action_manager import action_manager
from napari.utils.theme import available_themes, get_system_theme
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari._qt.dialogs.preferences_dialog import PreferencesDialog
    from napari.viewer import Viewer


def register_viewer_action(description):
    """
    Convenient decorator to register an action with the current ViewerModel

    It will use the function name as the action name. We force the description
    to be given instead of function docstring for translation purpose.
    """

    def _inner(func):
        action_manager.register_action(
            name=f'napari:{func.__name__}',
            command=func,
            description=description,
            keymapprovider=ViewerModel,
        )
        return func

    return _inner


@register_viewer_action(trans._("Reset scroll."))
def reset_scroll_progress(viewer: Viewer):
    # on key press
    viewer.dims._scroll_progress = 0
    yield

    # on key release
    viewer.dims._scroll_progress = 0


reset_scroll_progress.__doc__ = trans._("Reset dims scroll progress")


@register_viewer_action(trans._("Toggle 2D/3D view."))
def toggle_ndisplay(viewer: Viewer):
    if viewer.dims.ndisplay == 2:
        viewer.dims.ndisplay = 3
    else:
        viewer.dims.ndisplay = 2


# Making this an action makes vispy really unhappy during the tests
# on mac only with:
# ```
# RuntimeError: wrapped C/C++ object of type CanvasBackendDesktop has been deleted
# ```
@register_viewer_action(trans._("Toggle current viewer theme."))
def toggle_theme(viewer: ViewerModel):
    """Toggle theme for current viewer"""
    themes = available_themes()
    current_theme = viewer.theme
    # Check what the system theme is, to toggle properly
    if current_theme == 'system':
        current_theme = get_system_theme()
    idx = themes.index(current_theme)
    idx = (idx + 1) % len(themes)
    # Don't toggle to system, just among actual themes
    if themes[idx] == 'system':
        idx = (idx + 1) % len(themes)

    viewer.theme = themes[idx]


@register_viewer_action(trans._("Reset view to original state."))
def reset_view(viewer: Viewer):
    viewer.reset_view()


@register_viewer_action(trans._("Delete selected layers."))
def delete_selected_layers(viewer: Viewer):
    viewer.layers.remove_selected()


@register_viewer_action(trans._("Increment dimensions slider to the left."))
def increment_dims_left(viewer: Viewer):
    viewer.dims._increment_dims_left()


@register_viewer_action(trans._("Increment dimensions slider to the right."))
def increment_dims_right(viewer: Viewer):
    viewer.dims._increment_dims_right()


@register_viewer_action(trans._("Move focus of dimensions slider up."))
def focus_axes_up(viewer: Viewer):
    viewer.dims._focus_up()


@register_viewer_action(trans._("Move focus of dimensions slider down."))
def focus_axes_down(viewer: Viewer):
    viewer.dims._focus_down()


# Use non-breaking spaces and non-breaking hyphen for Preferences table
@register_viewer_action(
    trans._(
        "Change order of the visible axes, e.g.\u00A0[0,\u00A01,\u00A02]\u00A0\u2011>\u00A0[2,\u00A00,\u00A01]."
    ),
)
def roll_axes(viewer: Viewer):
    viewer.dims._roll()


# Use non-breaking spaces and non-breaking hyphen for Preferences table
@register_viewer_action(
    trans._(
        "Transpose order of the last two visible axes, e.g.\u00A0[0,\u00A01]\u00A0\u2011>\u00A0[1,\u00A00]."
    ),
)
def transpose_axes(viewer: Viewer):
    viewer.dims.transpose()


@register_viewer_action(trans._("Toggle grid mode."))
def toggle_grid(viewer: Viewer):
    viewer.grid.enabled = not viewer.grid.enabled


@register_viewer_action(trans._("Toggle visibility of selected layers"))
def toggle_selected_visibility(viewer: Viewer):
    viewer.layers.toggle_selected_visibility()


@register_viewer_action(
    trans._(
        "Show/Hide IPython console (only available when napari started as standalone application)"
    )
)
def toggle_console_visibility(viewer: Viewer):
    viewer.window._qt_viewer.toggle_console_visibility()


@register_viewer_action(trans._("Press and hold for pan/zoom mode"))
def hold_for_pan_zoom(viewer: ViewerModel):
    selected_layer = viewer.layers.selection.active
    if selected_layer is None:
        yield
        return
    previous_mode = selected_layer.mode
    # Each layer has its own Mode enum class with different values,
    # but they should all have a PAN_ZOOM value. At the time of writing
    # these enums do not share a base class or protocol, so ignore the
    # attribute check for now.
    pan_zoom = selected_layer._modeclass.PAN_ZOOM  # type: ignore[attr-defined]
    if previous_mode != pan_zoom:
        selected_layer.mode = pan_zoom
        yield

        selected_layer.mode = previous_mode


@register_viewer_action(trans._("Show all key bindings"))
def show_shortcuts(viewer: Viewer):
    viewer.window._open_preferences_dialog()
    pref_dialog = cast('PreferencesDialog', viewer.window._pref_dialog)
    pref_list = pref_dialog._list
    for i in range(pref_list.count()):
        if (item := pref_list.item(i)) and item.text() == "Shortcuts":
            pref_list.setCurrentRow(i)
