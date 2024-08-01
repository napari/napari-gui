"""Qt Viewer Actions."""

from logging import getLogger
from typing import TYPE_CHECKING, Union

from app_model.types import Action, ToggleRule

from napari._app_model.constants import MenuId
from napari.components import _viewer_key_bindings
from napari.utils.misc import in_ipython, in_jupyter, in_python_repl
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.viewer import Viewer, ViewerModel


logger = getLogger(__name__)


def _get_viewer_ndisplay_status(
    viewer: Union['ViewerModel', 'Viewer'],
) -> bool:
    return viewer.dims.ndisplay == 3


def _get_viewer_grid_status(viewer: Union['ViewerModel', 'Viewer']) -> bool:
    return viewer.grid.enabled


# TODO: Add keybindings
Q_VIEWER_ACTIONS: list[Action] = [
    Action(
        id='napari.viewer.toggle_console_visibility',
        title=trans._(''),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=_viewer_key_bindings.toggle_console_visibility,
        tooltip=trans._(
            'Show/Hide IPython console (only available when napari started as standalone application)'
        ),
        enablement=not (in_ipython() or in_jupyter() or in_python_repl()),
    ),
    Action(
        id='napari.viewer.toggle_ndisplay',
        title=trans._(''),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=_viewer_key_bindings.toggle_ndisplay,
        tooltip=trans._('Toggle 2D/3D view.'),
        # TODO: Need of viewer ctx to write condition?
        toggled=ToggleRule(get_current=_get_viewer_ndisplay_status),
    ),
    Action(
        id='napari.viewer.roll_axes',
        title=trans._(''),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=_viewer_key_bindings.roll_axes,
        tooltip=trans._(
            'Change order of the visible axes, e.g.\u00a0[0,\u00a01,\u00a02]\u00a0\u2011>\u00a0[2,\u00a00,\u00a01].'
        ),
    ),
    Action(
        id='napari.viewer.transpose_axes',
        title=trans._(''),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=_viewer_key_bindings.transpose_axes,
        tooltip=trans._(
            'Transpose order of the last two visible axes, e.g.\u00a0[0,\u00a01]\u00a0\u2011>\u00a0[1,\u00a00].'
        ),
    ),
    Action(
        id='napari.viewer.toggle_grid',
        title=trans._(''),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=_viewer_key_bindings.toggle_grid,
        tooltip=trans._('Toggle grid mode.'),
        # TODO: Need of viewer ctx to write condition?
        toggled=ToggleRule(get_current=_get_viewer_grid_status),
    ),
    Action(
        id='napari.viewer.reset_view',
        title=trans._(''),
        menus=[
            {
                'id': MenuId.VIEWER_CONTROLS,
            }
        ],
        callback=_viewer_key_bindings.reset_view,
        tooltip=trans._('Reset view to original state.'),
    ),
]
