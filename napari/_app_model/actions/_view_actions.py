"""Actions related to the 'View' menu that do not require Qt.

View actions that do require Qt should go in
`napari/_qt/_qapp_model/qactions/_view.py`.
"""

from app_model.types import Action, ToggleRule

from napari._app_model.actions._toggle_action import ViewerToggleAction
from napari._app_model.constants import CommandId, MenuGroup, MenuId
from napari.settings import get_settings

VIEW_ACTIONS: list[Action] = []
MENUID_DICT = {
    'axes': MenuId.VIEW_AXES,
    'scale_bar': MenuId.VIEW_SCALEBAR,
    'slice_text': MenuId.VIEW_SLICE_TEXT,
}


for cmd, viewer_attr, sub_attr in (
    (CommandId.TOGGLE_VIEWER_AXES, 'axes', 'visible'),
    (CommandId.TOGGLE_VIEWER_AXES_COLORED, 'axes', 'colored'),
    (CommandId.TOGGLE_VIEWER_AXES_LABELS, 'axes', 'labels'),
    (CommandId.TOGGLE_VIEWER_AXES_DASHED, 'axes', 'dashed'),
    (CommandId.TOGGLE_VIEWER_AXES_ARROWS, 'axes', 'arrows'),
    (CommandId.TOGGLE_VIEWER_SCALE_BAR, 'scale_bar', 'visible'),
    (CommandId.TOGGLE_VIEWER_SCALE_BAR_COLORED, 'scale_bar', 'colored'),
    (CommandId.TOGGLE_VIEWER_SCALE_BAR_TICKS, 'scale_bar', 'ticks'),
    (CommandId.TOGGLE_VIEWER_SLICE_TEXT, 'slice_text', 'visible'),
    (CommandId.TOGGLE_VIEWER_SLICE_TEXT_COLORED, 'slice_text', 'colored'),
):
    VIEW_ACTIONS.append(
        ViewerToggleAction(
            id=cmd,
            title=cmd.command_title,
            viewer_attribute=viewer_attr,
            sub_attribute=sub_attr,
            menus=[{'id': MENUID_DICT[viewer_attr]}],
        )
    )


def _tooltip_visibility_toggle() -> None:
    settings = get_settings().appearance
    settings.layer_tooltip_visibility = not settings.layer_tooltip_visibility


def _get_current_tooltip_visibility() -> bool:
    return get_settings().appearance.layer_tooltip_visibility


VIEW_ACTIONS.extend(
    [
        # TODO: this could be made into a toggle setting Action subclass
        # using a similar pattern to the above ViewerToggleAction classes
        Action(
            id=CommandId.TOGGLE_LAYER_TOOLTIPS,
            title=CommandId.TOGGLE_LAYER_TOOLTIPS.command_title,
            menus=[
                {
                    'id': MenuId.MENUBAR_VIEW,
                    'group': MenuGroup.RENDER,
                    'order': 10,
                }
            ],
            callback=_tooltip_visibility_toggle,
            toggled=ToggleRule(get_current=_get_current_tooltip_visibility),
        ),
    ]
)
