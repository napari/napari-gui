from app_model.types import Action, SubmenuItem

from napari._app_model.constants import MenuGroup, MenuId
from napari._app_model.context import LayerListSelectionContextKeys as LLSCK
from napari.layers import _layer_actions
from napari.utils.translations import trans

LAYERS_SUBMENUS = [
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_VISUALIZE,
            title=trans._('Visualize'),
            group=MenuGroup.NAVIGATION,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_ANNOTATE,
            title=trans._('Annotate'),
            group=MenuGroup.NAVIGATION,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_DATA,
            title=trans._('Data'),
            group=MenuGroup.LAYERS.CONVERT,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_LAYER_TYPE,
            title=trans._('Layer Type'),
            group=MenuGroup.LAYERS.CONVERT,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_TRANSFORM,
            title=trans._('Transform'),
            group=MenuGroup.LAYERS.GEOMETRY,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_MEASURE,
            title=trans._('Measure'),
            group=MenuGroup.LAYERS.GEOMETRY,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_FILTER,
            title=trans._('Filter'),
            group=MenuGroup.LAYERS.GENERATE,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_REGISTER,
            title=trans._('Register'),
            group=MenuGroup.LAYERS.GENERATE,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_PROJECT,
            title=trans._('Project'),
            group=MenuGroup.LAYERS.GENERATE,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_SEGMENT,
            title=trans._('Segment'),
            group=MenuGroup.LAYERS.GENERATE,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_TRACK,
            title=trans._('Track'),
            group=MenuGroup.LAYERS.GENERATE,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_CLASSIFY,
            title=trans._('Classify'),
            group=MenuGroup.LAYERS.GENERATE,
        ),
    ),
]

# placeholder, add actions here!
LAYERS_ACTIONS: list[Action] = [
    Action(
        id='napari.layers.measure.shape',
        title='Enable shapes measuring',
        callback=_layer_actions._enable_measure_shapes,
        menus=[MenuId.MENUBAR_LAYERS.LAYERS_MEASURE, MenuId.LAYERLIST_CONTEXT],
        enablement=LLSCK.all_selected_layers_shapes,
    ),
]
