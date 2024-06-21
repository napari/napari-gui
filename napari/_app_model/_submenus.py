from app_model.types import SubmenuItem

from napari._app_model.constants import MenuGroup, MenuId
from napari._app_model.context import LayerListSelectionContextKeys as LLSCK
from napari.utils.translations import trans

SUBMENUS = [
    (
        MenuId.LAYERLIST_CONTEXT,
        SubmenuItem(
            submenu=MenuId.LAYERS_CONTEXT_CONVERT_DTYPE,
            title=trans._('Convert data type'),
            group=MenuGroup.LAYERLIST_CONTEXT.CONVERSION,
            order=None,
            enablement=LLSCK.all_selected_layers_labels,
        ),
    ),
    (
        MenuId.LAYERLIST_CONTEXT,
        SubmenuItem(
            submenu=MenuId.LAYERS_CONTEXT_PROJECT,
            title=trans._('Projections'),
            group=MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
            order=None,
            enablement=LLSCK.active_layer_is_image_3d,
        ),
    ),
    (
        MenuId.LAYERLIST_CONTEXT,
        SubmenuItem(
            submenu=MenuId.LAYERS_CONTEXT_COPY_SPATIAL,
            title=trans._('Copy scale and transforms'),
            group=MenuGroup.LAYERLIST_CONTEXT.COPY_SPATIAL,
            order=None,
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_NEW_LAYER,
            title=trans._('New Layer'),
            group=MenuGroup.NAVIGATION,
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_OPEN_WITH_PLUGIN,
            title=trans._('Open with Plugin'),
            group=MenuGroup.NAVIGATION,
            order=99,
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_SAMPLES,
            title=trans._('Open Sample'),
            group=MenuGroup.NAVIGATION,
            order=100,
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_IO_UTILITIES,
            title=trans._('IO Utilities'),
            group=MenuGroup.NAVIGATION,
            order=101,
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_ACQUIRE,
            title=trans._('Acquire'),
            group=MenuGroup.NAVIGATION,
            order=101,
        ),
    ),
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_AXES, title=trans._('Axes')),
    ),
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_SCALEBAR, title=trans._('Scale Bar')),
    ),
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
            submenu=MenuId.LAYERS_TRANSFORM,
            title=trans._('Transform'),
            group=MenuGroup.LAYERS.GEOMETRY,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_FILTER,
            title=trans._('Filter'),
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
