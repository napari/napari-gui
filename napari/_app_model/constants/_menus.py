"""All Menus that are available anywhere in the napari GUI are defined here.

These might be menubar menus, context menus, or other menus.  They could
even be "toolbars", such as the set of mode buttons on the layer list.
A "menu" needn't just be a literal QMenu (though it usually is): it is better
thought of as a set of related commands.

Internally, prefer using the `MenuId` enum instead of the string literal.

SOME of these (but definitely not all) will be exposed as "contributable"
menus for plugins to contribute commands and submenu items to.
"""

from napari.utils.compat import StrEnum


class MenuId(StrEnum):
    """Id representing a menu somewhere in napari."""

    MENUBAR_FILE = 'napari/file'
    FILE_OPEN_WITH_PLUGIN = 'napari/file/open_with_plugin'
    FILE_SAMPLES = 'napari/file/samples'

    MENUBAR_VIEW = 'napari/view'
    VIEW_AXES = 'napari/view/axes'
    VIEW_SCALEBAR = 'napari/view/scalebar'
    VIEW_SLICE_TEXT = 'napari/view/slice_text'

    MENUBAR_PLUGINS = 'napari/plugins'

    MENUBAR_HELP = 'napari/help'

    LAYERLIST_CONTEXT = 'napari/layers/context'
    LAYERS_CONVERT_DTYPE = 'napari/layers/convert_dtype'
    LAYERS_PROJECT = 'napari/layers/project'

    def __str__(self) -> str:
        return self.value


# XXX: the structure/usage pattern of this class may change in the future
class MenuGroup:
    NAVIGATION = 'navigation'  # always the first group in any menu
    RENDER = '1_render'
    # Plugins menubar
    PLUGINS = '1_plugins'
    PLUGIN_MULTI_SUBMENU = '2_plugin_multi_submenu'
    PLUGIN_SINGLE_CONTRIBUTIONS = '3_plugin_contributions'
    # File menubar
    PREFERENCES = '2_preferences'
    SAVE = '3_save'
    CLOSE = '4_close'

    class LAYERLIST_CONTEXT:
        CONVERSION = '1_conversion'
        SPLIT_MERGE = '5_split_merge'
        LINK = '9_link'


# TODO: add these to docs, with a lookup for what each menu is/does.
_CONTRIBUTABLES = {MenuId.LAYERLIST_CONTEXT.value}
"""Set of all menu ids that can be contributed to by plugins."""


def is_menu_contributable(menu_id: str) -> bool:
    """Return True if the given menu_id is a menu that plugins can contribute to."""
    return (
        menu_id in _CONTRIBUTABLES if menu_id.startswith('napari/') else True
    )
