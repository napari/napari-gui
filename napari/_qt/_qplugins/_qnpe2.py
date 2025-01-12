"""Plugin related functions that require Qt.

Non-Qt plugin functions can be found in: `napari/plugins/_npe2.py`
"""

from __future__ import annotations

import inspect
from functools import partial
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from app_model import Action
from app_model.types import SubmenuItem, ToggleRule
from magicgui.type_map._magicgui import MagicFactory
from magicgui.widgets import FunctionGui, Widget
from npe2 import plugin_manager as pm
from qtpy.QtWidgets import QWidget

from napari._app_model import get_app_model
from napari._app_model.constants import MenuGroup, MenuId
from napari._qt._qapp_model.injection._qproviders import (
    _provide_viewer_or_raise,
    _provide_window,
    _provide_window_or_raise,
)
from napari.errors.reader_errors import MultipleReaderError
from napari.plugins import menu_item_template, plugin_manager
from napari.plugins._npe2 import _when_group_order, get_widget_contribution
from napari.utils.events import Event
from napari.utils.translations import trans
from napari.viewer import Viewer

if TYPE_CHECKING:
    from npe2.manifest import PluginManifest
    from npe2.plugin_manager import PluginName
    from npe2.types import WidgetCreator

    from napari.qt import QtViewer


# TODO: This is a separate function from `_build_samples_submenu_actions` so it
# can be easily deleted once npe1 is no longer supported.
def _rebuild_npe1_samples_menu() -> None:  # pragma: no cover
    """Register submenu and actions for all npe1 plugins, clearing all first."""
    app = get_app_model()
    # Unregister all existing npe1 sample menu actions and submenus
    if unreg := plugin_manager._unreg_sample_submenus:
        unreg()
    if unreg := plugin_manager._unreg_sample_actions:
        unreg()

    sample_actions: list[Action] = []
    sample_submenus: list[Any] = []
    for plugin_name, samples in plugin_manager._sample_data.items():
        multiprovider = len(samples) > 1
        if multiprovider:
            submenu_id = f'napari/file/samples/{plugin_name}'
            submenu = (
                MenuId.FILE_SAMPLES,
                SubmenuItem(submenu=submenu_id, title=trans._(plugin_name)),
            )
            sample_submenus.append(submenu)
        else:
            submenu_id = MenuId.FILE_SAMPLES

        for sample_name, sample_dict in samples.items():
            _add_sample_partial = partial(
                _add_sample,
                plugin=plugin_name,
                sample=sample_name,
            )

            display_name = sample_dict['display_name'].replace('&', '&&')
            if multiprovider:
                title = display_name
            else:
                title = menu_item_template.format(plugin_name, display_name)

            action: Action = Action(
                id=f'{plugin_name}:{display_name}',
                title=title,
                menus=[{'id': submenu_id, 'group': MenuGroup.NAVIGATION}],
                callback=_add_sample_partial,
            )
            sample_actions.append(action)

    if sample_submenus:
        unreg_sample_submenus = app.menus.append_menu_items(sample_submenus)
        plugin_manager._unreg_sample_submenus = unreg_sample_submenus
    if sample_actions:
        unreg_sample_actions = app.register_actions(sample_actions)
        plugin_manager._unreg_sample_actions = unreg_sample_actions


# TODO: This should be deleted once npe1 is no longer supported.
def _toggle_or_get_widget_npe1(
    plugin: str,
    widget_name: str,
    name: str,
    hook_type: str,
) -> None:  # pragma: no cover
    """Toggle if widget already built otherwise return widget for npe1."""
    window = _provide_window_or_raise(
        msg='Note that widgets cannot be opened in headless mode.'
    )

    if window and (dock_widget := window._dock_widgets.get(name)):
        dock_widget.setVisible(not dock_widget.isVisible())
        return

    if hook_type == 'dock':
        window.add_plugin_dock_widget(plugin, widget_name)
    else:
        window._add_plugin_function_widget(plugin, widget_name)


def _rebuild_npe1_plugins_menu() -> None:
    """Register widget submenu and actions for all npe1 plugins, clearing all first."""
    app = get_app_model()

    # Unregister all existing npe1 plugin menu actions and submenus
    if unreg := plugin_manager._unreg_plugin_submenus:
        unreg()
    if unreg := plugin_manager._unreg_plugin_actions:
        unreg()

    widget_actions: list[Action] = []
    widget_submenus: list[Any] = []
    for hook_type, (plugin_name, widgets) in chain(
        plugin_manager.iter_widgets()
    ):
        multiprovider = len(widgets) > 1
        if multiprovider:
            submenu_id = f'napari/plugins/{plugin_name}'
            submenu = (
                MenuId.MENUBAR_PLUGINS,
                SubmenuItem(
                    submenu=submenu_id,
                    title=trans._(plugin_name),
                    group=MenuGroup.PLUGIN_MULTI_SUBMENU,
                ),
            )
            widget_submenus.append(submenu)
        else:
            submenu_id = MenuId.MENUBAR_PLUGINS

        for widget_name in widgets:
            full_name = menu_item_template.format(plugin_name, widget_name)
            title = widget_name if multiprovider else full_name

            _widget_callback = partial(
                _toggle_or_get_widget_npe1,
                plugin=plugin_name,
                widget_name=widget_name,
                name=full_name,
                hook_type=hook_type,
            )
            _get_current_dock_status_partial = partial(
                _get_current_dock_status,
                full_name=full_name,
            )
            action: Action = Action(
                id=f'{plugin_name}:{widget_name.replace("&", "&&")}',
                title=title.replace('&', '&&'),
                menus=[
                    {
                        'id': submenu_id,
                        'group': MenuGroup.PLUGIN_SINGLE_CONTRIBUTIONS,
                    }
                ],
                callback=_widget_callback,
                toggled=ToggleRule(
                    get_current=_get_current_dock_status_partial
                ),
            )
            widget_actions.append(action)

    if widget_submenus:
        unreg_plugin_submenus = app.menus.append_menu_items(widget_submenus)
        plugin_manager._unreg_plugin_submenus = unreg_plugin_submenus
    if widget_actions:
        unreg_plugin_actions = app.register_actions(widget_actions)
        plugin_manager._unreg_plugin_actions = unreg_plugin_actions


def _get_contrib_parent_menu(
    multiprovider: bool,
    parent_menu: MenuId,
    mf: PluginManifest,
    group: str | None = None,
) -> tuple[str, list[tuple[str, SubmenuItem]]]:
    """Get parent menu of plugin contribution (samples/widgets).

    If plugin provides multiple contributions, create a new submenu item.
    """
    submenu: list[tuple[str, SubmenuItem]] = []
    if multiprovider:
        submenu_id = f'{parent_menu}/{mf.name}'
        submenu = [
            (
                parent_menu,
                SubmenuItem(
                    submenu=submenu_id,
                    title=trans._(mf.display_name),
                    group=group,
                ),
            ),
        ]
    else:
        submenu_id = parent_menu
    return submenu_id, submenu


# Note `QtViewer` gets added to `injection_store.namespace` during
# `init_qactions` so does not need to be imported for type annotation resolution
def _add_sample(qt_viewer: QtViewer, plugin: str, sample: str) -> None:
    from napari._qt.dialogs.qt_reader_dialog import handle_gui_reading

    try:
        qt_viewer.viewer.open_sample(plugin, sample)
    except MultipleReaderError as e:
        handle_gui_reading(
            [str(p) for p in e.paths],
            qt_viewer,
            stack=False,
        )


def _build_samples_submenu_actions(
    mf: PluginManifest,
) -> tuple[list[tuple[str, SubmenuItem]], list[Action]]:
    """Build sample data submenu and actions for a single npe2 plugin manifest."""
    from napari._app_model.constants import MenuGroup, MenuId
    from napari.plugins import menu_item_template

    # If no sample data, return
    if not mf.contributions.sample_data:
        return [], []

    sample_data = mf.contributions.sample_data
    multiprovider = len(sample_data) > 1
    submenu_id, submenu = _get_contrib_parent_menu(
        multiprovider,
        MenuId.FILE_SAMPLES,
        mf,
    )

    sample_actions: list[Action] = []
    for sample in sample_data:
        _add_sample_partial = partial(
            _add_sample,
            plugin=mf.name,
            sample=sample.key,
        )

        if multiprovider:
            title = sample.display_name
        else:
            title = menu_item_template.format(
                mf.display_name, sample.display_name
            )
        # To display '&' instead of creating a shortcut
        title = title.replace('&', '&&')

        action: Action = Action(
            id=f'{mf.name}:{sample.key}',
            title=title,
            menus=[{'id': submenu_id, 'group': MenuGroup.NAVIGATION}],
            callback=_add_sample_partial,
        )
        sample_actions.append(action)
    return submenu, sample_actions


def _get_widget_viewer_param(
    widget_callable: WidgetCreator, widget_name: str
) -> str:
    """Get widget parameter name for `viewer` (if any) and check type."""
    if inspect.isclass(widget_callable) and issubclass(
        widget_callable,
        (QWidget, Widget),
    ):
        widget_param = ''
        try:
            sig = inspect.signature(widget_callable.__init__)
        except ValueError:  # pragma: no cover
            # Inspection can fail when adding to bundled viewer as it thinks widget is
            # a builtin
            pass
        else:
            for param in sig.parameters.values():
                if param.name == 'napari_viewer' or param.annotation in (
                    'napari.viewer.Viewer',
                    Viewer,
                ):
                    widget_param = param.name
                    break

    # For magicgui type widget contributions, `Viewer` injection is done by
    # `magicgui.register_type`.
    elif isinstance(widget_callable, MagicFactory) or inspect.isfunction(
        widget_callable
    ):
        widget_param = ''
    else:
        raise TypeError(
            trans._(
                "'{widget}' must be `QtWidgets.QWidget` or `magicgui.widgets.Widget` subclass, `MagicFactory` instance or function. Please raise an issue in napari GitHub with the plugin and widget you were trying to use.",
                deferred=True,
                widget=widget_name,
            )
        )
    return widget_param


def _toggle_or_get_widget(
    plugin: str,
    widget_name: str,
    full_name: str,
) -> tuple[FunctionGui | QWidget | Widget, str] | None:
    """Toggle if widget already built otherwise return widget.

    Returned widget will be added to main window by a processor.
    Note for magicgui type widget contributions, `Viewer` injection is done by
    `magicgui.register_type` instead of a provider via annnotation.
    """
    viewer = _provide_viewer_or_raise(
        msg='Note that widgets cannot be opened in headless mode.',
    )

    window = viewer.window
    if window and (dock_widget := window._dock_widgets.get(full_name)):
        dock_widget.setVisible(not dock_widget.isVisible())
        return None

    # Get widget param name (if any) and check type
    widget_callable, _ = get_widget_contribution(plugin, widget_name)  # type: ignore [misc]
    widget_param = _get_widget_viewer_param(widget_callable, widget_name)

    kwargs = {}
    if widget_param:
        kwargs[widget_param] = viewer
    return widget_callable(**kwargs), full_name


def _get_current_dock_status(full_name: str) -> bool:
    window = _provide_window_or_raise(
        msg='Note that widgets cannot be opened in headless mode.',
    )
    if full_name in window._dock_widgets:
        return window._dock_widgets[full_name].isVisible()
    return False


def _build_widgets_submenu_actions(
    mf: PluginManifest,
) -> tuple[list[tuple[str, SubmenuItem]], list[Action]]:
    """Build widget submenu and actions for a single npe2 plugin manifest."""
    # If no widgets, return
    if not mf.contributions.widgets:
        return [], []

    # if this plugin declares any menu items, its actions should have the
    # plugin name.
    # TODO: update once plugin has self menus - they shouldn't exclude it
    # from the shorter name
    declares_menu_items = any(
        len(pm.instance()._command_menu_map[mf.name][command.id])
        for command in mf.contributions.commands or []
    )
    widgets = mf.contributions.widgets
    multiprovider = len(widgets) > 1
    default_submenu_id, default_submenu = _get_contrib_parent_menu(
        multiprovider,
        MenuId.MENUBAR_PLUGINS,
        mf,
        MenuGroup.PLUGIN_MULTI_SUBMENU,
    )
    needs_full_title = declares_menu_items or not multiprovider

    widget_actions: list[Action] = []
    for widget in widgets:
        full_name = menu_item_template.format(
            mf.display_name,
            widget.display_name,
        )

        _widget_callback = partial(
            _toggle_or_get_widget,
            plugin=mf.name,
            widget_name=widget.display_name,
            full_name=full_name,
        )
        _get_current_dock_status_partial = partial(
            _get_current_dock_status,
            full_name=full_name,
        )

        action_menus = [
            dict({'id': menu_key}, **_when_group_order(menu_item))
            for menu_key, menu_items in pm.instance()
            ._command_menu_map[mf.name][widget.command]
            .items()
            for menu_item in menu_items
        ] + [
            {
                'id': default_submenu_id,
                'group': MenuGroup.PLUGIN_SINGLE_CONTRIBUTIONS,
            }
        ]
        title = full_name if needs_full_title else widget.display_name
        # To display '&' instead of creating a shortcut
        title = title.replace('&', '&&')

        widget_actions.append(
            Action(
                id=f'{mf.name}:{widget.display_name}',
                title=title,
                callback=_widget_callback,
                menus=action_menus,
                toggled=ToggleRule(
                    get_current=_get_current_dock_status_partial
                ),
            )
        )
    return default_submenu, widget_actions


def _register_qt_actions(mf: PluginManifest) -> None:
    """Register samples and widget actions and submenus from a manifest.

    This is called when a plugin is registered or enabled and it adds the
    plugin's sample and widget actions and submenus to the app model registry.
    """
    app = get_app_model()
    samples_submenu, sample_actions = _build_samples_submenu_actions(mf)
    widgets_submenu, widget_actions = _build_widgets_submenu_actions(mf)

    context = pm.get_context(cast('PluginName', mf.name))
    actions = sample_actions + widget_actions
    if actions:
        context.register_disposable(app.register_actions(actions))
    submenus = samples_submenu + widgets_submenu
    if submenus:
        context.register_disposable(app.menus.append_menu_items(submenus))

    # Register dispose functions to remove plugin widgets from widget dictionary
    # `window._dock_widgets`
    if window := _provide_window():
        for widget in mf.contributions.widgets or ():
            widget_event = Event(type_name='', value=widget.display_name)

            def _remove_widget(event: Event = widget_event) -> None:
                window._remove_dock_widget(event)

            context.register_disposable(_remove_widget)
