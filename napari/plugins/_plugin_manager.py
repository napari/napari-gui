import contextlib
import warnings
from collections.abc import Callable, Iterable, Iterator
from functools import partial
from pathlib import Path
from types import FunctionType
from typing import (
    Any,
)
from warnings import warn

from napari_plugin_engine import (
    HookImplementation,
    PluginManager as PluginManager,
)
from napari_plugin_engine.hooks import HookCaller
from typing_extensions import TypedDict

from napari._pydantic_compat import ValidationError
from napari.plugins import hook_specifications
from napari.settings import get_settings
from napari.types import AugmentedWidget, LayerData, SampleDict, WidgetCallable
from napari.utils.events import EmitterGroup, EventedSet
from napari.utils.misc import camel_to_spaces
from napari.utils.theme import Theme, register_theme, unregister_theme
from napari.utils.translations import trans


class PluginHookOption(TypedDict):
    """Custom type specifying plugin and enabled state."""

    plugin: str
    enabled: bool


CallOrderDict = dict[str, list[PluginHookOption]]


class NapariPluginManager(PluginManager):
    """PluginManager subclass for napari-specific functionality.

    Notes
    -----

    The events emitted by the plugin include:

    * registered (value: str)
        Emitted after plugin named `value` has been registered.
    * unregistered (value: str)
        Emitted after plugin named `value` has been unregistered.
    * enabled (value: str)
        Emitted after plugin named `value` has been removed from the block list.
    * disabled (value: str)
        Emitted after plugin named `value` has been added to the block list.
    """

    ENTRY_POINT = 'napari.plugin'

    def __init__(self) -> None:
        super().__init__('napari', discover_entry_point=self.ENTRY_POINT)

        self.events = EmitterGroup(
            source=self,
            registered=None,
            unregistered=None,
            enabled=None,
            disabled=None,
        )
        self._blocked: EventedSet[str] = EventedSet()
        self._blocked.events.changed.connect(self._on_blocked_change)

        # set of package names to skip when discovering, used for skipping
        # npe2 stuff
        self._skip_packages: set[str] = set()

        with self.discovery_blocked():
            self.add_hookspecs(hook_specifications)

        # dicts to store maps from extension -> plugin_name
        self._extension2reader: dict[str, str] = {}
        self._extension2writer: dict[str, str] = {}

        self._sample_data: dict[str, dict[str, SampleDict]] = {}
        self._dock_widgets: dict[
            str, dict[str, tuple[WidgetCallable, dict[str, Any]]]
        ] = {}
        self._function_widgets: dict[str, dict[str, Callable[..., Any]]] = {}
        self._theme_data: dict[str, dict[str, Theme]] = {}

        # TODO: remove once npe1 deprecated
        # appmodel sample menu actions/submenu unregister functions used in
        # `_rebuild_npe1_samples_menu`
        self._unreg_sample_submenus = None
        self._unreg_sample_actions = None
        # appmodel plugins menu actions/submenu unregister functions used in
        # `_rebuild_npe1_plugins_menu`
        self._unreg_plugin_submenus = None
        self._unreg_plugin_actions = None

    def _initialize(self) -> None:
        with self.discovery_blocked():
            from napari.settings import get_settings

            # dicts to store maps from extension -> plugin_name
            plugin_settings = get_settings().plugins
            self._extension2reader.update(plugin_settings.extension2reader)
            self._extension2writer.update(plugin_settings.extension2writer)

    def register(self, namespace: Any, name: str | None = None) -> str | None:
        name = super().register(namespace, name=name)
        if name:
            self.events.registered(value=name)
        return name

    def iter_available(
        self,
        path: str | None = None,
        entry_point: str | None = None,
        prefix: str | None = None,
    ) -> Iterator[tuple[str, str, str | None]]:
        # overriding to skip npe2 plugins
        for item in super().iter_available(path, entry_point, prefix):
            if item[-1] not in self._skip_packages:
                yield item

    def unregister(
        self,
        name_or_object: Any,
    ) -> Any | None:
        if isinstance(name_or_object, str):
            _name = name_or_object
        else:
            _name = self.get_name(name_or_object)

        plugin = super().unregister(name_or_object)

        # unregister any theme that was associated with the
        # unregistered plugin
        self.unregister_theme_colors(_name)

        # remove widgets, sample data, theme data
        for _dict in (
            self._dock_widgets,
            self._sample_data,
            self._theme_data,
            self._function_widgets,
        ):
            _dict.pop(_name, None)

        self.events.unregistered(value=_name)

        return plugin

    def _on_blocked_change(self, event) -> None:
        # things that are "added to the blocked list" become disabled
        for item in event.added:
            self.events.disabled(value=item)

        # things that are "removed from the blocked list" become enabled
        for item in event.removed:
            self.events.enabled(value=item)

        if event.removed:
            # if an event was removed from the "disabled" list...
            # let's reregister.  # TODO: might be able to be more direct here.
            self.discover()

        get_settings().plugins.disabled_plugins = set(self._blocked)

    def call_order(self, first_result_only=True) -> CallOrderDict:
        """Returns the call order from the plugin manager.

        Returns
        -------
        call_order : CallOrderDict
            mapping of hook_specification name, to a list of dicts with keys:
            {'plugin', 'hook_impl', 'enabled'}.  Plugins earlier in the dict are called
            sooner.
        """

        order: CallOrderDict = {}
        for spec_name, caller in self.hooks.items():
            # no need to save call order unless we only use first result
            if first_result_only and not caller.is_firstresult:
                continue
            impls = caller.get_hookimpls()
            # no need to save call order if there is only a single item
            if len(impls) > 1:
                order[spec_name] = [
                    {
                        'plugin': f'{impl.plugin_name}--{impl.function.__name__}',
                        'enabled': impl.enabled,
                    }
                    for impl in reversed(impls)
                ]
        return order

    def set_call_order(self, new_order: CallOrderDict):
        """Sets the plugin manager call order to match settings plugin values.

        Note: Run this after load_settings_plugin_defaults, which
        sets the default values in settings.

        Parameters
        ----------
        new_order : CallOrderDict
            mapping of hook_specification name, to a list of dicts with keys:
            {'plugin', 'enabled'}.  Plugins earlier in the dict are called
            sooner.
        """
        for spec_name, hook_caller in self.hooks.items():
            if spec_name in new_order:
                order = []
                for p in new_order.get(spec_name, []):
                    try:
                        plugin = p['plugin']
                        hook_impl_name = None
                        if '--' in plugin:
                            plugin, hook_impl_name = tuple(plugin.split('--'))

                        enabled = p['enabled']
                        # the plugin may not be there if its been disabled.
                        hook_caller._set_plugin_enabled(plugin, enabled)

                        hook_impls = hook_caller.get_hookimpls()
                        # get the HookImplementation objects matching this entry
                        hook_impl = list(
                            filter(
                                # plugin name has to match
                                lambda impl: impl.plugin_name == plugin
                                and (
                                    # if we have a hook_impl_name it must match
                                    not hook_impl_name
                                    or impl.function.__name__ == hook_impl_name
                                ),
                                hook_impls,
                            )
                        )
                        order.extend(hook_impl)
                    except KeyError:
                        continue
                if order:
                    hook_caller.bring_to_front(order)

    # SAMPLE DATA ---------------------------

    def register_sample_data(
        self,
        data: dict[str, str | Callable[..., Iterable[LayerData]]],
        hookimpl: HookImplementation,
    ):
        """Register sample data dict returned by `napari_provide_sample_data`.

        Each key in `data` is a `sample_name` (the string that will appear in the
        `Open Sample` menu), and the value is either a string, or a callable that
        returns an iterable of ``LayerData`` tuples, where each tuple is a 1-, 2-,
        or 3-tuple of ``(data,)``, ``(data, meta)``, or ``(data, meta,
        layer_type)``.

        Parameters
        ----------
        data : Dict[str, Union[str, Callable[..., Iterable[LayerData]]]]
            A mapping of {sample_name->data}
        hookimpl : HookImplementation
            The hook implementation that returned the dict
        """
        plugin_name = hookimpl.plugin_name
        hook_name = 'napari_provide_sample_data'
        if not isinstance(data, dict):
            warn_message = trans._(
                'Plugin {plugin_name!r} provided a non-dict object to {hook_name!r}: data ignored.',
                deferred=True,
                plugin_name=plugin_name,
                hook_name=hook_name,
            )
            warn(message=warn_message)
            return

        _data: dict[str, SampleDict] = {}
        for name, _datum in list(data.items()):
            if isinstance(_datum, dict):
                datum: SampleDict = _datum
                if 'data' not in _datum or 'display_name' not in _datum:
                    warn_message = trans._(
                        'In {hook_name!r}, plugin {plugin_name!r} provided an invalid dict object for key {name!r} that does not have required keys: "data" and "display_name". Ignoring',
                        deferred=True,
                        hook_name=hook_name,
                        plugin_name=plugin_name,
                        name=name,
                    )
                    warn(message=warn_message)
                    continue
            else:
                datum = {'data': _datum, 'display_name': name}

            if not (
                callable(datum['data'])
                or isinstance(datum['data'], (str, Path))
            ):
                warn_message = trans._(
                    'Plugin {plugin_name!r} provided invalid data for key {name!r} in the dict returned by {hook_name!r}. (Must be str, callable, or dict), got ({dtype}).',
                    deferred=True,
                    plugin_name=plugin_name,
                    name=name,
                    hook_name=hook_name,
                    dtype=type(datum['data']),
                )
                warn(message=warn_message)
                continue
            _data[name] = datum

        if plugin_name not in self._sample_data:
            self._sample_data[plugin_name] = {}

        self._sample_data[plugin_name].update(_data)

    def available_samples(self) -> tuple[tuple[str, str], ...]:
        """Return a tuple of sample data keys provided by plugins.

        Returns
        -------
        sample_keys : Tuple[Tuple[str, str], ...]
            A sequence of 2-tuples ``(plugin_name, sample_name)`` showing available
            sample data provided by plugins.  To load sample data into the viewer,
            use :meth:`napari.Viewer.open_sample`.

        Examples
        --------

        .. code-block:: python

            from napari.plugins import available_samples

            sample_keys = available_samples()
            if sample_keys:
                # load first available sample
                viewer.open_sample(*sample_keys[0])
        """
        return tuple(
            (p, s) for p in self._sample_data for s in self._sample_data[p]
        )

    # THEME DATA ------------------------------------

    def register_theme_colors(
        self,
        data: dict[str, dict[str, str | tuple | list]],
        hookimpl: HookImplementation,
    ):
        """Register theme data dict returned by `napari_experimental_provide_theme`.

        The `theme` data should be provided as an iterable containing dictionary
        of values, where the ``folder`` value will be used as theme name.
        """
        plugin_name = hookimpl.plugin_name
        hook_name = '`napari_experimental_provide_theme`'
        if not isinstance(data, dict):
            warn_message = trans._(
                'Plugin {plugin_name!r} provided a non-dict object to {hook_name!r}: data ignored',
                deferred=True,
                plugin_name=plugin_name,
                hook_name=hook_name,
            )
            warn(message=warn_message)
            return

        _data = {}
        for theme_id, theme_colors in data.items():
            try:
                theme = Theme.parse_obj(theme_colors)
                register_theme(theme_id, theme, plugin_name)
                _data[theme_id] = theme
            except (KeyError, ValidationError) as err:
                warn_msg = trans._(
                    'In {hook_name!r}, plugin {plugin_name!r} provided an invalid dict object for creating themes. {err!r}',
                    deferred=True,
                    hook_name=hook_name,
                    plugin_name=plugin_name,
                    err=err,
                )
                warn(message=warn_msg)
                continue

        if plugin_name not in self._theme_data:
            self._theme_data[plugin_name] = {}
        self._theme_data[plugin_name].update(_data)

    def unregister_theme_colors(self, plugin_name: str):
        """Unregister theme data from napari."""
        if plugin_name not in self._theme_data:
            return

        # unregister all themes that were provided by the plugins
        for theme_id in self._theme_data[plugin_name]:
            unregister_theme(theme_id)

        # since its possible that disabled/removed plugin was providing the
        # current theme, we check for this explicitly and if this the case,
        # theme is automatically changed to default `dark` theme
        settings = get_settings()
        current_theme = settings.appearance.theme
        if current_theme in self._theme_data[plugin_name]:
            settings.appearance.theme = 'dark'  # type: ignore
            warnings.warn(
                message=trans._(
                    'The current theme {current_theme!r} was provided by the plugin {plugin_name!r} which was disabled or removed. Switched theme to the default.',
                    deferred=True,
                    plugin_name=plugin_name,
                    current_theme=current_theme,
                )
            )

    def discover_themes(self) -> None:
        """Trigger discovery of theme plugins.

        As a "historic" hook, this should only need to be called once.
        (historic here means that even plugins that are discovered after this
        is called will be added.)
        """
        if self._theme_data:
            return
        self.hook.napari_experimental_provide_theme.call_historic(
            result_callback=partial(self.register_theme_colors), with_impl=True
        )

    # FUNCTION & DOCK WIDGETS -----------------------

    def iter_widgets(self) -> Iterator[tuple[str, tuple[str, dict[str, Any]]]]:
        from itertools import chain, repeat

        # The content of contribution dictionaries is name of plugin and
        # list of its names of widgets contributed by this plugin
        # as this order do not depend on the order of contributions in file
        # we sort it to make it easier searchable.

        dock_widgets = zip(
            repeat('dock'),
            (
                (name, sorted(cont))
                for name, cont in self._dock_widgets.items()
            ),
        )
        func_widgets = zip(
            repeat('func'),
            (
                (name, sorted(cont))
                for name, cont in self._function_widgets.items()
            ),
        )
        yield from chain(dock_widgets, func_widgets)

    def register_dock_widget(
        self,
        args: AugmentedWidget | list[AugmentedWidget],
        hookimpl: HookImplementation,
    ):
        plugin_name = hookimpl.plugin_name
        hook_name = '`napari_experimental_provide_dock_widget`'
        for arg in args if isinstance(args, list) else [args]:
            if isinstance(arg, tuple):
                if not arg:
                    warn_message = trans._(
                        'Plugin {plugin_name!r} provided an invalid tuple to {hook_name}.  Skipping',
                        deferred=True,
                        plugin_name=plugin_name,
                        hook_name=hook_name,
                    )
                    warn(message=warn_message)
                    continue
                _cls = arg[0]
                kwargs = arg[1] if len(arg) > 1 else {}
            else:
                _cls, kwargs = (arg, {})

            if not callable(_cls):
                warn_message = trans._(
                    'Plugin {plugin_name!r} provided a non-callable object (widget) to {hook_name}: {_cls!r}. Widget ignored.',
                    deferred=True,
                    plugin_name=plugin_name,
                    hook_name=hook_name,
                    _cls=_cls,
                )
                warn(message=warn_message)

                continue

            if not isinstance(kwargs, dict):
                warn_message = trans._(
                    'Plugin {plugin_name!r} provided invalid kwargs to {hook_name} for class {clsname}. Widget ignored.',
                    deferred=True,
                    plugin_name=plugin_name,
                    hook_name=hook_name,
                    clsname=_cls.__name__,
                )
                warn(message=warn_message)
                continue

            # Get widget name
            name = str(kwargs.get('name', '')) or camel_to_spaces(
                _cls.__name__
            )

            if plugin_name not in self._dock_widgets:
                # tried defaultdict(dict) but got odd KeyErrors...
                self._dock_widgets[plugin_name] = {}
            elif name in self._dock_widgets[plugin_name]:
                warn_message = trans._(
                    'Plugin {plugin_name!r} has already registered a dock widget {name!r} which has now been overwritten',
                    deferred=True,
                    plugin_name=plugin_name,
                    name=name,
                )
                warn(message=warn_message)

            self._dock_widgets[plugin_name][name] = (_cls, kwargs)

    def register_function_widget(
        self,
        args: Callable | list[Callable],
        hookimpl: HookImplementation,
    ):
        plugin_name = hookimpl.plugin_name
        hook_name = '`napari_experimental_provide_function`'
        for func in args if isinstance(args, list) else [args]:
            if not isinstance(func, FunctionType):
                warn_message = trans._(
                    'Plugin {plugin_name!r} provided a non-callable type to {hook_name}: {functype!r}. Function widget ignored.',
                    deferred=True,
                    functype=type(func),
                    plugin_name=plugin_name,
                    hook_name=hook_name,
                )

                if isinstance(func, tuple):
                    warn_message += trans._(
                        ' To provide multiple function widgets please use a LIST of callables',
                        deferred=True,
                    )
                warn(message=warn_message)
                continue

            # Get function name
            name = func.__name__.replace('_', ' ')

            if plugin_name not in self._function_widgets:
                # tried defaultdict(dict) but got odd KeyErrors...
                self._function_widgets[plugin_name] = {}
            elif name in self._function_widgets[plugin_name]:
                warn_message = trans._(
                    'Plugin {plugin_name!r} has already registered a function widget {name!r} which has now been overwritten',
                    deferred=True,
                    plugin_name=plugin_name,
                    name=name,
                )
                warn(message=warn_message)

            self._function_widgets[plugin_name][name] = func

    def discover_sample_data(self):
        if self._sample_data:
            return
        self.hook.napari_provide_sample_data.call_historic(
            result_callback=partial(self.register_sample_data), with_impl=True
        )

    def discover_widgets(self):
        """Trigger discovery of dock_widgets plugins.

        As a "historic" hook, this should only need to be called once.
        (historic here means that even plugins that are discovered after this
        is called will be added.)
        """

        if self._dock_widgets:
            return
        self.hook.napari_experimental_provide_dock_widget.call_historic(
            partial(self.register_dock_widget), with_impl=True
        )
        self.hook.napari_experimental_provide_function.call_historic(
            partial(self.register_function_widget), with_impl=True
        )

    def get_widget(
        self, plugin_name: str, widget_name: str | None = None
    ) -> tuple[WidgetCallable, dict[str, Any]]:
        """Get widget `widget_name` provided by plugin `plugin_name`.

        Note: it's important that :func:`discover_dock_widgets` has been called
        first, otherwise plugins may not be found yet.  (Typically, that is done
        in qt_main_window)

        Parameters
        ----------
        plugin_name : str
            Name of a plugin providing a widget
        widget_name : str, optional
            Name of a widget provided by `plugin_name`. If `None`, and the
            specified plugin provides only a single widget, that widget will be
            returned, otherwise a ValueError will be raised, by default None

        Returns
        -------
        plugin_widget : Tuple[Callable, dict]
            Tuple of (widget_class, options).

        Raises
        ------
        KeyError
            If plugin `plugin_name` is not installed or does not provide any widgets.
        KeyError
            If plugin does not provide a widget named `widget_name`.
        ValueError
            If `widget_name` is not provided, but `plugin_name` provides more than
            one widget
        """
        plg_wdgs = self._dock_widgets.get(plugin_name)
        if not plg_wdgs:
            msg = trans._(
                'Plugin {plugin_name!r} is not installed or does not provide any dock widgets',
                plugin_name=plugin_name,
                deferred=True,
            )
            raise KeyError(msg)

        if not widget_name:
            if len(plg_wdgs) > 1:
                msg = trans._(
                    'Plugin {plugin_name!r} provides more than 1 dock_widget. Must also provide "widget_name" from {avail}',
                    avail=set(plg_wdgs),
                    plugin_name=plugin_name,
                    deferred=True,
                )
                raise ValueError(msg)

            widget_name = next(iter(plg_wdgs))
        else:
            if widget_name not in plg_wdgs:
                msg = trans._(
                    'Plugin {plugin_name!r} does not provide a widget named {widget_name!r}',
                    plugin_name=plugin_name,
                    widget_name=widget_name,
                    deferred=True,
                )
                raise KeyError(msg)

        return plg_wdgs[widget_name]

    def get_reader_for_extension(self, extension: str) -> str | None:
        """Return reader plugin assigned to `extension`, or None."""
        return self._get_plugin_for_extension(extension, type_='reader')

    def assign_reader_to_extensions(
        self, reader: str, extensions: str | Iterable[str]
    ) -> None:
        """Assign a specific reader plugin to `extensions`.

        Parameters
        ----------
        reader : str
            Name of a plugin offering a reader hook.
        extensions : Union[str, Iterable[str]]
            Name(s) of extensions to always write with `reader`
        """
        from napari.settings import get_settings

        self._assign_plugin_to_extensions(reader, extensions, type_='reader')
        extension2readers = get_settings().plugins.extension2reader
        get_settings().plugins.extension2reader = {
            **extension2readers,
            **self._extension2reader,
        }

    def get_writer_for_extension(self, extension: str) -> str | None:
        """Return writer plugin assigned to `extension`, or None."""
        return self._get_plugin_for_extension(extension, type_='writer')

    def assign_writer_to_extensions(
        self, writer: str, extensions: str | Iterable[str]
    ) -> None:
        """Assign a specific writer plugin to `extensions`.

        Parameters
        ----------
        writer : str
            Name of a plugin offering a writer hook.
        extensions : Union[str, Iterable[str]]
            Name(s) of extensions to always write with `writer`
        """
        from napari.settings import get_settings

        self._assign_plugin_to_extensions(writer, extensions, type_='writer')
        get_settings().plugins.extension2writer = self._extension2writer

    def _get_plugin_for_extension(
        self, extension: str, type_: str
    ) -> str | None:
        """helper method for public get_<type_>_for_extension functions."""
        ext_map = getattr(self, f'_extension2{type_}', None)
        if ext_map is None:
            raise ValueError(
                trans._(
                    'invalid plugin type: {type_!r}',
                    deferred=True,
                    type_=type_,
                )
            )

        if not extension.startswith('.'):
            extension = f'.{extension}'

        plugin = ext_map.get(extension)
        # make sure it's still an active plugin
        if plugin and (plugin not in self.plugins):
            del self.ext_map[plugin]
            return None
        return plugin

    def _assign_plugin_to_extensions(
        self,
        plugin: str,
        extensions: str | Iterable[str],
        type_: str | None = None,
    ) -> None:
        """helper method for public assign_<type_>_to_extensions functions."""
        caller: HookCaller = getattr(self.hook, f'napari_get_{type_}', None)
        if caller is None:
            raise ValueError(
                trans._(
                    'invalid plugin type: {type_!r}',
                    deferred=True,
                    type_=type_,
                )
            )

        plugins = caller.get_hookimpls()
        if plugin not in {p.plugin_name for p in plugins}:
            msg = trans._(
                '{plugin!r} is not a valid {type_} plugin name',
                plugin=plugin,
                type_=type_,
                deferred=True,
            )
            raise ValueError(msg)

        ext_map = getattr(self, f'_extension2{type_}')
        if isinstance(extensions, str):
            extensions = [extensions]
        for ext in extensions:
            if not ext.startswith('.'):
                ext = f'.{ext}'
            ext_map[ext] = plugin

            func = None
            # give warning that plugin *may* not be able to read that extension
            with contextlib.suppress(Exception):
                func = caller._call_plugin(plugin, path=f'_testing_{ext}')
            if func is None:
                msg = trans._(
                    'plugin {plugin!r} did not return a {type_} function when provided a path ending in {ext!r}. This *may* indicate a typo?',
                    deferred=True,
                    plugin=plugin,
                    type_=type_,
                    ext=ext,
                )
                warn(msg)
