from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from inspect import isgeneratorfunction
from typing import TYPE_CHECKING, Any

from napari.utils.events import EmitterGroup
from napari.utils.interactions import Shortcut
from napari.utils.translations import trans

if TYPE_CHECKING:
    from concurrent.futures import Future
    from typing import Protocol

    from napari.utils.key_bindings import KeymapProvider

    class SignalInstance(Protocol):
        def connect(self, callback: Callable) -> None: ...

    class Button(Protocol):
        clicked: SignalInstance

        def setToolTip(self, text: str) -> None: ...

    class ShortcutEvent:
        name: str
        shortcut: str
        tooltip: str


@dataclass
class Action:
    command: Callable
    description: str
    keymapprovider: KeymapProvider  # subclassclass or instance of a subclass
    repeatable: bool = False

    @cached_property
    def injected(self) -> Callable[..., Future]:
        """command with napari objects injected.

        This will inject things like the current viewer, or currently selected
        layer into the commands.  See :func:`inject_napari_dependencies` for
        details.
        """
        from napari._app_model import get_app_model

        return get_app_model().injection_store.inject(self.command)


class ActionManager:
    """
    Manage the bindings between buttons; shortcuts, callbacks gui elements...

    The action manager is aware of the various buttons, keybindings and other
    elements that may trigger an action and is able to synchronise all of those.
    Thus when a shortcut is bound; this should be capable of updating the
    buttons tooltip menus etc to show the shortcuts, descriptions...

    In most cases this should also allow to bind non existing shortcuts,
    actions, buttons, in which case they will be bound only once the actions are
    registered.


    >>> def callback(qtv, number):
    ...     qtv.dims[number] +=1

    >>> action_manager.register_action('bump one', callback,
    ...     'Add one to dims',
    ...     None)

    The callback signature is going to be inspected and required globals passed
    in.
    """

    _actions: dict[str, Action]

    def __init__(self) -> None:
        # map associating a name/id with a Comm
        self._actions: dict[str, Action] = {}
        self._shortcuts: dict[str, list[str]] = defaultdict(list)
        self._stack: list[str] = []
        self._tooltip_include_action_name = False
        self.events = EmitterGroup(source=self, shorcut_changed=None)

    def _debug(self, val):
        self._tooltip_include_action_name = val

    def _validate_action_name(self, name):
        if len(name.split(':')) != 2:
            raise ValueError(
                trans._(
                    'Action names need to be in the form `package:name`, got {name!r}',
                    name=name,
                    deferred=True,
                )
            )

    def register_action(
        self,
        name: str,
        command: Callable,
        description: str,
        keymapprovider: KeymapProvider | None,
        repeatable: bool = False,
    ):
        """
        Register an action for future usage

        An action is generally a callback associated with
         - a name (unique), usually `packagename:name`
         - a description
         - A keymap provider (easier for focus and backward compatibility).
         - a boolean repeatability flag indicating whether it can be auto-
           repeated (i.e. when a key is held down); defaults to False

        Actions can then be later bound/unbound from button elements, and
        shortcuts; and the action manager will take care of modifying the keymap
        of instances to handle shortcuts; and UI elements to have tooltips with
        descriptions and shortcuts;

        Parameters
        ----------
        name : str
            unique name/id of the command that can be used to refer to this
            command
        command : callable
            take 0, or 1 parameter; if `keymapprovider` is not None, will be
            called with `keymapprovider` as first parameter.
        description : str
            Long string to describe what the command does, will be used in
            tooltips.
        keymapprovider : KeymapProvider
            KeymapProvider class or instance to use to bind the shortcut(s) when
            registered. This make sure the shortcut is active only when an
            instance of this is in focus.
        repeatable : bool
            a boolean flag indicating whether the action can be autorepeated.
            Defaults to False.


        Notes
        -----
        Registering an action, binding buttons and shortcuts can happen in any
        order and should have the same effect. In particular registering an
        action can happen later (plugin loading), while user preference
        (keyboard shortcut), has already been happen. When this is the case, the
        button and shortcut binding is delayed until an action with the
        corresponding name is registered.

        See Also
        --------
        bind_button, bind_shortcut

        """

        self._validate_action_name(name)
        self._actions[name] = Action(
            command, description, keymapprovider, repeatable
        )
        if keymapprovider:
            self._update_shortcut_bindings(name)

    def _update_shortcut_bindings(self, name: str):
        """
        Update the key mappable for given action name
        to trigger the action within the given context and
        """
        if name not in self._actions:
            return
        if name not in self._shortcuts:
            return
        action = self._actions[name]
        km_provider = action.keymapprovider
        if hasattr(km_provider, 'bind_key'):
            for shortcut in self._shortcuts[name]:
                # NOTE: it would be better if we could bind `self.trigger` here
                # as it allow the action manager to be a convenient choke point
                # to monitor all commands (useful for undo/redo, etc...), but
                # the generator pattern in the keybindings caller makes that
                # difficult at the moment, since `self.trigger(name)` is not a
                # generator function (but action.injected is)
                km_provider.bind_key(shortcut, action.injected, overwrite=True)

    def bind_button(
        self, name: str, button: Button, extra_tooltip_text=''
    ) -> None:
        """
        Bind `button` to trigger Action `name` on click.

        Parameters
        ----------
        name : str
            name of the corresponding action in the form ``packagename:name``
        button : Button
            A object providing Button interface (like QPushButton) that, when
            clicked, should trigger the action. The tooltip will be set to the
            action description and the corresponding shortcut if available.
        extra_tooltip_text : str
            Extra text to add at the end of the tooltip. This is useful to
            convey more information about this action as the action manager may
            update the tooltip based on the action name.

        Notes
        -----
        calling `bind_button` can be done before an action with the
        corresponding name is registered, in which case the effect will be
        delayed until the corresponding action is registered.

        Note: this method cannot be used with generator functions,
        see https://github.com/napari/napari/issues/4164 for details.
        """
        self._validate_action_name(name)

        if (action := self._actions.get(name)) and isgeneratorfunction(
            getattr(action, 'command', None)
        ):
            raise ValueError(
                trans._(
                    '`bind_button` cannot be used with generator functions',
                    deferred=True,
                )
            )

        def _trigger():
            self.trigger(name)

        button.clicked.connect(_trigger)
        if name in self._actions:
            button.setToolTip(
                f'{self._build_tooltip(name)} {extra_tooltip_text}'
            )

        def _update_tt(event: ShortcutEvent):
            if event.name == name:
                button.setToolTip(f'{event.tooltip} {extra_tooltip_text}')

        # if it's a QPushbutton, we'll remove it when it gets destroyed
        until = getattr(button, 'destroyed', None)
        self.events.shorcut_changed.connect(_update_tt, until=until)

    def bind_shortcut(self, name: str, shortcut: str) -> None:
        """
        bind shortcut `shortcut` to trigger action `name`

        Parameters
        ----------
        name : str
            name of the corresponding action in the form ``packagename:name``
        shortcut : str
            Shortcut to assign to this action use dash as separator. See
            `Shortcut` for known modifiers.

        Notes
        -----
        calling `bind_button` can be done before an action with the
        corresponding name is registered, in which case the effect will be
        delayed until the corresponding action is registered.
        """
        self._validate_action_name(name)
        if shortcut in self._shortcuts[name]:
            return
        self._shortcuts[name].append(shortcut)
        self._update_shortcut_bindings(name)
        self._emit_shortcut_change(name, shortcut)

    def unbind_shortcut(self, name: str) -> list[str] | None:
        """
        Unbind all shortcuts for a given action name.

        Parameters
        ----------
        name : str
            name of the action in the form `packagename:name` to unbind.

        Returns
        -------
        shortcuts: set of str | None
            Previously bound shortcuts or None if not such shortcuts was bound,
            or no such action exists.

        Warns
        -----
        UserWarning:
            When trying to unbind an action unknown form the action manager,
            this warning will be emitted.

        """
        action = self._actions.get(name, None)
        if action is None:
            warnings.warn(
                trans._(
                    'Attempting to unbind an action which does not exists ({name}), this may have no effects. This can happen if your settings are out of date, if you upgraded napari, upgraded or deactivated a plugin, or made a typo in in your custom keybinding.',
                    name=name,
                ),
                UserWarning,
                stacklevel=2,
            )

        shortcuts = self._shortcuts.get(name)
        if shortcuts:
            if action and hasattr(action.keymapprovider, 'bind_key'):
                for shortcut in shortcuts:
                    action.keymapprovider.bind_key(shortcut)(None)
            del self._shortcuts[name]

        self._emit_shortcut_change(name)
        return shortcuts

    def _emit_shortcut_change(self, name: str, shortcut=''):
        tt = self._build_tooltip(name) if name in self._actions else ''
        self.events.shorcut_changed(name=name, shortcut=shortcut, tooltip=tt)

    def _build_tooltip(self, name: str) -> str:
        """Build tooltip for action `name`."""
        ttip = self._actions[name].description

        if name in self._shortcuts:
            jstr = ' ' + trans._p('<keysequence> or <keysequence>', 'or') + ' '
            shorts = jstr.join(f'{Shortcut(s)}' for s in self._shortcuts[name])
            ttip += f' ({shorts})'

        ttip += f'[{name}]' if self._tooltip_include_action_name else ''
        return ttip

    def _get_layer_shortcuts(self, layers) -> dict:
        """
        Get shortcuts filtered by the given layers.

        Parameters
        ----------
        layers : list of layers
            Layers to use for shortcuts filtering.

        Returns
        -------
        dict
            Dictionary of layers with dictionaries of shortcuts to
            descriptions.
        """
        layer_shortcuts = {}
        for layer in layers:
            layer_shortcuts[layer] = {}
            for name, shortcuts in self._shortcuts.items():
                action = self._actions.get(name, None)
                if action and layer == action.keymapprovider:
                    for shortcut in shortcuts:
                        layer_shortcuts[layer][str(shortcut)] = (
                            action.description
                        )

        return layer_shortcuts

    def _get_provider_actions(self, provider) -> dict:
        """
        Get actions filtered by the given provider.

        Parameters
        ----------
        provider : KeymapProvider
            Provider to use for actions filtering.

        Returns
        -------
        provider_actions: dict
            Dictionary of names of actions with action values for a provider.

        """
        return {
            name: action
            for name, action in self._actions.items()
            if action and provider == action.keymapprovider
        }

    def _get_active_shortcuts(self, active_keymap):
        """
        Get active shortcuts for the given active keymap.

        Parameters
        ----------
        active_keymap : KeymapProvider
            The active keymap provider.

        Returns
        -------
        dict
            Dictionary of shortcuts to descriptions.
        """
        active_func_names = [i[1].__name__ for i in active_keymap.items()]
        active_shortcuts = {}
        for name, shortcuts in self._shortcuts.items():
            action = self._actions.get(name, None)
            if action and action.command.__name__ in active_func_names:
                for shortcut in shortcuts:
                    active_shortcuts[str(shortcut)] = action.description

        return active_shortcuts

    def _get_repeatable_shortcuts(self, active_keymap) -> list:
        """
        Get active, repeatable shortcuts for the given active keymap.

        Parameters
        ----------
        active_keymap : KeymapProvider
            The active keymap provider.

        Returns
        -------
        list
            List of shortcuts that are repeatable.
        """
        active_func_names = {i[1].__name__ for i in active_keymap.items()}
        active_repeatable_shortcuts = []
        for name, shortcuts in self._shortcuts.items():
            action = self._actions.get(name, None)
            if (
                action
                and action.command.__name__ in active_func_names
                and action.repeatable
            ):
                active_repeatable_shortcuts.extend(shortcuts)

        return active_repeatable_shortcuts

    def trigger(self, name: str) -> Any:
        """Trigger the action `name`."""
        return self._actions[name].injected()


action_manager = ActionManager()
