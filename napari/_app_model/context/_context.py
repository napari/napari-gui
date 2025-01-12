from __future__ import annotations

import collections.abc
from typing import TYPE_CHECKING, Any, Final

from app_model.expressions import (
    Context,
    create_context as _create_context,
    get_context as _get_context,
)

from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.utils.events import Event

__all__ = ['Context', 'SettingsAwareContext', 'create_context', 'get_context']


class ContextMapping(collections.abc.Mapping):
    """Wrap app-model contexts, allowing keys to be evaluated at query time.

    `ContextMapping` objects are created from a context any time someone calls
    `NapariApplication.get_context`. This usually happens just before a menu is
    about to be shown, when we update the menu's actions' states based on the
    values of the context keys. The call to `get_context` triggers the creation
    of the `ContextMapping` which stores (or, in the case of functional keys,
    evaluates then stores) the value of each context key. Once keys are
    evaluated, they are cached within the object for future accessing of the
    same keys. However, any new `get_context` calls will create a brand new
    `ContextMapping` object.
    """

    def __init__(self, initial_values: collections.abc.Mapping):
        self._initial_context_mapping = initial_values
        self._evaluated_context_mapping: dict[str, Any] = {}

    def __getitem__(self, key):
        if key in self._evaluated_context_mapping:
            return self._evaluated_context_mapping[key]
        if key not in self._initial_context_mapping:
            raise KeyError(f'Key {key!r} not found')
        value = self._initial_context_mapping[key]
        if callable(value):
            value = value()
        self._evaluated_context_mapping[key] = value
        return value

    def __contains__(self, item):
        return item in self._initial_context_mapping

    def __len__(self):
        return len(self._initial_context_mapping)

    def __iter__(self):
        return iter(self._initial_context_mapping)


class SettingsAwareContext(Context):
    """A special context that allows access of settings using `settings.`

    This takes no parents, and will always be a root context.
    """

    _PREFIX: Final[str] = 'settings.'

    def __init__(self) -> None:
        super().__init__()
        from napari.settings import get_settings

        self._settings = get_settings()
        self._settings.events.changed.connect(self._update_key)

    def _update_key(self, event: Event):
        self.changed.emit({f'{self._PREFIX}{event.key}'})

    def __del__(self):
        self._settings.events.changed.disconnect(self._update_key)

    def __missing__(self, key: str) -> Any:
        if key.startswith(self._PREFIX):
            splits = [k for k in key.split('.')[1:] if k]
            val: Any = self._settings
            if splits:
                while splits:
                    val = getattr(val, splits.pop(0))
                if hasattr(val, 'dict'):
                    val = val.dict()
                return val
        return super().__missing__(key)

    def new_child(self, m: dict | None = None) -> Context:  # type: ignore
        """New ChainMap with a new map followed by all previous maps.

        If no map is provided, an empty dict is used.
        """
        # important to use self, not *self.maps
        return Context(m or {}, self)  # type: ignore

    def __setitem__(self, k: str, v: Any) -> None:
        if k.startswith(self._PREFIX):
            raise ValueError(
                trans._(
                    'Cannot set key starting with {prefix!r}',
                    deferred=True,
                    prefix=self._PREFIX,
                )
            )

        return super().__setitem__(k, v)

    def __bool__(self):
        # settings mappings are always populated, so we can always return True
        return True


def create_context(
    obj: object,
    max_depth: int = 20,
    start: int = 2,
    root: Context | None = None,
) -> Context:
    return _create_context(
        obj=obj,
        max_depth=max_depth,
        start=start,
        root=root,
        root_class=SettingsAwareContext,
    )


def get_context(obj: object) -> ContextMapping:
    return ContextMapping(_get_context(obj))
