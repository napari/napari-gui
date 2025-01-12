"""MutableMapping that emits events when altered."""

from collections.abc import Mapping, Sequence

from napari.utils.events.containers._dict import _K, _T, TypedMutableMapping
from napari.utils.events.event import EmitterGroup, Event
from napari.utils.events.types import SupportsEvents


class EventedDict(TypedMutableMapping[_K, _T]):
    """Mutable dictionary that emits events when altered.

    This class is designed to behave exactly like builtin ``dict``, but
    will emit events before and after all mutations (addition, removal, and
    changing).

    Parameters
    ----------
    data : Mapping, optional
        Dictionary to initialize the class with.
    basetype : type of sequence of types, optional
        Type of the element in the dictionary.

    Events
    ------
    changing (key: K)
        emitted before an item at ``key`` is changed
    changed (key: K, old_value: T, value: T)
        emitted when item at ``key`` is changed from ``old_value`` to ``value``
    adding (key: K)
        emitted before an item is added to the dictionary with ``key``
    added (key: K, value: T)
        emitted after ``value`` was added to the dictionary with ``key``
    removing (key: K)
        emitted before ``key`` is removed from the dictionary
    removed (key: K, value: T)
        emitted after ``key`` was removed from the dictionary
    updated (key, K, value: T)
        emitted after ``value`` of ``key`` was changed. Only implemented by
        subclasses to give them an option to trigger some update after ``value``
        was changed and this class did not register it. This can be useful if
        the ``basetype`` is not an evented object.
    """

    events: EmitterGroup

    def __init__(
        self,
        data: Mapping[_K, _T] | None = None,
        basetype: type[_T] | Sequence[type[_T]] = (),
    ) -> None:
        _events = {
            'changing': None,
            'changed': None,
            'adding': None,
            'added': None,
            'removing': None,
            'removed': None,
            'updated': None,
        }
        # For inheritance: If the mro already provides an EmitterGroup, add...
        if hasattr(self, 'events') and isinstance(self.events, EmitterGroup):
            self.events.add(**_events)
        else:
            # otherwise create a new one
            self.events = EmitterGroup(
                source=self, auto_connect=False, **_events
            )
        super().__init__(data, basetype)

    def __setitem__(self, key: _K, value: _T) -> None:
        old = self._dict.get(key)
        if value is old or value == old:
            return
        if old is None:
            self.events.adding(key=key)
            super().__setitem__(key, value)
            self.events.added(key=key, value=value)
            self._connect_child_emitters(value)
        else:
            self.events.changing(key=key)
            super().__setitem__(key, value)
            self.events.changed(key=key, old_value=old, value=value)

    def __delitem__(self, key: _K) -> None:
        self.events.removing(key=key)
        self._disconnect_child_emitters(self[key])
        item = self._dict.pop(key)
        self.events.removed(key=key, value=item)

    def _reemit_child_event(self, event: Event) -> None:
        """An item in the dict emitted an event.  Re-emit with key"""
        if not hasattr(event, 'key'):
            event.key = self.key(event.source)

        # re-emit with this object's EventEmitter
        self.events(event)

    def _disconnect_child_emitters(self, child: _T) -> None:
        """Disconnect all events from the child from the re-emitter."""
        if isinstance(child, SupportsEvents):
            child.events.disconnect(self._reemit_child_event)

    def _connect_child_emitters(self, child: _T) -> None:
        """Connect all events from the child to be re-emitted."""
        if isinstance(child, SupportsEvents):
            # make sure the event source has been set on the child
            if child.events.source is None:
                child.events.source = child
            child.events.connect(self._reemit_child_event)

    def key(self, value: _T) -> _K | None:
        """Return first instance of value."""
        for k, v in self._dict.items():
            if v is value or v == value:
                return k
        return None
