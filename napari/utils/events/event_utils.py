from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol

    class Emitter(Protocol):
        def connect(self, callback: Callable): ...

        def disconnect(self, callback: Callable): ...


def disconnect_events(emitter, listener):
    """Disconnect all events between an emitter group and a listener.

    Parameters
    ----------
    emitter : napari.utils.events.event.EmitterGroup
        Emitter group.
    listener : Object
        Any object that has been connected to.
    """
    for em in emitter.emitters.values():
        em.disconnect(listener)


def connect_setattr(emitter: Emitter, obj, attr: str):
    ref = weakref.ref(obj)

    def _cb(*value):
        if (ob := ref()) is None:
            emitter.disconnect(_cb)
            return
        setattr(ob, attr, value[0] if len(value) == 1 else value)

    emitter.connect(_cb)
    # There are scenarios where emitter is deleted before obj.
    # Also there is no option to create weakref to QT Signal
    # but even if keep reference to base object and signal name it is possible to meet
    # problem with C++ "wrapped C/C++ object has been deleted"

    # In all of these 3 functions, this should be uncommented instead of using
    # the if clause in _cb but that causes a segmentation fault in tests
    # weakref.finalize(obj, emitter.disconnect, _cb)


def connect_no_arg(emitter: Emitter, obj, attr: str):
    ref = weakref.ref(obj)

    def _cb(*_value):
        if (ob := ref()) is None:
            emitter.disconnect(_cb)
            return
        getattr(ob, attr)()

    emitter.connect(_cb)
    # as in connect_setattr
    # weakref.finalize(obj, emitter.disconnect, _cb)


def connect_setattr_value(emitter: Emitter, obj, attr: str):
    """To get value from Event"""
    ref = weakref.ref(obj)

    def _cb(value):
        if (ob := ref()) is None:
            emitter.disconnect(_cb)
            return
        setattr(ob, attr, value.value)

    emitter.connect(_cb)
    # weakref.finalize(obj, emitter.disconnect, _cb)
