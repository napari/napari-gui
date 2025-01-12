# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# # LICENSE.txt
# Vispy licensing terms
# ---------------------

# Vispy is licensed under the terms of the (new) BSD license:
#
# Copyright (c) 2013-2017, Vispy Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of Vispy Development Team nor the names of its
#   contributors may be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Exceptions
# ----------
#
# The examples code in the examples directory can be considered public
# domain, unless otherwise indicated in the corresponding source file.

"""
The event module implements the classes that make up the event system.
The Event class and its subclasses are used to represent "stuff that happens".
The EventEmitter class provides an interface to connect to events and
to emit events. The EmitterGroup groups EventEmitter objects.

For more information see http://github.com/vispy/vispy/wiki/API_Events

"""

import contextlib
import inspect
import os
import warnings
import weakref
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import partial
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
)

from vispy.util.logs import _handle_exception

from napari.utils.migrations import rename_argument
from napari.utils.translations import trans


class Event:
    """Class describing events that occur and can be reacted to with callbacks.
    Each event instance contains information about a single event that has
    occurred such as a key press, mouse motion, timer activation, etc.

    Subclasses: :class:`KeyEvent`, :class:`MouseEvent`, :class:`TouchEvent`,
    :class:`StylusEvent`

    The creation of events and passing of events to the appropriate callback
    functions is the responsibility of :class:`EventEmitter` instances.

    Note that each event object has an attribute for each of the input
    arguments listed below.

    Parameters
    ----------
    type : str
       String indicating the event type (e.g. mouse_press, key_release)
    native : object (optional)
       The native GUI event object
    **kwargs : keyword arguments
        All extra keyword arguments become attributes of the event object.
    """

    @rename_argument(
        from_name='type',
        to_name='type_name',
        version='0.6.0',
        since_version='0.4.18',
    )
    def __init__(
        self, type_name: str, native: Any = None, **kwargs: Any
    ) -> None:
        # stack of all sources this event has been emitted through
        self._sources: list[Any] = []
        self._handled: bool = False
        self._blocked: bool = False
        # Store args
        self._type = type_name
        self._native = native
        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def source(self) -> Any:
        """The object that the event applies to (i.e. the source of the event)."""
        return self._sources[-1] if self._sources else None

    @property
    def sources(self) -> list[Any]:
        """List of objects that the event applies to (i.e. are or have
        been a source of the event). Can contain multiple objects in case
        the event traverses a hierarchy of objects.
        """
        return self._sources

    def _push_source(self, source):
        self._sources.append(source)

    def _pop_source(self):
        return self._sources.pop()

    @property
    def type(self) -> str:
        # No docstring; documented in class docstring
        return self._type

    @property
    def native(self) -> Any:
        # No docstring; documented in class docstring
        return self._native

    @property
    def handled(self) -> bool:
        """This boolean property indicates whether the event has already been
        acted on by an event handler. Since many handlers may have access to
        the same events, it is recommended that each check whether the event
        has already been handled as well as set handled=True if it decides to
        act on the event.
        """
        return self._handled

    @handled.setter
    def handled(self, val) -> None:
        self._handled = bool(val)

    @property
    def blocked(self) -> bool:
        """This boolean property indicates whether the event will be delivered
        to event callbacks. If it is set to True, then no further callbacks
        will receive the event. When possible, it is recommended to use
        Event.handled rather than Event.blocked.
        """
        return self._blocked

    @blocked.setter
    def blocked(self, val) -> None:
        self._blocked = bool(val)

    def __repr__(self) -> str:
        # Try to generate a nice string representation of the event that
        # includes the interesting properties.
        # need to keep track of depth because it is
        # very difficult to avoid excessive recursion.
        global _event_repr_depth
        _event_repr_depth += 1
        try:
            if _event_repr_depth > 2:
                return '<...>'
            attrs = []
            for name in dir(self):
                if name.startswith('_'):
                    continue
                # select only properties
                if not hasattr(type(self), name) or not isinstance(
                    getattr(type(self), name), property
                ):
                    continue
                attr = getattr(self, name)

                attrs.append(f'{name}={attr!r}')
        finally:
            _event_repr_depth -= 1
        return f'<{self.__class__.__name__} {" ".join(attrs)}>'

    def __str__(self) -> str:
        """Shorter string representation"""
        return self.__class__.__name__

    # mypy fix for dynamic attribute access
    def __getattr__(self, name: str) -> Any:
        return object.__getattribute__(self, name)


_event_repr_depth = 0


Callback = Union[Callable[[Event], None], Callable[[], None]]
CallbackRef = tuple['weakref.ReferenceType[Any]', str]  # dereferenced method
CallbackStr = tuple[
    Union['weakref.ReferenceType[Any]', object], str
]  # dereferenced method


_T = TypeVar('_T')


class _WeakCounter(Generic[_T]):
    """
    Similar to collection counter but has weak keys.

    It will only implement the methods we use here.
    """

    def __init__(self) -> None:
        self._counter: weakref.WeakKeyDictionary[_T, int] = (
            weakref.WeakKeyDictionary()
        )
        self._nonecount = 0

    def update(self, iterable: Iterable[_T]):
        for it in iterable:
            if it is None:
                self._nonecount += 1
            else:
                self._counter[it] = self.get(it, 0) + 1

    def get(self, key: _T, default: int) -> int:
        if key is None:
            return self._nonecount
        return self._counter.get(key, default)


class EventEmitter:
    """Encapsulates a list of event callbacks.

    Each instance of EventEmitter represents the source of a stream of similar
    events, such as mouse click events or timer activation events. For
    example, the following diagram shows the propagation of a mouse click
    event to the list of callbacks that are registered to listen for that
    event::

       User clicks    |Canvas creates
       mouse on       |MouseEvent:                |'mouse_press' EventEmitter:         |callbacks in sequence: # noqa
       Canvas         |                           |                                    |  # noqa
                   -->|event = MouseEvent(...) -->|Canvas.events.mouse_press(event) -->|callback1(event)  # noqa
                      |                           |                                 -->|callback2(event)  # noqa
                      |                           |                                 -->|callback3(event)  # noqa

    Callback functions may be added or removed from an EventEmitter using
    :func:`connect() <vispy.event.EventEmitter.connect>` or
    :func:`disconnect() <vispy.event.EventEmitter.disconnect>`.

    Calling an instance of EventEmitter will cause each of its callbacks
    to be invoked in sequence. All callbacks are invoked with a single
    argument which will be an instance of :class:`Event <vispy.event.Event>`.

    EventEmitters are generally created by an EmitterGroup instance.

    Parameters
    ----------
    source : object
        The object that the generated events apply to. All emitted Events will
        have their .source property set to this value.
    type_name: str or None
        String indicating the event type (e.g. mouse_press, key_release)
    event_class : subclass of Event
        The class of events that this emitter will generate.
    """

    @rename_argument('type', 'type_name', '0.6.0', '0.4.18')
    def __init__(
        self,
        source: Any = None,
        type_name: str | None = None,
        event_class: type[Event] = Event,
    ) -> None:
        # connected callbacks
        self._callbacks: list[Callback | CallbackRef] = []
        # used when connecting new callbacks at specific positions
        self._callback_refs: list[str | None] = []
        self._callback_pass_event: list[bool] = []

        # count number of times this emitter is blocked for each callback.
        self._blocked: dict[Callback | None, int] = {None: 0}
        self._block_counter: _WeakCounter[Callback | None] = _WeakCounter()

        # used to detect emitter loops
        self._emitting = False
        self.source = source
        self.default_args = {}
        if type_name is not None:
            self.default_args['type_name'] = type_name

        assert inspect.isclass(event_class)
        self.event_class = event_class

        self._ignore_callback_errors: bool = False  # True
        self.print_callback_errors = 'reminders'  # 'reminders'

    @property
    def ignore_callback_errors(self) -> bool:
        """Whether exceptions during callbacks will be caught by the emitter

        This allows it to continue invoking other callbacks if an error
        occurs.
        """
        return self._ignore_callback_errors

    @ignore_callback_errors.setter
    def ignore_callback_errors(self, val: bool):
        self._ignore_callback_errors = val

    @property
    def print_callback_errors(self) -> str:
        """Print a message and stack trace if a callback raises an exception

        Valid values are "first" (only show first instance), "reminders" (show
        complete first instance, then counts), "always" (always show full
        traceback), or "never".

        This assumes ignore_callback_errors=True. These will be raised as
        warnings, so ensure that the vispy logging level is set to at
        least "warning".
        """
        return self._print_callback_errors

    @print_callback_errors.setter
    def print_callback_errors(
        self,
        val: Literal['first', 'reminders', 'always', 'never'],
    ):
        if val not in ('first', 'reminders', 'always', 'never'):
            raise ValueError(
                trans._(
                    'print_callback_errors must be "first", "reminders", "always", or "never"',
                    deferred=True,
                )
            )
        self._print_callback_errors = val

    @property
    def callback_refs(self) -> tuple[str | None, ...]:
        """The set of callback references"""
        return tuple(self._callback_refs)

    @property
    def callbacks(self) -> tuple[Callback | CallbackRef, ...]:
        """The set of callbacks"""
        return tuple(self._callbacks)

    @property
    def source(self) -> Any:
        """The object that events generated by this emitter apply to"""
        return (
            None if self._source is None else self._source()
        )  # get object behind weakref

    @source.setter
    def source(self, s):
        self._source = None if s is None else weakref.ref(s)

    def _is_core_callback(self, callback: CallbackRef | Callback, core: str):
        """
        Check if the callback is a core callback

        Parameters
        ----------
        callback : Union[CallbackRef, Callback]
            The callback to check. Callback could be function or
            weak reference to object method coded using weakreference
            to object and method name stored in tuple.
        core : str
            Name of core module, for example 'napari'.
        """
        if isinstance(callback, partial):
            callback = callback.func
        if not isinstance(callback, tuple):
            try:
                return callback.__module__.startswith(f'{core}.')
            except AttributeError:
                return False
        obj = callback[0]()  # get object behind weakref
        if obj is None:  # object is dead
            return False
        try:
            return obj.__module__.startswith(f'{core}.')
        except AttributeError:
            return False

    def connect(
        self,
        callback: Union[Callback, CallbackRef, CallbackStr, 'EventEmitter'],
        ref: bool | str = False,
        position: Literal['first', 'last'] = 'last',
        before: str | Callback | list[str | Callback] | None = None,
        after: str | Callback | list[str | Callback] | None = None,
        until: Optional['EventEmitter'] = None,
    ):
        """Connect this emitter to a new callback.

        Parameters
        ----------
        callback : function | tuple
            *callback* may be either a callable object or a tuple
            (object, attr_name) where object.attr_name will point to a
            callable object. Note that only a weak reference to ``object``
            will be kept.
        ref : bool | str
            Reference used to identify the callback in ``before``/``after``.
            If True, the callback ref will automatically determined (see
            Notes). If False, the callback cannot be referred to by a string.
            If str, the given string will be used. Note that if ``ref``
            is not unique in ``callback_refs``, an error will be thrown.
        position : str
            If ``'first'``, the first eligible position is used (that
            meets the before and after criteria), ``'last'`` will use
            the last position.
        before : str | callback | list of str or callback | None
            List of callbacks that the current callback should precede.
            Can be None if no before-criteria should be used.
        after : str | callback | list of str or callback | None
            List of callbacks that the current callback should follow.
            Can be None if no after-criteria should be used.
        until : optional eventEmitter
            if provided, when the event `until` is emitted, `callback`
            will be disconnected from this emitter.

        Notes
        -----
        If ``ref=True``, the callback reference will be determined from:

            1. If ``callback`` is ``tuple``, the second element in the tuple.
            2. The ``__name__`` attribute.
            3. The ``__class__.__name__`` attribute.

        The current list of callback refs can be obtained using
        ``event.callback_refs``. Callbacks can be referred to by either
        their string reference (if given), or by the actual callback that
        was attached (e.g., ``(canvas, 'swap_buffers')``).

        If the specified callback is already connected, then the request is
        ignored.

        If before is None and after is None (default), the new callback will
        be added to the beginning of the callback list. Thus the
        callback that is connected _last_ will be the _first_ to receive
        events from the emitter.
        """
        callbacks = self.callbacks
        callback_refs = self.callback_refs

        old_callback = callback

        callback, pass_event = self._normalize_cb(callback)

        if callback in callbacks:
            return None

        # deal with the ref
        _ref: str | None
        if isinstance(ref, bool):
            if ref:
                if isinstance(callback, tuple):
                    _ref = callback[1]
                elif hasattr(callback, '__name__'):  # function
                    _ref = callback.__name__
                else:  # Method, or other
                    _ref = callback.__class__.__name__
            else:
                _ref = None
        elif isinstance(ref, str):
            _ref = ref
        else:
            raise TypeError(
                trans._(
                    'ref must be a bool or string',
                    deferred=True,
                )
            )
        if _ref is not None and _ref in self._callback_refs:
            raise ValueError(
                trans._('ref "{ref}" is not unique', deferred=True, ref=_ref)
            )

        # positions
        if position not in ('first', 'last'):
            raise ValueError(
                trans._(
                    'position must be "first" or "last", not {position}',
                    deferred=True,
                    position=position,
                )
            )
        core_callbacks_indexes = [
            i
            for i, c in enumerate(self._callbacks)
            if self._is_core_callback(c, 'napari')
        ]
        core_callbacks_count = (
            max(core_callbacks_indexes) + 1 if core_callbacks_indexes else 0
        )
        if self._is_core_callback(callback, 'napari'):
            callback_bounds = (0, core_callbacks_count)
        else:
            callback_bounds = (core_callbacks_count, len(callback_refs))

        # bounds: upper & lower bnds (inclusive) of possible cb locs
        bounds: list[int] = []
        for ri, criteria in enumerate((before, after)):
            if criteria is None or criteria == []:
                bounds.append(
                    callback_bounds[1] if ri == 0 else callback_bounds[0]
                )
            else:
                if not isinstance(criteria, list):
                    criteria = [criteria]
                for c in criteria:
                    count = sum(
                        c in [cn, cc]
                        for cn, cc in zip(
                            callback_refs, callbacks, strict=False
                        )
                    )
                    if count != 1:
                        raise ValueError(
                            trans._(
                                'criteria "{criteria}" is in the current callback list {count} times:\n{callback_refs}\n{callbacks}',
                                deferred=True,
                                criteria=criteria,
                                count=count,
                                callback_refs=callback_refs,
                                callbacks=callbacks,
                            )
                        )
                matches = [
                    ci
                    for ci, (cn, cc) in enumerate(
                        zip(callback_refs, callbacks, strict=False)
                    )
                    if (cc in criteria or cn in criteria)
                ]
                bounds.append(matches[0] if ri == 0 else (matches[-1] + 1))
        if bounds[0] < bounds[1]:  # i.e., "place before" < "place after"
            raise RuntimeError(
                trans._(
                    'cannot place callback before "{before}" and after "{after}" for callbacks: {callback_refs}',
                    deferred=True,
                    before=before,
                    after=after,
                    callback_refs=callback_refs,
                )
            )
        idx = bounds[1] if position == 'first' else bounds[0]  # 'last'

        # actually add the callback
        self._callbacks.insert(idx, callback)
        self._callback_refs.insert(idx, _ref)
        self._callback_pass_event.insert(idx, pass_event)

        if until is not None:
            until.connect(partial(self.disconnect, callback))

        return old_callback  # allows connect to be used as a decorator

    def disconnect(
        self, callback: Callback | CallbackRef | None | object = None
    ):
        """Disconnect a callback from this emitter.

        If no callback is specified, then *all* callbacks are removed.
        If the callback was not already connected, then the call does nothing.
        """
        if callback is None:
            self._callbacks = []
            self._callback_refs = []
            self._callback_pass_event = []
        elif isinstance(callback, (Callable, tuple)):
            callback, _pass_event = self._normalize_cb(callback)
            if callback in self._callbacks:
                idx = self._callbacks.index(callback)
                self._callbacks.pop(idx)
                self._callback_refs.pop(idx)
                self._callback_pass_event.pop(idx)
        else:
            index_list = []
            for idx, local_callback in enumerate(self._callbacks):
                if not (
                    isinstance(local_callback, Sequence)
                    and isinstance(local_callback[0], weakref.ref)
                ):
                    continue
                if (
                    local_callback[0]() is callback
                    or local_callback[0]() is None
                ):
                    index_list.append(idx)
            for idx in index_list[::-1]:
                self._callbacks.pop(idx)
                self._callback_refs.pop(idx)
                self._callback_pass_event.pop(idx)

    @staticmethod
    def _get_proper_name(callback):
        assert inspect.ismethod(callback)
        obj = callback.__self__
        if (
            not hasattr(obj, callback.__name__)
            or getattr(obj, callback.__name__) != callback
        ):
            # some decorators will alter method.__name__, so that obj.method
            # will not be equal to getattr(obj, obj.method.__name__). We check
            # for that case here and traverse to find the right method here.
            for name in dir(obj):
                meth = getattr(obj, name)
                if inspect.ismethod(meth) and meth == callback:
                    return obj, name
            raise RuntimeError(
                trans._(
                    'During bind method {callback} of object {obj} an error happen',
                    deferred=True,
                    callback=callback,
                    obj=obj,
                )
            )
        return obj, callback.__name__

    @staticmethod
    def _check_signature(fun: Callable) -> bool:
        """
        Check if function will accept event parameter
        """
        signature = inspect.signature(fun)
        parameters_list = list(signature.parameters.values())

        if sum(map(_is_pos_arg, parameters_list)) > 1:
            raise RuntimeError(
                trans._(
                    'Binning function cannot have more than one positional argument',
                    deferred=True,
                )
            )

        return any(
            x.kind
            in [
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ]
            for x in signature.parameters.values()
        )

    def _normalize_cb(self, callback) -> tuple[CallbackRef | Callback, bool]:
        # dereference methods into a (self, method_name) pair so that we can
        # make the connection without making a strong reference to the
        # instance.
        start_callback = callback
        if inspect.ismethod(callback):
            callback = self._get_proper_name(callback)
        # always use a weak ref
        if isinstance(callback, tuple) and not isinstance(
            callback[0], weakref.ref
        ):
            callback = (weakref.ref(callback[0]), *callback[1:])

        if isinstance(start_callback, Callable):
            callback = callback, self._check_signature(start_callback)
        else:
            obj = callback[0]()
            if obj is None:
                callback = callback, False
            else:
                callback_fun = getattr(obj, callback[1])
                callback = callback, self._check_signature(callback_fun)

        return callback

    def __call__(self, *args, **kwargs) -> Event:
        """__call__(**kwargs)
        Invoke all callbacks for this emitter.

        Emit a new event object, created with the given keyword
        arguments, which must match with the input arguments of the
        corresponding event class. Note that the 'type' argument is
        filled in by the emitter.

        Alternatively, the emitter can also be called with an Event
        instance as the only argument. In this case, the specified
        Event will be used rather than generating a new one. This allows
        customized Event instances to be emitted and also allows EventEmitters
        to be chained by connecting one directly to another.

        Note that the same Event instance is sent to all callbacks.
        This allows some level of communication between the callbacks
        (notably, via Event.handled) but also requires that callbacks
        be careful not to inadvertently modify the Event.
        """
        # This is a VERY highly used method; must be fast!
        blocked = self._blocked

        # create / massage event as needed
        event = self._prepare_event(*args, **kwargs)

        # Add our source to the event; remove it after all callbacks have been
        # invoked.
        event._push_source(self.source)
        self._emitting = True
        try:
            if blocked.get(None, 0) > 0:  # this is the same as self.blocked()
                self._block_counter.update([None])
                return event

            _log_event_stack(event)

            rem: list[CallbackRef] = []
            for cb, pass_event in zip(
                self._callbacks[:], self._callback_pass_event[:], strict=False
            ):
                if isinstance(cb, tuple):
                    obj = cb[0]()
                    if obj is None:
                        rem.append(cb)  # add dead weakref
                        continue
                    old_cb = cb
                    cb = getattr(obj, cb[1], None)
                    if cb is None:
                        warnings.warn(
                            trans._(
                                'Problem with function {old_cb} of {obj} connected to event {self_}',
                                deferred=True,
                                old_cb=old_cb[1],
                                obj=obj,
                                self_=self,
                            ),
                            stacklevel=2,
                            category=RuntimeWarning,
                        )
                        continue
                    cb = cast(Callback, cb)

                if blocked.get(cb, 0) > 0:
                    self._block_counter.update([cb])
                    continue

                self._invoke_callback(cb, event if pass_event else None)
                if event.blocked:
                    break

            # remove callbacks to dead objects
            for cb in rem:
                self.disconnect(cb)
        finally:
            self._emitting = False
            ps = event._pop_source()
            if ps is not self.source:
                raise RuntimeError(
                    trans._(
                        'Event source-stack mismatch.',
                        deferred=True,
                    )
                )

        return event

    def _invoke_callback(
        self, cb: Callback | Callable[[], None], event: Event | None
    ):
        try:
            if event is not None:
                cb(event)
            else:
                cb()
        except Exception as e:  # noqa: BLE001
            # dead Qt object with living python pointer. not importing Qt
            # here... but this error is consistent across backends
            if (
                isinstance(e, RuntimeError)
                and 'C++' in str(e)
                and str(e).endswith(('has been deleted', 'already deleted.'))
            ):
                self.disconnect(cb)
                return
            _handle_exception(
                self.ignore_callback_errors,
                self.print_callback_errors,
                self,
                cb_event=(cb, event),
            )

    def _prepare_event(self, *args, **kwargs) -> Event:
        # When emitting, this method is called to create or otherwise alter
        # an event before it is sent to callbacks. Subclasses may extend
        # this method to make custom modifications to the event.
        if len(args) == 1 and not kwargs and isinstance(args[0], Event):
            event: Event = args[0]
            # Ensure that the given event matches what we want to emit
            assert isinstance(event, self.event_class)
        elif not args:
            _kwargs = self.default_args.copy()
            _kwargs.update(kwargs)
            event = self.event_class(**_kwargs)
        else:
            raise ValueError(
                trans._(
                    'Event emitters can be called with an Event instance or with keyword arguments only.',
                    deferred=True,
                )
            )
        return event

    def blocked(self, callback: Callback | None = None) -> bool:
        """Return boolean indicating whether the emitter is blocked for
        the given callback.
        """
        return self._blocked.get(callback, 0) > 0

    def block(self, callback: Callback | None = None):
        """Block this emitter. Any attempts to emit an event while blocked
        will be silently ignored. If *callback* is given, then the emitter
        is only blocked for that specific callback.

        Calls to block are cumulative; the emitter must be unblocked the same
        number of times as it is blocked.
        """
        self._blocked[callback] = self._blocked.get(callback, 0) + 1

    def unblock(self, callback: Callback | None = None):
        """Unblock this emitter. See :func:`event.EventEmitter.block`.

        Note: Use of ``unblock(None)`` only reverses the effect of
        ``block(None)``; it does not unblock callbacks that were explicitly
        blocked using ``block(callback)``.
        """
        if callback not in self._blocked or self._blocked[callback] == 0:
            raise RuntimeError(
                trans._(
                    'Cannot unblock {self_} for callback {callback}; emitter was not previously blocked.',
                    deferred=True,
                    self_=self,
                    callback=callback,
                )
            )
        b = self._blocked[callback] - 1
        if b == 0 and callback is not None:
            del self._blocked[callback]
        else:
            self._blocked[callback] = b

    def blocker(self, callback: Callback | None = None):
        """Return an EventBlocker to be used in 'with' statements

        Notes
        -----
        For example, one could do::

            with emitter.blocker():
                pass  # ..do stuff; no events will be emitted..
        """
        return EventBlocker(self, callback)


class WarningEmitter(EventEmitter):
    """
    EventEmitter subclass used to allow deprecated events to be used with a
    warning message.
    """

    def __init__(
        self,
        message: str,
        category: type[Warning] = FutureWarning,
        stacklevel: int = 3,
        *args,
        **kwargs,
    ) -> None:
        self._message = message
        self._warned = False
        self._category = category
        self._stacklevel = stacklevel
        EventEmitter.__init__(self, *args, **kwargs)

    def connect(self, cb, *args, **kwargs):
        self._warn(cb)
        return EventEmitter.connect(self, cb, *args, **kwargs)

    def _invoke_callback(self, cb, event):
        self._warn(cb)
        return EventEmitter._invoke_callback(self, cb, event)

    def _warn(self, cb):
        if self._warned:
            return

        # don't warn about unimplemented connections
        if isinstance(cb, tuple) and getattr(cb[0], cb[1], None) is None:
            return

        import warnings

        warnings.warn(
            self._message, category=self._category, stacklevel=self._stacklevel
        )
        self._warned = True


class EmitterGroup(EventEmitter):
    """EmitterGroup instances manage a set of related
    :class:`EventEmitters <vispy.event.EventEmitter>`.
    Its primary purpose is to provide organization for objects
    that make use of multiple emitters and to reduce the boilerplate code
    needed to initialize those emitters with default connections.

    EmitterGroup instances are usually stored as an 'events' attribute on
    objects that use multiple emitters. For example::

         EmitterGroup  EventEmitter
                 |       |
        Canvas.events.mouse_press
        Canvas.events.resized
        Canvas.events.key_press

    EmitterGroup is also a subclass of
    :class:`EventEmitters <vispy.event.EventEmitter>`,
    allowing it to emit its own
    events. Any callback that connects directly to the EmitterGroup will
    receive *all* of the events generated by the group's emitters.

    Parameters
    ----------
    source : object
        The object that the generated events apply to.
    auto_connect : bool
        If *auto_connect* is True, then one connection will
        be made for each emitter that looks like
        :func:`emitter.connect((source, 'on_' + event_name))
        <vispy.event.EventEmitter.connect>`.
        This provides a simple mechanism for automatically connecting a large
        group of emitters to default callbacks.  By default, false.
    emitters : keyword arguments
        See the :func:`add <vispy.event.EmitterGroup.add>` method.
    """

    def __init__(
        self,
        source: Any = None,
        auto_connect: bool = False,
        **emitters: type[Event] | EventEmitter | None,
    ) -> None:
        EventEmitter.__init__(self, source)

        self.auto_connect = auto_connect
        self.auto_connect_format = 'on_%s'
        self._emitters: dict[str, EventEmitter] = {}
        # whether the sub-emitters have been connected to the group:
        self._emitters_connected: bool = False
        self.add(**emitters)  # type: ignore

    def __getattr__(self, name) -> EventEmitter:
        return object.__getattribute__(self, name)

    def __getitem__(self, name: str) -> EventEmitter:
        """
        Return the emitter assigned to the specified name.
        Note that emitters may also be retrieved as an attribute of the
        EmitterGroup.
        """
        return self._emitters[name]

    def __setitem__(
        self, name: str, emitter: type[Event] | EventEmitter | None
    ):
        """
        Alias for EmitterGroup.add(name=emitter)
        """
        self.add(**{name: emitter})  # type: ignore

    def add(
        self,
        auto_connect: bool | None = None,
        **kwargs: type[Event] | EventEmitter | None,
    ):
        """Add one or more EventEmitter instances to this emitter group.
        Each keyword argument may be specified as either an EventEmitter
        instance or an Event subclass, in which case an EventEmitter will be
        generated automatically::

            # This statement:
            group.add(mouse_press=MouseEvent,
                      mouse_release=MouseEvent)

            # ..is equivalent to this statement:
            group.add(mouse_press=EventEmitter(group.source, 'mouse_press',
                                               MouseEvent),
                      mouse_release=EventEmitter(group.source, 'mouse_press',
                                                 MouseEvent))
        """
        if auto_connect is None:
            auto_connect = self.auto_connect

        # check all names before adding anything
        for name in kwargs:
            if name in self._emitters:
                raise ValueError(
                    trans._(
                        "EmitterGroup already has an emitter named '{name}'",
                        deferred=True,
                        name=name,
                    )
                )
            if hasattr(self, name):
                raise ValueError(
                    trans._(
                        "The name '{name}' cannot be used as an emitter; it is already an attribute of EmitterGroup",
                        deferred=True,
                        name=name,
                    )
                )

        # add each emitter specified in the keyword arguments
        for name, emitter in kwargs.items():
            if emitter is None:
                emitter = Event

            if inspect.isclass(emitter) and issubclass(emitter, Event):  # type: ignore
                emitter = EventEmitter(
                    source=self.source,
                    type_name=name,
                    event_class=emitter,  # type: ignore
                )
            elif not isinstance(emitter, EventEmitter):
                raise RuntimeError(
                    trans._(
                        'Emitter must be specified as either an EventEmitter instance or Event subclass. (got {name}={emitter})',
                        deferred=True,
                        name=name,
                        emitter=emitter,
                    )
                )

            # give this emitter the same source as the group.
            emitter.source = self.source

            setattr(self, name, emitter)  # this is a bummer for typing.
            self._emitters[name] = emitter

            if (
                auto_connect
                and self.source is not None
                and hasattr(self.source, self.auto_connect_format % name)
            ):
                emitter.connect((self.source, self.auto_connect_format % name))

            # If emitters are connected to the group already, then this one
            # should be connected as well.
            if self._emitters_connected:
                emitter.connect(self)

    @property
    def emitters(self) -> dict[str, EventEmitter]:
        """List of current emitters in this group."""
        return self._emitters

    def __iter__(self) -> Iterator[str]:
        """
        Iterates over the names of emitters in this group.
        """
        yield from self._emitters

    def block_all(self):
        """
        Block all emitters in this group by increase counter of semaphores for each event emitter
        """
        self.block()
        for em in self._emitters.values():
            em.block()

    def unblock_all(self):
        """
        Unblock all emitters in this group, by decrease counter of semaphores for each event emitter.
        if block is called twice and unblock is called once, then events will be still blocked.
        See `Semaphore (programming) <https://en.wikipedia.org/wiki/Semaphore_(programming)>`__.
        """
        self.unblock()
        for em in self._emitters.values():
            em.unblock()

    def connect(
        self,
        callback: Union[Callback, CallbackRef, EventEmitter, 'EmitterGroup'],
        ref: bool | str = False,
        position: Literal['first', 'last'] = 'first',
        before: str | Callback | list[str | Callback] | None = None,
        after: str | Callback | list[str | Callback] | None = None,
    ):
        """Connect the callback to the event group. The callback will receive
        events from *all* of the emitters in the group.

        See :func:`EventEmitter.connect() <vispy.event.EventEmitter.connect>`
        for arguments.
        """
        self._connect_emitters(True)
        return EventEmitter.connect(
            self, callback, ref, position, before, after
        )

    def disconnect(self, callback: Callback | None = None):
        """Disconnect the callback from this group. See
        :func:`connect() <vispy.event.EmitterGroup.connect>` and
        :func:`EventEmitter.connect() <vispy.event.EventEmitter.connect>` for
        more information.
        """
        ret = EventEmitter.disconnect(self, callback)
        if len(self._callbacks) == 0:
            self._connect_emitters(False)
        return ret

    def _connect_emitters(self, connect):
        # Connect/disconnect all sub-emitters from the group. This allows the
        # group to emit an event whenever _any_ of the sub-emitters emit,
        # while simultaneously eliminating the overhead if nobody is listening.
        if connect:
            for emitter in self:
                if not isinstance(self[emitter], WarningEmitter):
                    self[emitter].connect(self)
        else:
            for emitter in self:
                self[emitter].disconnect(self)

        self._emitters_connected = connect

    @property
    def ignore_callback_errors(self):
        return super().ignore_callback_errors

    @ignore_callback_errors.setter
    def ignore_callback_errors(self, ignore):
        EventEmitter.ignore_callback_errors.fset(self, ignore)
        for emitter in self._emitters.values():
            if isinstance(emitter, EventEmitter):
                emitter.ignore_callback_errors = ignore
            elif isinstance(emitter, EmitterGroup):
                emitter.ignore_callback_errors_all(ignore)

    def blocker_all(self) -> 'EventBlockerAll':
        """Return an EventBlockerAll to be used in 'with' statements

        Notes
        -----
        For example, one could do::

            with emitter.blocker_all():
                pass  # ..do stuff; no events will be emitted..
        """
        return EventBlockerAll(self)


class EventBlocker:
    """Represents a block for an EventEmitter to be used in a context
    manager (i.e. 'with' statement).
    """

    def __init__(self, target: EventEmitter, callback=None) -> None:
        self.target = target
        self.callback = callback
        self._base_count = target._block_counter.get(callback, 0)

    @property
    def count(self):
        n_blocked = self.target._block_counter.get(self.callback, 0)
        return n_blocked - self._base_count

    def __enter__(self):
        self.target.block(self.callback)
        return self

    def __exit__(self, *args):
        self.target.unblock(self.callback)


class EventBlockerAll:
    """Represents a block_all for an EmitterGroup to be used in a context
    manager (i.e. 'with' statement).
    """

    def __init__(self, target: EmitterGroup) -> None:
        self.target = target

    def __enter__(self):
        self.target.block_all()

    def __exit__(self, *args):
        self.target.unblock_all()


def _is_pos_arg(param: inspect.Parameter):
    """
    Check if param is positional or named and has no default parameter.
    """
    return (
        param.kind
        in [
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ]
        and param.default == inspect.Parameter.empty
    )


with contextlib.suppress(ModuleNotFoundError):
    # this could move somewhere higher up in napari imports ... but where?
    __import__('dotenv').load_dotenv()


def _noop(*a, **k):
    pass


_log_event_stack = _noop


def set_event_tracing_enabled(enabled=True, cfg=None):
    global _log_event_stack
    if enabled:
        from napari.utils.events.debugging import log_event_stack

        if cfg is not None:
            _log_event_stack = partial(log_event_stack, cfg=cfg)
        else:
            _log_event_stack = log_event_stack
    else:
        _log_event_stack = _noop


if os.getenv('NAPARI_DEBUG_EVENTS', '').lower() in ('1', 'true'):
    set_event_tracing_enabled(True)
