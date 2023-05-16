"""Nestable MutableSequence that emits events when altered.

see module docstring of evented_list.py for more details
"""
from __future__ import annotations

import contextlib
import logging
from collections import defaultdict
from typing import (
    DefaultDict,
    Generator,
    Iterable,
    MutableSequence,
    NewType,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

from napari.utils.events.containers._evented_list import EventedList, Index
from napari.utils.events.event import Event
from napari.utils.translations import trans

logger = logging.getLogger(__name__)

NestedIndex = Tuple[Index, ...]
MaybeNestedIndex = Union[Index, NestedIndex]
ParentIndex = NewType('ParentIndex', Tuple[int, ...])
_T = TypeVar("_T")


def ensure_tuple_index(index: MaybeNestedIndex) -> NestedIndex:
    """Return index as a tuple of ints or slices.

    Parameters
    ----------
    index : Tuple[Union[int, slice], ...] or int or slice
        An index as an int, tuple, or slice

    Returns
    -------
    NestedIndex
        The index, guaranteed to be a tuple.

    Raises
    ------
    TypeError
        If the input ``index`` is not an ``int``, ``slice``, or ``tuple``.
    """
    if isinstance(index, (slice, int)):
        return (index,)  # single integer inserts to self
    if isinstance(index, tuple):
        return index

    raise TypeError(
        trans._(
            "Invalid nested index: {index}. Must be an int or tuple",
            deferred=True,
            index=index,
        )
    )


def split_nested_index(index: MaybeNestedIndex) -> tuple[ParentIndex, Index]:
    """Given a nested index, return (nested_parent_index, row).

    Parameters
    ----------
    index : MaybeNestedIndex
        An index as an int, tuple, or slice

    Returns
    -------
    Tuple[NestedIndex, Index]
        A tuple of ``parent_index``, ``row``

    Raises
    ------
    ValueError
        If any of the items in the returned ParentIndex tuple are not ``int``.

    Examples
    --------
    >>> split_nested_index((1, 2, 3, 4))
    ((1, 2, 3), 4)
    >>> split_nested_index(1)
    ((), 1)
    >>> split_nested_index(())
    ((), -1)
    """
    index = ensure_tuple_index(index)
    if index:
        *first, last = index
        if any(not isinstance(p, int) for p in first):
            raise ValueError(
                trans._(
                    'The parent index must be a tuple of int',
                    deferred=True,
                )
            )
        return cast(ParentIndex, tuple(first)), last
    return ParentIndex(()), -1  # empty tuple appends to self


class NestableEventedList(EventedList[_T]):
    """Nestable Mutable Sequence that emits recursive events when altered.

    ``NestableEventedList`` instances can be indexed with a ``tuple`` of
    ``int`` (e.g. ``mylist[0, 2, 1]``) to retrieve nested child objects.

    A key property of this class is that when new mutable sequences are added
    to the list, they are themselves converted to a ``NestableEventedList``,
    and all of the ``EventEmitter`` objects in the child are connect to the
    parent object's ``_reemit_child_event`` method (assuming the child has
    an attribute called ``events`` that is an instance of ``EmitterGroup``).
    When ``_reemit_child_event`` receives an event from a child object, it
    remits the event, but changes any ``index`` keys in the event to a
    ``NestedIndex`` (a tuple of ``int``) such that indices emitted by any given
    ``NestableEventedList`` are always relative to itself.


    Parameters
    ----------
    data : iterable, optional
        Elements to initialize the list with. by default None.
    basetype : type or sequence of types, optional
        Type of the elements in the list.
    lookup : dict of Type[L] : function(object) -> L
        Mapping between a type, and a function that converts items in the list
        to that type.

    Events
    ------
    types used:
        Index = Union[int, Tuple[int, ...]]

    inserting (index: Index)
        emitted before an item is inserted at ``index``
    inserted (index: Index, value: T)
        emitted after ``value`` is inserted at ``index``
    removing (index: Index)
        emitted before an item is removed at ``index``
    removed (index: Index, value: T)
        emitted after ``value`` is removed at ``index``
    moving (index: Index, new_index: Index)
        emitted before an item is moved from ``index`` to ``new_index``
    moved (index: Index, new_index: Index, value: T)
        emitted after ``value`` is moved from ``index`` to ``new_index``
    changed (index: Index, old_value: T, value: T)
        emitted when ``index`` is set from ``old_value`` to ``value``
    changed <OVERLOAD> (index: slice, old_value: list[_T], value: list[_T])
        emitted when slice ``index`` is set from ``old_value`` to ``value``
    reordered (value: self)
        emitted when the list is reordered (eg. moved/reversed).
    """

    # WAIT!! ... Read the ._list module docs before reimplement these classes
    # def append(self, item): ...
    # def clear(self): ...
    # def pop(self, index=-1): ...
    # def extend(self, value: Iterable[_T]): ...
    # def remove(self, value: T): ...

    @overload  # type: ignore
    def __getitem__(self, key: int) -> Union[_T, NestableEventedList[_T]]:
        ...  # pragma: no cover

    @overload
    def __getitem__(self, key: ParentIndex) -> NestableEventedList[_T]:
        ...  # pragma: no cover

    @overload
    def __getitem__(self, key: slice) -> NestableEventedList[_T]:
        ...  # pragma: no cover

    @overload
    def __getitem__(
        self, key: NestedIndex
    ) -> Union[_T, NestableEventedList[_T]]:
        ...  # pragma: no cover

    def __getitem__(self, key: MaybeNestedIndex):
        if isinstance(key, tuple):
            item: NestableEventedList[_T] = self
            for idx in key:
                if not isinstance(item, MutableSequence):
                    raise IndexError(f'index out of range: {key}')
                item = item[idx]
            return item
        return super().__getitem__(key)

    @overload
    def __setitem__(self, key: Union[int, NestedIndex], value: _T):
        ...  # pragma: no cover

    @overload
    def __setitem__(self, key: slice, value: Iterable[_T]):
        ...  # pragma: no cover

    def __setitem__(self, key: MaybeNestedIndex, value):
        # NOTE: if we check isinstance(..., MutableList), then we'll actually
        # clobber object of specialized classes being inserted into the list
        # (for instance, subclasses of NestableEventedList)
        # this check is more conservative, but will miss some "nestable" things
        if isinstance(value, list):
            value = self.__class__(value)
        if isinstance(key, tuple):
            parent_i, index = split_nested_index(key)
            self[parent_i].__setitem__(index, value)
            return
        self._connect_child_emitters(value)
        super().__setitem__(key, value)

    def _delitem_indices(
        self, key: MaybeNestedIndex
    ) -> Iterable[tuple[EventedList[_T], int]]:
        if isinstance(key, tuple):
            parent_i, index = split_nested_index(key)
            if isinstance(index, slice):
                indices = sorted(
                    range(*index.indices(len(parent_i))), reverse=True
                )
            else:
                indices = [index]
            return [(self[parent_i], i) for i in indices]
        return super()._delitem_indices(key)

    def insert(self, index: int, value: _T):
        """Insert object before index."""
        # this is delicate, we want to preserve the evented list when nesting
        # but there is a high risk here of clobbering attributes of a special
        # child class, so checking for self.__class__ helps
        if isinstance(value, list) and not isinstance(value, self.__class__):
            value = self.__newlike__(value)
        super().insert(index, value)

    def _reemit_child_event(self, event: Event):
        """An item in the list emitted an event.  Re-emit with index"""
        if hasattr(event, 'index'):
            # This event is coming from a nested List...
            # update the index as a nested index.
            ei = (self.index(event.source), *ensure_tuple_index(event.index))
            for attr in ('index', 'new_index'):
                if hasattr(event, attr):
                    setattr(event, attr, ei)

        # if the starting event was from a nestable envented list, we can
        # use the same event type here (e.g: removed, inserted)
        if isinstance(event.source, NestableEventedList):
            emitter = getattr(self.events, event.type, self.events)
        else:
            emitter = self.events

        # same as normal evented_list, but now we need to account for the
        # potentially different emitter
        if not hasattr(event, 'index'):
            with contextlib.suppress(ValueError):
                event.index = self.index(event.source)

        emitter(event)

    def _non_negative_index(
        self, parent_index: ParentIndex, dest_index: Index
    ) -> Index:
        """Make sure dest_index is a positive index inside parent_index."""
        destination_group = self[parent_index]
        # not handling slice indexes
        if isinstance(dest_index, int) and dest_index < 0:
            dest_index += len(destination_group) + 1
        return dest_index

    def _move_plan(
        self, sources: Iterable[MaybeNestedIndex], dest_index: NestedIndex
    ) -> Generator[tuple[NestedIndex, NestedIndex], None, None]:
        """Prepared indices for a complicated nested multi-move.

        Given a set of possibly-nested ``sources`` from anywhere in the tree,
        and a single ``dest_index``, this function computes and yields
        ``(from_index, to_index)`` tuples that can be used sequentially in
        single move operations.  It keeps track of what has moved where and
        updates the source and destination indices to reflect the model at each
        point in the process.

        This is useful for a drag-drop operation with a QtModel/View.

        Parameters
        ----------
        sources : Iterable[tuple[int, ...]]
            An iterable of tuple[int] that should be moved to ``dest_index``.
            (Note: currently, the order of ``sources`` will NOT be maintained.)
        dest_index : Tuple[int]
            The destination for sources.

        Yields
        ------
        Generator[tuple[int, ...], None, None]
            [description]

        Raises
        ------
        ValueError
            If any source terminal or the destination terminal index is a slice
        IndexError
            If any of the sources are the root object: ``()``.
        NotImplementedError
            If a slice is provided in the middle of a source index.
        """

        dest_par, dest_i = split_nested_index(dest_index)
        if isinstance(dest_i, slice):
            raise TypeError(
                trans._(
                    "Destination index may not be a slice",
                    deferred=True,
                )
            )
        dest_i = cast(int, self._non_negative_index(dest_par, dest_i))

        # need to update indices as we pop, so we keep track of the indices
        # we have previously popped
        popped: DefaultDict[NestedIndex, list[int]] = defaultdict(list)
        dumped: list[int] = []

        # we iterate indices from the end first, so pop() always works
        for idx in sorted(sources, reverse=True):
            if isinstance(idx, (int, slice)):
                idx = (idx,)
            if idx == ():
                raise IndexError(
                    trans._(
                        "Group cannot move itself",
                        deferred=True,
                    )
                )

            # i.e. we need to increase the (src_par, ...) by 1 for each time
            # we have previously inserted items in front of the (src_par, ...)
            _parlen = len(dest_par)
            if len(idx) > _parlen:
                _idx: list[Index] = list(idx)
                if isinstance(_idx[_parlen], slice):
                    raise NotImplementedError(
                        trans._(
                            "Can't yet deal with slice source indices in multimove",
                            deferred=True,
                        )
                    )
                _idx[_parlen] += sum(x <= _idx[_parlen] for x in dumped)
                idx = tuple(_idx)

            src_par, src_i = split_nested_index(idx)
            if isinstance(src_i, slice):
                raise TypeError(
                    trans._(
                        "Terminal source index may not be a slice",
                        deferred=True,
                    )
                )

            if src_i < 0:
                src_i += len(self[src_par])

            # we need to decrement the src_i by 1 for each time we have
            # previously pulled items out from in front of the src_i
            src_i -= sum(x <= src_i for x in popped.get(src_par, []))

            # we need to decrement the dest_i by 1 for each time we have
            # previously pulled items out from in front of the dest_i
            ddec = sum(x <= dest_i for x in popped.get(dest_par, []))

            # skip noop
            if src_par == dest_par and src_i == dest_i - ddec:
                continue

            yield (*src_par, src_i), (*dest_par, dest_i - ddec)
            popped[src_par].append(src_i)
            dumped.append(dest_i - ddec)

    def move(
        self,
        src_index: Union[int, NestedIndex],
        dest_index: Union[int, NestedIndex] = (0,),
    ) -> bool:
        """Move a single item from ``src_index`` to ``dest_index``.

        Parameters
        ----------
        src_index : Union[int, NestedIndex]
            The index of the object to move
        dest_index : Union[int, NestedIndex], optional
            The destination.  Object will be inserted before ``dest_index.``,
            by default, will insert at the front of the root list.

        Returns
        -------
        bool
            Whether the operation completed successfully

        Raises
        ------
        ValueError
            If the terminal source is a slice, or if the source is this root
            object
        """
        logger.debug(
            "move(src_index=%s, dest_index=%s)",
            src_index,
            dest_index,
        )
        src_par_i, src_i = split_nested_index(src_index)
        dest_par_i, dest_i = split_nested_index(dest_index)
        dest_i = self._non_negative_index(dest_par_i, dest_i)
        dest_index = (*dest_par_i, dest_i)

        if isinstance(src_i, slice):
            raise TypeError(
                trans._(
                    "Terminal source index may not be a slice",
                    deferred=True,
                )
            )

        if isinstance(dest_i, slice):
            raise TypeError(
                trans._(
                    "Destination index may not be a slice",
                    deferred=True,
                )
            )

        if src_i == ():
            raise ValueError(
                trans._(
                    "Group cannot move itself",
                    deferred=True,
                )
            )

        if src_par_i == dest_par_i and isinstance(dest_i, int):
            if dest_i > src_i:
                dest_i -= 1
            if src_i == dest_i:
                return False

        self.events.moving(index=src_index, new_index=dest_index)

        dest_par = self[dest_par_i]  # grab this before popping src_i

        with self.events.blocker_all():
            value = self[src_par_i].pop(src_i)
            dest_par.insert(dest_i, value)

        self.events.moved(index=src_index, new_index=dest_index, value=value)
        self.events.reordered(value=self)
        return True

    def _type_check(self, e) -> _T:
        if isinstance(e, list) and not isinstance(e, self.__class__):
            return self.__newlike__(e)
        if self._basetypes:
            _types = (*tuple(self._basetypes), NestableEventedList)
            if not isinstance(e, _types):
                raise TypeError(
                    trans._(
                        'Cannot add object with type {dtype!r} to TypedList expecting type {types_!r}',
                        deferred=True,
                        dtype=type(e),
                        types_=_types,
                    )
                )
        return e

    def _iter_indices(self, start=0, stop=None, root=()):
        """Iter indices from start to stop.

        Depth first traversal of the tree
        """
        for i, item in enumerate(self[start:stop]):
            yield (*root, i) if root else i
            if isinstance(item, NestableEventedList):
                yield from item._iter_indices(root=(*root, i))

    def has_index(self, index: Union[int, Tuple[int, ...]]) -> bool:
        """Return true if `index` is valid for this nestable list."""
        if isinstance(index, int):
            return -len(self) <= index < len(self)
        if isinstance(index, tuple):
            try:
                self[index]
            except IndexError:
                return False
            else:
                return True
        raise TypeError(f"Not supported index type {type(index)}")
