import warnings
from typing import Any, TypeVar

from napari.utils.events.containers._evented_list import EventedList
from napari.utils.events.containers._nested_list import NestableEventedList
from napari.utils.events.containers._selection import Selectable
from napari.utils.translations import trans

_T = TypeVar('_T')


class SelectableEventedList(Selectable[_T], EventedList[_T]):
    """List model that also supports selection.

    Events
    ------
    inserting (index: int)
        emitted before an item is inserted at ``index``
    inserted (index: int, value: T)
        emitted after ``value`` is inserted at ``index``
    removing (index: int)
        emitted before an item is removed at ``index``
    removed (index: int, value: T)
        emitted after ``value`` is removed at ``index``
    moving (index: int, new_index: int)
        emitted before an item is moved from ``index`` to ``new_index``
    moved (index: int, new_index: int, value: T)
        emitted after ``value`` is moved from ``index`` to ``new_index``
    changed (index: int, old_value: T, value: T)
        emitted when item at ``index`` is changed from ``old_value`` to ``value``
    changed <OVERLOAD> (index: slice, old_value: List[_T], value: List[_T])
        emitted when item at ``index`` is changed from ``old_value`` to ``value``
    reordered (value: self)
        emitted when the list is reordered (eg. moved/reversed).

    selection.changed (added: Set[_T], removed: Set[_T])
        Emitted when the set changes, includes item(s) that have been added
        and/or removed from the set.
    selection.active (value: _T)
        emitted when the current item has changed.
    selection._current (value: _T)
        emitted when the current item has changed. (Private event)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._activate_on_insert = True
        super().__init__(*args, **kwargs)
        # bound/unbound methods are ambiguous for mypy so we need to ignore
        # https://mypy.readthedocs.io/en/stable/error_code_list.html?highlight=method-assign#check-that-assignment-target-is-not-a-method-method-assign
        self.selection._pre_add_hook = self._preselect_hook  # type: ignore[method-assign]

    def _preselect_hook(self, value: _T) -> _T:
        """Called before adding an item to the selection."""
        if value not in self:
            raise ValueError(
                trans._(
                    'Cannot select item that is not in list: {value!r}',
                    deferred=True,
                    value=value,
                )
            )
        return value

    def _process_delete_item(self, item: _T) -> None:
        self.selection.discard(item)

    def insert(self, index: int, value: _T) -> None:
        super().insert(index, value)
        if self._activate_on_insert:
            # Make layer selected and unselect all others
            self.selection.active = value

    def select_all(self) -> None:
        """Select all items in the list."""
        self.selection.update(self)

    def remove_selected(self) -> None:
        """Remove selected items from list."""
        idx = 0
        initial_len = len(self)
        for i in list(self.selection):
            idx = self.index(i)
            self.remove(i)
        if initial_len > len(self):
            if isinstance(idx, int):
                new = max(0, (idx - 1))
                do_add = len(self) > new
            else:
                *root, _idx = idx
                new = (*tuple(root), _idx - 1) if _idx >= 1 else tuple(root)
                do_add = len(self) > new[0]
            if do_add:
                self.selection.add(self[new])

    def move_selected(self, index: int, insert: int) -> None:
        """Reorder list by moving the item at index and inserting it
        at the insert index. If additional items are selected these will
        get inserted at the insert index too. This allows for rearranging
        the list based on dragging and dropping a selection of items, where
        index is the index of the primary item being dragged, and insert is
        the index of the drop location, and the selection indicates if
        multiple items are being dragged. If the moved layer is not selected
        select it.

        This method is deprecated. Please use layers.move_multiple
        with layers.selection instead.

        Parameters
        ----------
        index : int
            Index of primary item to be moved
        insert : int
            Index that item(s) will be inserted at
        """
        # this is just here for now to support the old layerlist API
        warnings.warn(
            trans._(
                'move_selected is deprecated since 0.4.16. Please use layers.move_multiple with layers.selection instead.',
                deferred=True,
            ),
            FutureWarning,
            stacklevel=2,
        )
        if self[index] not in self.selection:
            self.selection.select_only(self[index])
            moving = [index]
        else:
            moving = [i for i, x in enumerate(self) if x in self.selection]
        offset = insert >= index
        self.move_multiple(moving, insert + offset)

    def select_next(self, step: int = 1, shift: bool = False) -> None:
        """Selects next item from list."""
        if self.selection and self.selection._current:
            idx = self.index(self.selection._current) + step
            if len(self) > idx >= 0:
                next_layer = self[idx]
                if shift:
                    if next_layer in self.selection:
                        self.selection.remove(self.selection._current)
                        self.selection._current = next_layer
                    else:
                        self.selection.add(next_layer)
                        self.selection._current = next_layer
                else:
                    self.selection.active = next_layer
        elif len(self) > 0:
            self.selection.active = self[-1 if step > 0 else 0]

    def select_previous(self, shift: bool = False) -> None:
        """Selects previous item from list."""
        self.select_next(-1, shift=shift)


class SelectableNestableEventedList(
    SelectableEventedList[_T], NestableEventedList[_T]
):
    pass
