from __future__ import annotations

from collections.abc import MutableSequence
from typing import TYPE_CHECKING, Any, Generic, Tuple, TypeVar, Union

from qtpy.QtCore import QAbstractItemModel, QModelIndex, Qt

from napari.utils.events import disconnect_events
from napari.utils.events.containers import SelectableEventedList
from napari.utils.translations import trans

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget


ItemType = TypeVar("ItemType")

ItemRole = Qt.UserRole
SortRole = Qt.UserRole + 1
ThumbnailRole = Qt.UserRole + 2

_BASE_FLAGS = (
    Qt.ItemFlag.ItemIsSelectable
    | Qt.ItemFlag.ItemIsEditable
    | Qt.ItemFlag.ItemIsUserCheckable
    | Qt.ItemFlag.ItemIsDragEnabled
    | Qt.ItemFlag.ItemIsEnabled
)


class _BaseEventedItemModel(QAbstractItemModel, Generic[ItemType]):
    """A QAbstractItemModel desigend to work with `SelectableEventedList`.

    :class:`~napari.utils.events.SelectableEventedList` is our pure python
    model of a mutable sequence that supports the concept of "currently
    selected/active items".  It emits events when the list is altered (e.g.,
    by appending, inserting, removing items), or when the selection model is
    altered.

    This class is an adapter between that interface and Qt's
    `QAbstractItemModel` interface.  It allows python users to interact with
    the list in the "usual" python ways, updating any Qt Views that may
    be connected, and also updates the python list object if any GUI events
    occur in the view.

    For a "plain" (flat) list, use the
    :class:`napari._qt.containers.QtListModel` subclass.
    For a nested list-of-lists using the Group/Node classes, use the
    :class:`napari._qt.containers.QtNodeTreeModel` subclass.

    For convenience, the :func:`napari._qt.containers.create_model` factory
    function will return the appropriate `_BaseEventedItemModel` instance given
    a python `EventedList` object.

    .. note::

        In most cases, if you want a "GUI widget" to go along with an
        ``EventedList`` object, it will not be necessary to instantiate the
        ``EventedItemModel`` directly.  Instead, use one of the
        :class:`napari._qt.containers.QtListView` or
        :class:`napari._qt.containers.QtNodeTreeView` views, or the
        :func:`napari._qt.containers.create_view` factory function.

    Key concepts and references:
        - Qt `Model/View Programming
          <https://doc.qt.io/qt-5/model-view-programming.html>`_
        - Qt `Model Subclassing Reference
          <https://doc.qt.io/qt-5/model-view-programming.html#model-subclassing-reference>`_
        - `Model Index <https://doc.qt.io/qt-5/model-view-programming.html#model-indexes>`_
        - `Simple Tree Model Example
          <https://doc.qt.io/qt-5/qtwidgets-itemviews-simpletreemodel-example.html>`_
    """

    _root: SelectableEventedList[ItemType]

    # ########## Reimplemented Public Qt Functions ##################

    def __init__(
        self, root: SelectableEventedList[ItemType], parent: QWidget = None
    ) -> None:
        super().__init__(parent=parent)
        self.setRoot(root)

    def parent(self, index):
        """Return the parent of the model item with the given ``index``.

        (The parent in a basic list is always the root, Tree models will need
        to reimplement)
        """
        return QModelIndex()

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> Any:
        """Returns data stored under `role` for the item at `index`.

        A given `QModelIndex` can store multiple types of data, each
        with its own "ItemDataRole".  ItemType-specific subclasses will likely
        want to customize this method (and likely `setData` as well) for
        different data roles.

        see: https://doc.qt.io/qt-5/qt.html#ItemDataRole-enum

        """
        if role == Qt.DisplayRole:
            return str(self.getItem(index))
        if role == ItemRole:
            return self.getItem(index)
        if role == SortRole:
            return index.row()
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """Returns the item flags for the given `index`.

        This describes the properties of a given item in the model.  We set
        them to be editable, checkable, dragable, droppable, etc...
        If index is not a list, we additionally set `Qt.ItemNeverHasChildren`
        (for optimization). Editable models must return a value containing
        `Qt.ItemIsEditable`.

        See Qt.ItemFlags https://doc.qt.io/qt-5/qt.html#ItemFlag-enum
        """
        if not index.isValid() or index.model() is not self:
            # we allow drops outside the items
            return Qt.ItemFlag.ItemIsDropEnabled
        if isinstance(self.getItem(index), MutableSequence):
            return _BASE_FLAGS | Qt.ItemFlag.ItemIsDropEnabled
        return _BASE_FLAGS | Qt.ItemFlag.ItemNeverHasChildren

    def columnCount(self, parent: QModelIndex) -> int:
        """Return the number of columns for the children of the given `parent`.

        In a list view, and most tree views, the number of columns is always 1.
        """
        return 1

    def rowCount(self, parent: QModelIndex = None) -> int:
        """Returns the number of rows under the given parent.

        When the parent is valid it means that rowCount is returning the number
        of children of parent.
        """
        if parent is None:
            parent = QModelIndex()
        try:
            return len(self.getItem(parent))
        except TypeError:
            return 0

    def index(
        self, row: int, column: int = 0, parent: QModelIndex = None
    ) -> QModelIndex:
        """Return a QModelIndex for item at `row`, `column` and `parent`."""

        # NOTE: the use of `self.createIndex(row, col, object)`` will create a
        # model index that stores a pointer to the object, which can be
        # retrieved later with index.internalPointer().  That's convenient and
        # performant, and very important tree structures, but it causes a bug
        # if integers (or perhaps values that get garbage collected?) are in
        # the list, because `createIndex` is an overloaded function and
        # `self.createIndex(row, col, <int>)` will assume that the third
        # argument *is* the id of the object (not the object itself).  This
        # will then cause a segfault if `index.internalPointer()` is used
        # later.

        # so we need to either:
        #   1. refuse store integers in this model
        #   2. never store the object (and incur the penalty of
        #      self.getItem(idx) each time you want to get the value of an idx)
        #   3. Have special treatment when we encounter integers in the model
        #   4. Wrap every object in *another* object (which is basically what
        #      Qt does with QAbstractItem)... ugh.
        #
        # Unfortunately, all of those come at a cost... as this is a very
        # frequently called function :/

        if parent is None:
            parent = QModelIndex()

        return (
            self.createIndex(row, column, self.getItem(parent)[row])
            if self.hasIndex(row, column, parent)
            else QModelIndex()  # instead of index error, Qt wants null index
        )

    def supportedDropActions(self) -> Qt.DropActions:
        """Returns the drop actions supported by this model.

        The default implementation returns `Qt.CopyAction`. We re-implement to
        support only `Qt.MoveAction`. See also dropMimeData(), which must
        handle each supported drop action type.
        """
        return Qt.MoveAction

    # ###### Non-Qt methods added for SelectableEventedList Model ############

    def setRoot(self, root: SelectableEventedList[ItemType]):
        """Call during __init__, to set the python model and connections"""
        if not isinstance(root, SelectableEventedList):
            raise TypeError(
                trans._(
                    "root must be an instance of {class_name}",
                    deferred=True,
                    class_name=SelectableEventedList,
                )
            )
        current_root = getattr(self, "_root", None)
        if root is current_root:
            return

        if current_root is not None:
            # we're changing roots... disconnect previous root
            disconnect_events(self._root.events, self)

        self._root = root
        self._root.events.removing.connect(self._on_begin_removing)
        self._root.events.removed.connect(self._on_end_remove)
        self._root.events.inserting.connect(self._on_begin_inserting)
        self._root.events.inserted.connect(self._on_end_insert)
        self._root.events.moving.connect(self._on_begin_moving)
        self._root.events.moved.connect(self._on_end_move)
        self._root.events.connect(self._process_event)

    def _split_nested_index(
        self, nested_index: Union[int, Tuple[int, ...]]
    ) -> Tuple[QModelIndex, int]:
        """Return (parent_index, row) for a given index."""
        if isinstance(nested_index, int):
            return QModelIndex(), nested_index
        # Tuple indexes are used in NestableEventedList, so we support them
        # here so that subclasses needn't reimplmenet our _on_begin_* methods
        par = QModelIndex()
        *_p, idx = nested_index
        for i in _p:
            par = self.index(i, 0, par)
        return par, idx

    def _on_begin_inserting(self, event):
        """Begins a row insertion operation.

        See Qt documentation:
        https://doc.qt.io/qt-5/qabstractitemmodel.html#beginInsertRows
        """
        par, idx = self._split_nested_index(event.index)
        self.beginInsertRows(par, idx, idx)

    def _on_end_insert(self):
        """Must be called after insert operation to update model."""
        self.endInsertRows()

    def _on_begin_removing(self, event):
        """Begins a row removal operation.

        See Qt documentation:
        https://doc.qt.io/qt-5/qabstractitemmodel.html#beginRemoveRows
        """
        par, idx = self._split_nested_index(event.index)
        self.beginRemoveRows(par, idx, idx)

    def _on_end_remove(self):
        """Must be called after remove operation to update model."""
        self.endRemoveRows()

    def _on_begin_moving(self, event):
        """Begins a row move operation.

        See Qt documentation:
        https://doc.qt.io/qt-5/qabstractitemmodel.html#beginMoveRows
        """
        src_par, src_idx = self._split_nested_index(event.index)
        dest_par, dest_idx = self._split_nested_index(event.new_index)

        self.beginMoveRows(src_par, src_idx, src_idx, dest_par, dest_idx)

    def _on_end_move(self):
        """Must be called after move operation to update model."""
        self.endMoveRows()

    def getItem(self, index: QModelIndex) -> ItemType:
        """Return python object for a given `QModelIndex`.

        An invalid `QModelIndex` will return the root object.
        """
        return self._root[index.row()] if index.isValid() else self._root

    def _process_event(self, event):
        # for subclasses to handle ItemType-specific data
        pass
