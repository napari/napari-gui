import logging
import pickle
from typing import List, Optional, Tuple, TypeVar

from qtpy.QtCore import QMimeData, QModelIndex, Qt

from napari._qt.containers._base_item_model import (
    SortRole,
    _BaseEventedItemModel,
)
from napari.utils.translations import trans
from napari.utils.tree import Group, Node

logger = logging.getLogger(__name__)
NodeType = TypeVar("NodeType", bound=Node)
NodeMIMEType = "application/x-tree-node"


class QtNodeTreeModel(_BaseEventedItemModel[NodeType]):
    """A QItemModel for a tree of ``Node`` and ``Group`` objects.

    Designed to work with :class:`napari.utils.tree.Group` and
    :class:`~napari._qt.containers.QtNodeTreeView`.

    See docstring of :class:`_BaseEventedItemModel` and
    :class:`~napari._qt.containers.QtNodeTreeView` for additional background.
    """

    _root: Group[NodeType]

    # ########## Reimplemented Public Qt Functions ##################

    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        """Return data stored under ``role`` for the item at ``index``.

        A given class:`QModelIndex` can store multiple types of data, each with
        its own "ItemDataRole".
        """
        item = self.getItem(index)
        if role == Qt.ItemDataRole.DisplayRole:
            return item._node_name()
        if role == Qt.ItemDataRole.UserRole:
            return self.getItem(index)
        if role == SortRole:
            return index.row()
        return None

    def index(
        self, row: int, column: int = 0, parent: QModelIndex = None
    ) -> QModelIndex:
        """Return a QModelIndex for item at `row`, `column` and `parent`."""

        # NOTE: self.createIndex(row, col, object) will create a model index
        # that *stores* a pointer to the object, which can be retrieved later
        # with index.internalPointer().  That's convenient and performant, but
        # it comes with a bug if integers are in the list, because
        # `createIndex` is overloaded and `self.createIndex(row, col, <int>)`
        # will assume that the third argument *is* the id of the object (not
        # the object itself).  This will then cause a segfault if
        # `index.internalPointer()` is used later.

        # XXX: discuss
        # so we need to either:
        #   1. refuse store integers in this model
        #   2. never store the object (and incur the penalty of
        #      self.getItem(idx) each time you want to get the value of an idx)
        #   3. Have special treatment when we encounter integers in the model
        if parent is None:
            parent = QModelIndex()
        return (
            self.createIndex(row, column, self.getItem(parent)[row])
            if self.hasIndex(row, column, parent)
            else QModelIndex()  # instead of index error, Qt wants null index
        )

    def getItem(self, index: QModelIndex) -> NodeType:
        """Return python object for a given `QModelIndex`.

        An invalid `QModelIndex` will return the root object.
        """
        if index.isValid():
            item = index.internalPointer()
            if item is not None:
                return item
        return self._root

    def parent(self, index: QModelIndex) -> QModelIndex:
        """Return the parent of the model item with the given ``index``.

        If the item has no parent, an invalid QModelIndex is returned.
        """
        if not index.isValid():
            return QModelIndex()  # null index

        parentItem = self.getItem(index).parent
        if parentItem is None or parentItem == self._root:
            return QModelIndex()

        # A common convention used in models that expose tree data structures
        # is that only items in the first column have children. So here,the
        # column of the returned is 0.
        row = parentItem.index_in_parent() or 0
        return self.createIndex(row, 0, parentItem)

    def mimeTypes(self) -> List[str]:
        """Returns the list of allowed MIME types.

        By default, the built-in models and views use an internal MIME type:
        application/x-qabstractitemmodeldatalist.

        When implementing drag and drop support in a custom model, if you will
        return data in formats other than the default internal MIME type,
        reimplement this function to return your list of MIME types.

        If you reimplement this function in your custom model, you must also
        reimplement the member functions that call it: mimeData() and
        dropMimeData().

        Returns
        -------
        list of str
            MIME types allowed for drag & drop support
        """
        return [NodeMIMEType, "text/plain"]

    def mimeData(self, indices: List[QModelIndex]) -> Optional['NodeMimeData']:
        """Return an object containing serialized data from `indices`.

        The format used to describe the encoded data is obtained from the
        mimeTypes() function. The implementation uses the default MIME type
        returned by the default implementation of mimeTypes(). If you
        reimplement mimeTypes() in your custom model to return more MIME types,
        reimplement this function to make use of them.
        """
        # If the list of indexes is empty, or there are no supported MIME types
        # nullptr is returned rather than a serialized empty list.
        if not indices:
            return 0
        return NodeMimeData([self.getItem(i) for i in indices])

    def dropMimeData(
        self,
        data: QMimeData,
        action: Qt.DropAction,
        destRow: int,
        col: int,
        parent: QModelIndex,
    ) -> bool:
        """Handles `data` from a drag and drop operation ending with `action`.

        The specified row, column and parent indicate the location of an item
        in the model where the operation ended. It is the responsibility of the
        model to complete the action at the correct location.

        Returns
        -------
        bool ``True`` if the `data` and `action` were handled by the model;
            otherwise returns ``False``.
        """
        if not data or action != Qt.DropAction.MoveAction:
            return False
        if not data.hasFormat(self.mimeTypes()[0]):
            return False

        if isinstance(data, NodeMimeData):
            dest_idx = self.getItem(parent).index_from_root()
            dest_idx = (*dest_idx, destRow)
            moving_indices = data.node_indices()

            logger.debug(
                "dropMimeData: indices {ind} ➡ {idx}",
                extra={"ind": moving_indices, "idx": dest_idx},
            )

            if len(moving_indices) == 1:
                self._root.move(moving_indices[0], dest_idx)
            else:
                self._root.move_multiple(moving_indices, dest_idx)
            return True
        return False

    # ###### Non-Qt methods added for Group Model ############

    def setRoot(self, root: Group[NodeType]):
        if not isinstance(root, Group):
            raise TypeError(
                trans._(
                    "root node must be an instance of {Group}",
                    deferred=True,
                    Group=Group,
                )
            )
        super().setRoot(root)

    def nestedIndex(self, nested_index: Tuple[int, ...]) -> QModelIndex:
        """Return a QModelIndex for a given ``nested_index``."""
        parent = QModelIndex()
        if isinstance(nested_index, tuple):
            if not nested_index:
                return parent
            *parents, child = nested_index
            for i in parents:
                parent = self.index(i, 0, parent)
        elif isinstance(nested_index, int):
            child = nested_index
        else:
            raise TypeError(
                trans._(
                    "nested_index must be an int or tuple of int.",
                    deferred=True,
                )
            )
        return self.index(child, 0, parent)


class NodeMimeData(QMimeData):
    """An object to store Node data during a drag operation."""

    def __init__(self, nodes: Optional[List[NodeType]] = None) -> None:
        super().__init__()
        self.nodes: List[NodeType] = nodes or []
        if nodes:
            self.setData(NodeMIMEType, pickle.dumps(self.node_indices()))
            self.setText(" ".join(node._node_name() for node in nodes))

    def formats(self) -> List[str]:
        return [NodeMIMEType, "text/plain"]

    def node_indices(self) -> List[Tuple[int, ...]]:
        return [node.index_from_root() for node in self.nodes]

    def node_names(self) -> List[str]:
        return [node._node_name() for node in self.nodes]
