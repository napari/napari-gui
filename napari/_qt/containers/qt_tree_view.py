from __future__ import annotations

from collections.abc import MutableSequence
from typing import TYPE_CHECKING, TypeVar

from qtpy.QtWidgets import QTreeView

from ...utils.tree import Group, Node
from ._base_item_view import _BaseEventedItemView
from .qt_tree_model import QtNodeTreeModel

if TYPE_CHECKING:
    from qtpy.QtCore import QModelIndex
    from qtpy.QtWidgets import QWidget


NodeType = TypeVar("NodeType", bound=Node)


class QtNodeTreeView(QTreeView, _BaseEventedItemView[NodeType]):
    _root: Group[Node]

    def __init__(self, root: Group[Node], parent: QWidget = None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragDropMode(QTreeView.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setSelectionMode(QTreeView.ExtendedSelection)
        self.setRoot(root)

    def setRoot(self, root: Group[Node]):
        super().setRoot(root)

        # make tree look like a list if it contains no lists.
        self.model().rowsRemoved.connect(self._redecorate_root)
        self.model().rowsInserted.connect(self._redecorate_root)
        self._redecorate_root()

    def _redecorate_root(self, parent: QModelIndex = None, *_):
        """Add a branch/arrow column only if there are Groups in the root.

        This makes the tree fall back to looking like a simple list if there
        are no lists in the root level.
        """
        if not parent or not parent.isValid():
            hasgroup = any(isinstance(i, MutableSequence) for i in self._root)
            self.setRootIsDecorated(hasgroup)

    def model(self) -> QtNodeTreeModel[NodeType]:
        return super().model()
