from __future__ import annotations

from collections.abc import MutableSequence
from typing import TYPE_CHECKING, TypeVar

from qtpy.QtWidgets import QTreeView

from napari._qt.containers._base_item_view import _BaseEventedItemView
from napari._qt.containers.qt_tree_model import QtNodeTreeModel
from napari.utils.tree import Group, Node

if TYPE_CHECKING:
    from qtpy.QtCore import QModelIndex
    from qtpy.QtWidgets import QWidget  # type: ignore[attr-defined]


NodeType = TypeVar('NodeType', bound=Node)


class QtNodeTreeView(_BaseEventedItemView[NodeType], QTreeView):
    """A QListView for a :class:`~napari.utils.tree.Group`.

    Designed to work with :class:`~napari._qt.containers.QtNodeTreeModel`.

    This class is an adapter between :class:`~napari.utils.tree.Group` and Qt's
    `QAbstractItemView` interface (see `Qt Model/View Programming
    <https://doc.qt.io/qt-5/model-view-programming.html>`_). It allows python
    users to interact with a list of lists in the "usual" python ways, updating
    any Qt Views that may be connected, and also updates the python list object
    if any GUI events occur in the view.

    See docstring of :class:`_BaseEventedItemView` for additional background.
    """

    _root: Group[Node]

    def __init__(
        self, root: Group[Node], parent: QWidget | None = None
    ) -> None:
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
