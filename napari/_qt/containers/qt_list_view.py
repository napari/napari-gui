from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from qtpy.QtWidgets import QListView

from napari._qt.containers._base_item_view import _BaseEventedItemView
from napari._qt.containers.qt_list_model import QtListModel

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget  # type: ignore[attr-defined]

    from napari.utils.events.containers import SelectableEventedList

ItemType = TypeVar('ItemType')


class QtListView(_BaseEventedItemView[ItemType], QListView):
    """A QListView for a :class:`~napari.utils.events.SelectableEventedList`.

    Designed to work with :class:`~napari._qt.containers.QtListModel`.

    This class is an adapter between
    :class:`~napari.utils.events.SelectableEventedList` and Qt's
    `QAbstractItemView` interface (see `Qt Model/View Programming
    <https://doc.qt.io/qt-5/model-view-programming.html>`_). It allows python
    users to interact with the list in the "usual" python ways, updating any Qt
    Views that may be connected, and also updates the python list object if any
    GUI events occur in the view.

    See docstring of :class:`_BaseEventedItemView` for additional background.
    """

    _root: SelectableEventedList[ItemType]

    def __init__(
        self,
        root: SelectableEventedList[ItemType],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setDragDropMode(QListView.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setSelectionMode(QListView.ExtendedSelection)
        self.setRoot(root)

    def model(self) -> QtListModel[ItemType]:
        return super().model()
