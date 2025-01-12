from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QSortFilterProxyModel, Qt  # type: ignore[attr-defined]

from napari._qt._qapp_model.qactions._layerlist_context import (
    is_valid_spatial_in_clipboard,
)
from napari._qt.containers._base_item_model import (
    SortRole,
    _BaseEventedItemModel,
)
from napari._qt.containers._layer_delegate import LayerDelegate
from napari._qt.containers.qt_list_view import QtListView
from napari.layers import Layer
from napari.utils.translations import trans

if TYPE_CHECKING:
    from qtpy.QtGui import QKeyEvent  # type: ignore[attr-defined]
    from qtpy.QtWidgets import QWidget  # type: ignore[attr-defined]

    from napari.components.layerlist import LayerList


class ReverseProxyModel(QSortFilterProxyModel):
    """Proxy Model that reverses the view order of a _BaseEventedItemModel."""

    def __init__(self, model: _BaseEventedItemModel) -> None:
        super().__init__()
        self.setSourceModel(model)
        self.setSortRole(SortRole)
        self.sort(0, Qt.SortOrder.DescendingOrder)

    def dropMimeData(self, data, action, destRow, col, parent):
        """Handle destination row for dropping with reversed indices."""
        row = 0 if destRow == -1 else self.sourceModel().rowCount() - destRow
        return self.sourceModel().dropMimeData(data, action, row, col, parent)


class QtLayerList(QtListView[Layer]):
    """QItemView subclass specialized for the LayerList.

    This is as mostly for targetting with QSS, applying the delegate and
    reversing the view with ReverseProxyModel.
    """

    def __init__(self, root: LayerList, parent: QWidget | None = None) -> None:
        root._ctx['valid_spatial_json_clipboard'] = (
            is_valid_spatial_in_clipboard
        )
        super().__init__(root, parent)
        layer_delegate = LayerDelegate()
        self.setItemDelegate(layer_delegate)
        # To be able to update the loading indicator frame in the item delegate
        # smoothly and also be able to leave the item painted in a coherent
        # state (showing the loading indicator or the thumbnail)
        viewport = self.viewport()
        assert viewport is not None

        layer_delegate.loading_frame_changed.connect(viewport.update)

        self.setToolTip(trans._('Layer list'))

        # This reverses the order of the items in the view,
        # so items at the end of the list are at the top.
        self.setModel(ReverseProxyModel(self.model()))

    def keyPressEvent(self, e: QKeyEvent | None) -> None:
        """Override Qt event to pass events to the viewer."""
        if e is None:
            return
        # capture arrows with modifiers so they are handled by Viewer keybindings
        if (e.key() == Qt.Key.Key_Up or e.key() == Qt.Key.Key_Down) and (
            e.modifiers() & Qt.KeyboardModifier.AltModifier
            or e.modifiers() & Qt.KeyboardModifier.ControlModifier
            or e.modifiers() & Qt.KeyboardModifier.MetaModifier
            or e.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            e.ignore()
        elif e.key() != Qt.Key.Key_Space:
            super().keyPressEvent(e)
        if e.key() not in (
            Qt.Key.Key_Backspace,
            Qt.Key.Key_Delete,
            Qt.Key.Key_Return,
        ):
            e.ignore()  # pass key events up to viewer
