from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QImage

from napari._qt.containers._base_item_model import ItemRole, ThumbnailRole
from napari._qt.containers._layer_delegate import LayerDelegate

# from .qt_layer_list import ReverseProxyModel
from napari._qt.containers.qt_tree_model import QtNodeTreeModel
from napari._qt.containers.qt_tree_view import QtNodeTreeView
from napari.layers import Layer

if TYPE_CHECKING:
    from qtpy.QtCore import QModelIndex
    from qtpy.QtWidgets import QStyleOptionViewItem, QWidget

    from napari.layers.layergroup import LayerGroup


class QtLayerTreeModel(QtNodeTreeModel[Layer]):
    def __init__(self, root: LayerGroup, parent: QWidget = None):
        super().__init__(root, parent)
        self.data
        self.setRoot(root)

    # TODO: there's a lot of duplicated logic from QtLayerListModel here
    # condense...
    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        """Return data stored under ``role`` for the item at ``index``."""
        if not index.isValid():
            return None
        layer = self.getItem(index)
        if role == ItemRole:  # custom role: return the layer
            return layer
        if role == Qt.ItemDataRole.DisplayRole:  # used for item text
            return layer.name
        if role == Qt.ItemDataRole.TextAlignmentRole:  # alignment of the text
            return Qt.AlignCenter
        if role == Qt.ItemDataRole.EditRole:
            # used to populate line edit when editing
            return layer.name
        if role == Qt.ItemDataRole.ToolTipRole:  # for tooltip
            return layer.get_source_str()
        if (
            role == Qt.ItemDataRole.CheckStateRole
        ):  # the "checked" state of this item
            if layer.visible:
                parents_visible = all(p._visible for p in layer.iter_parents())
                return Qt.Checked if parents_visible else Qt.PartiallyChecked
            else:
                return Qt.Unchecked
        if role == Qt.ItemDataRole.SizeHintRole:  # determines size of item
            return QSize(200, 34)
        if role == ThumbnailRole:  # return the thumbnail
            thumbnail = layer.thumbnail
            return QImage(
                thumbnail,
                thumbnail.shape[1],
                thumbnail.shape[0],
                QImage.Format_RGBA8888,
            )
        # normally you'd put the icon in DecorationRole, but we do that in the
        # # LayerDelegate which is aware of the theme.
        # if role == Qt.ItemDataRole.DecorationRole:  # icon to show
        #     pass
        return super().data(index, role)

    def setData(
        self,
        index: QModelIndex,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if role == Qt.ItemDataRole.CheckStateRole:
            self.getItem(index).visible = value
        elif role == Qt.ItemDataRole.EditRole:
            self.getItem(index).name = value
            role = Qt.ItemDataRole.DisplayRole
        else:
            return super().setData(index, value, role=role)

        self.dataChanged.emit(index, index, [role])
        return True

    def _process_event(self, event):
        # The model needs to emit `dataChanged` whenever data has changed
        # for a given index, so that views can update themselves.
        # Here we convert native events to the dataChanged signal.
        if not hasattr(event, 'index'):
            self.dataChanged.emit(
                self.index(0), self.index(self.rowCount()), []
            )
            return
        role = {
            'thumbnail': ThumbnailRole,
            'visible': Qt.ItemDataRole.CheckStateRole,
            'name': Qt.ItemDataRole.DisplayRole,
        }.get(event.type)
        roles = [role] if role is not None else []
        top = self.nestedIndex(event.index)
        bot = self.index(top.row() + 1)
        self.dataChanged.emit(top, bot, roles)


class QtLayerTreeView(QtNodeTreeView):
    _root: LayerGroup
    model_class = QtLayerTreeModel

    def __init__(self, root: LayerGroup, parent: QWidget = None):
        super().__init__(root, parent)
        self.setItemDelegate(LayerDelegate())
        self.setIndentation(18)

        # couldn't set in qss for some reason
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)

        # # This reverses the order of the items in the view,
        # # so items at the end of the list are at the top.
        # # FIXME! not working at the moment, and causing test segfaults
        # self.setModel(ReverseProxyModel(self.model()))

    def viewOptions(self) -> QStyleOptionViewItem:
        options = super().viewOptions()
        options.decorationPosition = options.Right
        return options
