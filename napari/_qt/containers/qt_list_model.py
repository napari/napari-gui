import logging
import pickle
from collections.abc import Sequence
from typing import Optional, TypeVar

from qtpy.QtCore import QMimeData, QModelIndex, Qt

from napari._qt.containers._base_item_model import _BaseEventedItemModel

logger = logging.getLogger(__name__)
ListIndexMIMEType = 'application/x-list-index'
ItemType = TypeVar('ItemType')


class QtListModel(_BaseEventedItemModel[ItemType]):
    """A QItemModel for a :class:`~napari.utils.events.SelectableEventedList`.

    Designed to work with :class:`~napari._qt.containers.QtListView`.

    See docstring of :class:`_BaseEventedItemModel` and
    :class:`~napari._qt.containers.QtListView` for additional background.
    """

    def mimeTypes(self) -> list[str]:
        """Returns the list of allowed MIME types.

        When implementing drag and drop support in a custom model, if you will
        return data in formats other than the default internal MIME type,
        reimplement this function to return your list of MIME types.
        """
        return [ListIndexMIMEType, 'text/plain']

    def mimeData(self, indices: list[QModelIndex]) -> Optional['QMimeData']:
        """Return an object containing serialized data from `indices`.

        If the list of indexes is empty, or there are no supported MIME types,
        None is returned rather than a serialized empty list.
        """
        if not indices:
            return None
        items, indices = zip(
            *[(self.getItem(i), i.row()) for i in indices], strict=False
        )
        return ItemMimeData(items, indices)

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

        if isinstance(data, ItemMimeData):
            moving_indices = data.indices

            logger.debug(
                'dropMimeData: indices %s ➡ %s',
                moving_indices,
                destRow,
            )

            if len(moving_indices) == 1:
                return self._root.move(moving_indices[0], destRow)

            return bool(self._root.move_multiple(moving_indices, destRow))
        return False


class ItemMimeData(QMimeData):
    """An object to store list indices data during a drag operation."""

    def __init__(
        self, items: Sequence[ItemType], indices: Sequence[int]
    ) -> None:
        super().__init__()
        self.items = items
        self.indices = tuple(sorted(indices))
        if items:
            self.setData(ListIndexMIMEType, pickle.dumps(self.indices))
            self.setText(' '.join(str(item) for item in items))

    def formats(self) -> list[str]:
        return [ListIndexMIMEType, 'text/plain']
