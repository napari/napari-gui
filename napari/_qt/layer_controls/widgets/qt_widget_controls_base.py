from qtpy.QtCore import QObject, Qt
from qtpy.QtWidgets import QLabel, QWidget

from napari.layers.base.base import Layer
from napari.utils.events import disconnect_events


class QtWrappedLabel(QLabel):
    """
    QLabel subclass with the `wordWrap` activated (True) and text aligned
    to the right and vertically centered by default.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWordWrap(True)
        self.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )


class QtWidgetControlsBase(QObject):
    """
    Base class that defines base methods for wrapper classes that do the
    connection of events/signals between layer attributes and Qt widgets.

    This class is never directly instantiated anywhere.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent)
        # Setup layer
        self._layer = layer

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        """
        Enable access to the created labels and control widgets.

        Returns
        -------
        list : list[tuple[QtWrappedLabel, QWidget]]
            List of tuples of the label and widget controls available.

        """
        raise NotImplementedError

    def disconnect_widget_controls(self) -> None:
        """
        Disconnect layer from widget controls.
        """
        disconnect_events(self._layer.events, self)

    def deleteLater(self) -> None:
        disconnect_events(self._layer.events, self)
        super().deleteLater()
