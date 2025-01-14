import numpy as np
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtWidthSpinBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer line width
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    widthSpinBox : qtpy.QtWidgets.QDoubleSpinBox
        Spin box widget controlling edge line width of vectors.
    widthSpinBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the edge line width of vectors chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.edge_width.connect(self._on_edge_width_change)

        # Setup widgets
        # line width in pixels
        self.widthSpinBox = QDoubleSpinBox()
        self.widthSpinBox.setKeyboardTracking(False)
        self.widthSpinBox.setSingleStep(0.1)
        self.widthSpinBox.setMinimum(0.01)
        self.widthSpinBox.setMaximum(np.inf)
        self.widthSpinBox.setValue(self._layer.edge_width)
        self.widthSpinBox.valueChanged.connect(self.change_width)

        self.widthSpinBoxLabel = QtWrappedLabel(trans._('width:'))

    def change_width(self, value) -> None:
        """Change edge line width of vectors on the layer model.

        Parameters
        ----------
        value : float
            Line width of vectors.
        """
        self._layer.edge_width = value
        self.widthSpinBox.clearFocus()
        # TODO: Check other way to give focus without calling parent
        self.parent().setFocus()

    def _on_edge_width_change(self) -> None:
        """Receive layer model width change event and update width spinbox."""
        with self._layer.events.edge_width.blocker():
            self.widthSpinBox.setValue(self._layer.edge_width)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.widthSpinBoxLabel, self.widthSpinBox)]


class QtLengthSpinBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer line width
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    lengthSpinBox : qtpy.QtWidgets.QDoubleSpinBox
        Spinbox widget controlling line length of vectors.
        Multiplicative factor on projections for length of all vectors.
    lengthSpinBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for line length of vectors chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.length.connect(self._on_length_change)

        # Setup widgets
        # line length
        self.lengthSpinBox = QDoubleSpinBox()
        self.lengthSpinBox.setKeyboardTracking(False)
        self.lengthSpinBox.setSingleStep(0.1)
        self.lengthSpinBox.setValue(self._layer.length)
        self.lengthSpinBox.setMinimum(0.1)
        self.lengthSpinBox.setMaximum(np.inf)
        self.lengthSpinBox.valueChanged.connect(self.change_length)

        self.lengthSpinBoxLabel = QtWrappedLabel(trans._('length:'))

    def change_length(self, value: float) -> None:
        """Change length of vectors on the layer model.

        Multiplicative factor on projections for length of all vectors.

        Parameters
        ----------
        value : float
            Length of vectors.
        """
        self._layer.length = value
        self.lengthSpinBox.clearFocus()
        # TODO: Check other way to give focus without calling parent
        self.parent().setFocus()

    def _on_length_change(self) -> None:
        """Change length of vectors."""
        with self._layer.events.length.blocker():
            self.lengthSpinBox.setValue(self._layer.length)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.lengthSpinBoxLabel, self.lengthSpinBox)]
