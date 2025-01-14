from collections.abc import Iterable

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.widgets._slider_compat import QSlider
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtEdgeWidthSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current edge
    width layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    edgeWidthSlider : qtpy.QtWidgets.QSlider
        Slider controlling line edge width of layer.
    edgeWidthLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current edge width widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.edge_width.connect(self._on_edge_width_change)

        # Setup widgets
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(40)
        sld.setSingleStep(1)
        value = self._layer.current_edge_width
        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value.mean()
        sld.setValue(int(value))
        sld.valueChanged.connect(self.changeWidth)
        self.edgeWidthSlider = sld
        self.edgeWidthSlider.setToolTip(
            trans._(
                'Set the edge width of currently selected shapes and any added afterwards.'
            )
        )
        self.edgeWidthLabel = QtWrappedLabel(trans._('edge width:'))

    def changeWidth(self, value: float) -> None:
        """Change edge line width of shapes on the layer model.

        Parameters
        ----------
        value : float
            Line width of shapes.
        """
        self._layer.current_edge_width = float(value)

    def _on_edge_width_change(self) -> None:
        """Receive layer model edge line width change event and update slider."""
        with self._layer.events.edge_width.blocker():
            value = self._layer.current_edge_width
            value = np.clip(int(value), 0, 40)
            self.edgeWidthSlider.setValue(value)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.edgeWidthLabel, self.edgeWidthSlider)]
