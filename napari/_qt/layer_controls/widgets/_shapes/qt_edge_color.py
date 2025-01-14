from typing import Optional

import numpy as np
from qtpy.QtWidgets import QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtEdgeColorControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current edge
    color layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    edgeColorEdit : qtpy.QtWidgets.QSlider
        ColorSwatchEdit controlling current edge color of the layer.
    edgeColorLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current edge color chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.current_edge_color.connect(
            self._on_current_edge_color_change
        )

        # Setup widgets
        self.edgeColorEdit = QColorSwatchEdit(
            initial_color=self._layer.current_edge_color,
            tooltip=tooltip,
        )
        self._on_current_edge_color_change()
        self.edgeColorEdit.color_changed.connect(self.changeEdgeColor)
        self.edgeColorLabel = QtWrappedLabel(trans._('edge color:'))

    def changeEdgeColor(self, color: np.ndarray) -> None:
        """Change edge color of shapes.

        Parameters
        ----------
        color : np.ndarray
            Edge color for shapes, color name or hex string.
            Eg: 'white', 'red', 'blue', '#00ff00', etc.
        """
        with self._layer.events.current_edge_color.blocker():
            self._layer.current_edge_color = color

    def _on_current_edge_color_change(self) -> None:
        """Receive layer model edge color change event and update color swatch."""
        with qt_signals_blocked(self.edgeColorEdit):
            self.edgeColorEdit.setColor(self._layer.current_edge_color)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.edgeColorLabel, self.edgeColorEdit)]
