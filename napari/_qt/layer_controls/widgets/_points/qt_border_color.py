from typing import Optional

import numpy as np
from qtpy.QtCore import Slot
from qtpy.QtWidgets import QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtBorderColorControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current border
    color layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    borderColorEdit : napari._qt.widgets.qt_color_swatch.QColorSwatchEdit
        ColorSwatchEdit controlling current face color of the layer.
    borderColorEditLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current egde color chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.current_border_color.connect(
            self._on_current_border_color_change
        )
        self._layer._border.events.current_color.connect(
            self._on_current_border_color_change
        )

        # Setup widgets
        self.borderColorEdit = QColorSwatchEdit(
            initial_color=self._layer.current_border_color,
            tooltip=trans._(
                'Click to set the border color of currently selected points and any added afterwards.'
            ),
        )
        self.borderColorEdit.color_changed.connect(
            self.changeCurrentBorderColor
        )
        self.borderColorEditLabel = QtWrappedLabel(trans._('border color:'))

    @Slot(np.ndarray)
    def changeCurrentBorderColor(self, color: np.ndarray) -> None:
        """Update border color of layer model from color picker user input."""
        with self._layer.events.current_border_color.blocker(
            self._on_current_border_color_change
        ):
            self._layer.current_border_color = color

    def _on_current_border_color_change(self) -> None:
        """Receive layer.current_border_color() change event and update view."""
        with qt_signals_blocked(self.borderColorEdit):
            self.borderColorEdit.setColor(self._layer.current_border_color)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.borderColorEditLabel, self.borderColorEdit)]
