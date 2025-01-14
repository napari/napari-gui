import contextlib
from typing import Optional

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets._slider_compat import QSlider
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtCurrentSizeSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current
    size layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    sizeSlider : napari._qt.widgets._slider_compat.QSlider
        Slider controlling current size attribute of the layer.
    sizeSliderLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the size chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.size.connect(self._on_current_size_change)
        self._layer.events.current_size.connect(self._on_current_size_change)

        # Setup widgets
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setToolTip(
            trans._(
                'Change the size of currently selected points and any added afterwards.'
            )
        )
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(1)
        if self._layer.size.size:
            max_value = max(100, int(np.max(self._layer.size)) + 1)
        else:
            max_value = 100
        sld.setMaximum(max_value)
        sld.setSingleStep(1)
        value = self._layer.current_size
        sld.setValue(int(value))
        sld.valueChanged.connect(self.changeCurrentSize)
        self.sizeSlider = sld

        self.sizeSliderLabel = QtWrappedLabel(trans._('point size:'))

    def changeCurrentSize(self, value: float) -> None:
        """Change size of points on the layer model.

        Parameters
        ----------
        value : float
            Size of points.
        """
        with self._layer.events.current_size.blocker(
            self._on_current_size_change
        ):
            self._layer.current_size = value

    def _on_current_size_change(self) -> None:
        """Receive layer model size change event and update point size slider."""
        with qt_signals_blocked(self.sizeSlider):
            value = self._layer.current_size
            min_val = min(value) if isinstance(value, list) else value
            max_val = max(value) if isinstance(value, list) else value
            if min_val < self.sizeSlider.minimum():
                self.sizeSlider.setMinimum(max(1, int(min_val - 1)))
            if max_val > self.sizeSlider.maximum():
                self.sizeSlider.setMaximum(int(max_val + 1))
            with contextlib.suppress(TypeError):
                self.sizeSlider.setValue(int(value))

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.sizeSliderLabel, self.sizeSlider)]
