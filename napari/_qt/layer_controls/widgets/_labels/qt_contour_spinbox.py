from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
)
from superqt import QLargeIntSpinBox

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.layers.labels._labels_utils import get_dtype
from napari.utils._dtype import get_dtype_limits
from napari.utils.translations import trans


class QtContourSpinBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer contour
    thickness attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    contourSpinBox : superqt.QLargeSpinBox
        Spinbox to control the layer contour thickness.
    contourSpinBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the layer contour thickness chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.contour.connect(self._on_contour_change)

        # Setup widgets
        self.contourSpinBox = QLargeIntSpinBox()
        dtype_lims = get_dtype_limits(get_dtype(layer))
        self.contourSpinBox.setRange(0, dtype_lims[1])
        self.contourSpinBox.setToolTip(
            trans._('Set width of displayed label contours')
        )
        self.contourSpinBox.valueChanged.connect(self.change_contour)
        self.contourSpinBox.setKeyboardTracking(False)
        self.contourSpinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._on_contour_change()

        self.contourSpinBoxLabel = QtWrappedLabel(trans._('contour:'))

    def change_contour(self, value: int) -> None:
        """Change contour thickness.

        Parameters
        ----------
        value : int
            Thickness of contour.
        """
        self._layer.contour = value
        self.contourSpinBox.clearFocus()
        self.parent().setFocus()

    def _on_contour_change(self) -> None:
        """Receive layer model contour value change event and update spinbox."""
        with self._layer.events.contour.blocker():
            value = self._layer.contour
            self.contourSpinBox.setValue(value)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.contourSpinBoxLabel, self.contourSpinBox)]
