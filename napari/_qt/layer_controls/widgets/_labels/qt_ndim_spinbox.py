from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QSpinBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtNdimSpinBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer number of
    editable dimensions attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    ndimSpinBox : qtpy.QtWidgets.QSpinBox
        Spinbox to control the number of editable dimensions of label layer.
    ndimSpinBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the number of editable dimensions chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.n_edit_dimensions.connect(
            self._on_n_edit_dimensions_change
        )

        # Setup widgets
        ndim_sb = QSpinBox()
        self.ndimSpinBox = ndim_sb
        ndim_sb.setToolTip(trans._('Number of dimensions for label editing'))
        ndim_sb.valueChanged.connect(self.change_n_edit_dim)
        ndim_sb.setMinimum(2)
        ndim_sb.setMaximum(self._layer.ndim)
        ndim_sb.setSingleStep(1)
        ndim_sb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._on_n_edit_dimensions_change()

        self.ndimSpinBoxLabel = QtWrappedLabel(trans._('n edit dim:'))

    def change_n_edit_dim(self, value: int) -> None:
        """Change the number of editable dimensions of label layer.

        Parameters
        ----------
        value : int
            The number of editable dimensions to set.
        """
        self._layer.n_edit_dimensions = value
        self.ndimSpinBox.clearFocus()

        # TODO: Check how to decouple this
        self.parent().setFocus()

    def _on_n_edit_dimensions_change(self) -> None:
        """Receive layer model n-dim mode change event and update the checkbox."""
        with self._layer.events.n_edit_dimensions.blocker():
            value = self._layer.n_edit_dimensions
            self.ndimSpinBox.setValue(int(value))
            # TODO: Check how to decouple this
            self.parent()._set_polygon_tool_state()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.ndimSpinBoxLabel, self.ndimSpinBox)]
