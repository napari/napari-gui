from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.utils.events import disconnect_events
from napari.utils.translations import trans


class QtTextVisibilityControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the text visibility
    layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    textDispCheckBox : qtpy.QtWidgets.QCheckbox
        Checkbox controlling if text on the layer is visible or not.
    textDispLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the text visibility widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.text.events.visible.connect(
            self._on_text_visibility_change
        )

        # Setup widgets
        text_disp_cb = QCheckBox()
        text_disp_cb.setToolTip(trans._('Toggle text visibility'))
        text_disp_cb.setChecked(self._layer.text.visible)
        text_disp_cb.stateChanged.connect(self.change_text_visibility)
        self.textDispCheckBox = text_disp_cb
        self.textDispLabel = QtWrappedLabel(trans._('display text:'))

    def change_text_visibility(self, state: int) -> None:
        """Toggle the visibility of the text.

        Parameters
        ----------
        state : int
            Integer value of Qt.CheckState that indicates the check state of textDispCheckBox
        """
        with self._layer.text.events.visible.blocker(
            self._on_text_visibility_change
        ):
            self._layer.text.visible = (
                Qt.CheckState(state) == Qt.CheckState.Checked
            )

    def _on_text_visibility_change(self) -> None:
        """Receive layer model text visibiltiy change change event and update checkbox."""
        with self._layer.text.events.visible.blocker():
            self.textDispCheckBox.setChecked(self._layer.text.visible)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.textDispLabel, self.textDispCheckBox)]

    def disconnect_widget_controls(self) -> None:
        disconnect_events(self._layer.text.events, self)
        super().disconnect_widget_controls()

    def deleteLater(self) -> None:
        disconnect_events(self._layer.text.events, self)
        super().deleteLater()
