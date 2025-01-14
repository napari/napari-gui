from qtpy.QtWidgets import (
    QCheckBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtIdCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the id should be
    displayed attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    id_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox controlling if id of the layer should be shown.
    id_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for showing the id chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        # NOTE(arl): there are no events fired for changing checkbox (layer `display_id` attribute)

        # Setup widgets
        self.id_checkbox = QCheckBox()
        self.id_checkbox.stateChanged.connect(self.change_display_id)

        self.id_checkbox_label = QtWrappedLabel(trans._('show ID:'))

    def change_display_id(self, state) -> None:
        self._layer.display_id = self.id_checkbox.isChecked()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.id_checkbox_label, self.id_checkbox)]
