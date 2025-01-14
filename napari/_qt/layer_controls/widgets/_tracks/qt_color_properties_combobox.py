from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtColorPropertiesComboBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer color properties
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
        color_by_combobox : qtpy.QtWidgets.QComboBox
            ComboBox controlling current color property of the layer.
        color_by_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
            Label for the color property chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.color_by.connect(self._on_color_by_change)
        self._layer.events.properties.connect(self._on_properties_change)

        # Setup widgets
        # combo box for track coloring, we can get these from the properties
        # keys
        self.color_by_combobox = QComboBox()
        self.color_by_combobox.addItems(self._layer.properties_to_color_by)
        self.color_by_combobox.currentTextChanged.connect(self.change_color_by)

        self.color_by_combobox_label = QtWrappedLabel(trans._('color by:'))

        self._on_color_by_change()

    def change_color_by(self, value: str) -> None:
        self._layer.color_by = value

    def _on_color_by_change(self) -> None:
        """Receive layer model color_by change event and update combobox."""
        with self._layer.events.color_by.blocker():
            color_by = self._layer.color_by

            idx = self.color_by_combobox.findText(
                color_by, Qt.MatchFlag.MatchFixedString
            )
            self.color_by_combobox.setCurrentIndex(idx)

    def _on_properties_change(self) -> None:
        """Change the properties that can be used to color the tracks."""
        with qt_signals_blocked(self.color_by_combobox):
            self.color_by_combobox.clear()
            self.color_by_combobox.addItems(self._layer.properties_to_color_by)
        self._on_color_by_change()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.color_by_combobox_label, self.color_by_combobox)]
