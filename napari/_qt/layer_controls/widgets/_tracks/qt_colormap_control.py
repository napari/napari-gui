from qtpy.QtWidgets import QComboBox, QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.utils.translations import trans


class QtColormapComboBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer colormaps
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    colormap_combobox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current colormap of the layer.
    colormap_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the colormap chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.colormap.connect(self._on_colormap_change)

        # Setup widgets
        self.colormap_combobox = QComboBox()
        for name, colormap in AVAILABLE_COLORMAPS.items():
            display_name = colormap._display_name
            self.colormap_combobox.addItem(display_name, name)
        self.colormap_combobox.currentTextChanged.connect(self.change_colormap)

        self.colormap_combobox_label = QtWrappedLabel(trans._('colormap:'))

        self._on_colormap_change()

    def change_colormap(self, colormap: str):
        self._layer.colormap = self.colormap_combobox.currentData()

    def _on_colormap_change(self):
        """Receive layer model colormap change event and update combobox."""
        with self._layer.events.colormap.blocker():
            self.colormap_combobox.setCurrentIndex(
                self.colormap_combobox.findData(self._layer.colormap)
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.colormap_combobox_label, self.colormap_combobox)]
