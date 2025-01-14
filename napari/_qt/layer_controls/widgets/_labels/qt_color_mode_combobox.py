from typing import Optional

from qtpy.QtWidgets import (
    QComboBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.layers.labels._labels_constants import (
    LABEL_COLOR_MODE_TRANSLATIONS,
    LabelColorMode,
)
from napari.utils import CyclicLabelColormap
from napari.utils.translations import trans


class QtColorModeComboBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer color
    mode attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    colorModeComboBox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current color mode of the layer.
    colorModeComboBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the color mode chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.colormap.connect(self._on_colormap_change)

        # Setup widgets
        color_mode_comboBox = QComboBox()
        for data, text in LABEL_COLOR_MODE_TRANSLATIONS.items():
            data = data.value
            color_mode_comboBox.addItem(text, data)

        self.colorModeComboBox = color_mode_comboBox
        self._on_colormap_change()
        color_mode_comboBox.activated.connect(self.change_color_mode)

        self.colorModeComboBoxLabel = QtWrappedLabel(trans._('color mode:'))

    def change_color_mode(self) -> None:
        """Change color mode of label layer"""
        if self.colorModeComboBox.currentData() == LabelColorMode.AUTO.value:
            self._layer.colormap = self._layer._original_random_colormap
        else:
            self._layer.colormap = self._layer._direct_colormap

    def _on_colormap_change(self) -> None:
        enable_combobox = not self._layer._is_default_colors(
            self._layer._direct_colormap.color_dict
        )
        self.colorModeComboBox.setEnabled(enable_combobox)
        if not enable_combobox:
            self.colorModeComboBox.setToolTip(
                'Layer needs a user-set DirectLabelColormap to enable direct '
                'mode.'
            )
        if isinstance(self._layer.colormap, CyclicLabelColormap):
            self.colorModeComboBox.setCurrentIndex(
                self.colorModeComboBox.findData(LabelColorMode.AUTO.value)
            )
        else:
            self.colorModeComboBox.setCurrentIndex(
                self.colorModeComboBox.findData(LabelColorMode.DIRECT.value)
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.colorModeComboBoxLabel, self.colorModeComboBox)]
