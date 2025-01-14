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
from napari.layers.surface._surface_constants import SHADING_TRANSLATION
from napari.utils.translations import trans


class QtShadingComboBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer shading
    value attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    shadingComboBox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current shading value of the layer.
    shadingComboBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the shading value chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.shading.connect(self._on_shading_change)

        # Setup widgets
        shading_comboBox = QComboBox(parent)
        for display_name, shading in SHADING_TRANSLATION.items():
            shading_comboBox.addItem(display_name, shading)
        index = shading_comboBox.findData(
            SHADING_TRANSLATION[self._layer.shading]
        )
        shading_comboBox.setCurrentIndex(index)
        shading_comboBox.currentTextChanged.connect(self.changeShading)
        self.shadingComboBox = shading_comboBox

        self.shadingComboBoxLabel = QtWrappedLabel(trans._('shading:'))

    def changeShading(self, text: str) -> None:
        """Change shading value on the surface layer.
        Parameters
        ----------
        text : str
            Name of shading mode, eg: 'flat', 'smooth', 'none'.
        """
        self._layer.shading = self.shadingComboBox.currentData()

    def _on_shading_change(self) -> None:
        """Receive layer model shading change event and update combobox."""
        with self._layer.events.shading.blocker():
            self.shadingComboBox.setCurrentIndex(
                self.shadingComboBox.findData(
                    SHADING_TRANSLATION[self._layer.shading]
                )
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.shadingComboBoxLabel, self.shadingComboBox)]
