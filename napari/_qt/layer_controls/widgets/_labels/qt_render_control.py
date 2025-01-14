from typing import Optional

from qtpy.QtWidgets import QWidget
from superqt import QEnumComboBox

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.layers.labels._labels_constants import (
    IsoCategoricalGradientMode,
    LabelsRendering,
)
from napari.utils.translations import trans


class QtLabelRenderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer attribute for
    the method to render the labels and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    renderComboBox : superqt.QEnumComboBox
        Combobox to control current label render method.
    renderComboBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the way labels should be rendered chooser widget.
    isoGradientComboBox : superqt.QEnumComboBox
        Combobox to control gradient method when isosurface rendering is selected.
    isoGradientComboBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the chooser widget of the gradient to use when labels are using isosurface rendering.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.rendering.connect(self._on_rendering_change)
        self._layer.events.iso_gradient_mode.connect(
            self._on_iso_gradient_mode_change
        )

        # Setup widgets
        renderComboBox = QEnumComboBox(enum_class=LabelsRendering)
        renderComboBox.setCurrentEnum(LabelsRendering(self._layer.rendering))
        renderComboBox.currentEnumChanged.connect(self.changeRendering)
        self.renderComboBox = renderComboBox
        self.renderComboBoxLabel = QtWrappedLabel(trans._('rendering:'))

        isoGradientComboBox = QEnumComboBox(
            enum_class=IsoCategoricalGradientMode
        )
        isoGradientComboBox.setCurrentEnum(
            IsoCategoricalGradientMode(self._layer.iso_gradient_mode)
        )
        isoGradientComboBox.currentEnumChanged.connect(
            self.changeIsoGradientMode
        )
        isoGradientComboBox.setEnabled(
            self._layer.rendering == LabelsRendering.ISO_CATEGORICAL
        )
        self.isoGradientComboBox = isoGradientComboBox
        self.isoGradientComboBoxLabel = QtWrappedLabel(
            trans._('gradient\nmode:')
        )

    def changeRendering(self, rendering_mode: LabelsRendering):
        """Change rendering mode for image display.

        Parameters
        ----------
        rendering_mode : LabelsRendering
            Rendering mode used by vispy.
            Selects a preset rendering mode in vispy that determines how
            volume is displayed:
            * translucent: voxel colors are blended along the view ray until
              the result is opaque.
            * iso_categorical: isosurface for categorical data (e.g., labels).
              Cast a ray until a value greater than zero is encountered. At that
              location, lighning calculations are performed to give the visual
              appearance of a surface.
        """
        self.isoGradientComboBox.setEnabled(
            rendering_mode == LabelsRendering.ISO_CATEGORICAL
        )
        self._layer.rendering = rendering_mode

    def changeIsoGradientMode(self, gradient_mode: IsoCategoricalGradientMode):
        """Change gradient mode for isosurface rendering.

        Parameters
        ----------
        gradient_mode : IsoCategoricalGradientMode
            Gradient mode for the isosurface rendering method.
            Selects the finite-difference gradient method for the isosurface shader:
            * fast: simple finite difference gradient along each axis
            * smooth: isotropic Sobel gradient, smoother but more computationally expensive
        """
        self._layer.iso_gradient_mode = gradient_mode

    def _on_rendering_change(self):
        """Receive layer model rendering change event and update dropdown menu."""
        with self._layer.events.rendering.blocker():
            self.renderComboBox.setCurrentEnum(
                LabelsRendering(self._layer.rendering)
            )

    def _on_iso_gradient_mode_change(self):
        """Receive layer model iso_gradient_mode change event and update dropdown menu."""
        with self._layer.events.iso_gradient_mode.blocker():
            self.isoGradientComboBox.setCurrentEnum(
                IsoCategoricalGradientMode(self._layer.iso_gradient_mode)
            )

    def _on_display_change_hide(self):
        self.renderComboBox.hide()
        self.renderComboBoxLabel.hide()
        self.isoGradientComboBox.hide()
        self.isoGradientComboBoxLabel.hide()

    def _on_display_change_show(self):
        self.renderComboBox.show()
        self.renderComboBoxLabel.show()
        self.isoGradientComboBox.show()
        self.isoGradientComboBoxLabel.show()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.renderComboBoxLabel, self.renderComboBox),
            (self.isoGradientComboBoxLabel, self.isoGradientComboBox),
        ]
