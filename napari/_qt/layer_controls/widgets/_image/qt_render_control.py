from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QSlider, QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.widgets._slider_compat import QDoubleSlider
from napari.layers.base.base import Layer
from napari.layers.image._image_constants import (
    ImageRendering,
)
from napari.utils.translations import trans


class QtImageRenderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer attribute for
    the method to render an image and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    renderComboBox : qtpy.QtWidgets.QComboBox
        Combobox to control labels render method.
    renderComboBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the way labels should be rendered chooser widget.
    isoThresholdSlider : qtpy.QtWidgets.QSlider
        Slider controlling the isosurface threshold value for rendering.
    isoThresholdLabel : qtpy.QtWidgets.QLabel
        Label for the isosurface threshold slider widget.
    attenuationSlider : qtpy.QtWidgets.QSlider
        Slider controlling attenuation rate for `attenuated_mip` mode.
    attenuationLabel : qtpy.QtWidgets.QLabel
        Label for the attenuation slider widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.rendering.connect(self._on_rendering_change)
        self._layer.events.iso_threshold.connect(self._on_iso_threshold_change)
        self._layer.events.contrast_limits.connect(
            self._on_contrast_limits_change
        )
        self._layer.events.attenuation.connect(self._on_attenuation_change)

        # Setup widgets
        renderComboBox = QComboBox(parent)
        rendering_options = [i.value for i in ImageRendering]
        renderComboBox.addItems(rendering_options)
        index = renderComboBox.findText(
            self._layer.rendering, Qt.MatchFlag.MatchFixedString
        )
        renderComboBox.setCurrentIndex(index)
        renderComboBox.currentTextChanged.connect(self.changeRendering)
        self.renderComboBox = renderComboBox

        self.renderLabel = QtWrappedLabel(trans._('rendering:'))

        sld = QDoubleSlider(Qt.Orientation.Horizontal, parent=parent)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        cmin, cmax = self._layer.contrast_limits_range
        sld.setMinimum(cmin)
        sld.setMaximum(cmax)
        sld.setValue(self._layer.iso_threshold)
        sld.valueChanged.connect(self.changeIsoThreshold)
        self.isoThresholdSlider = sld

        self.isoThresholdLabel = QtWrappedLabel(trans._('iso threshold:'))

        sld = QSlider(Qt.Orientation.Horizontal, parent=parent)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        sld.setValue(int(self._layer.attenuation * 200))
        sld.valueChanged.connect(self.changeAttenuation)
        self.attenuationSlider = sld

        self.attenuationLabel = QtWrappedLabel(trans._('attenuation:'))

    def changeRendering(self, text):
        """Change rendering mode for image display.

        Parameters
        ----------
        text : str
            Rendering mode used by vispy.
            Selects a preset rendering mode in vispy that determines how
            volume is displayed:
            * translucent: voxel colors are blended along the view ray until
              the result is opaque.
            * mip: maximum intensity projection. Cast a ray and display the
              maximum value that was encountered.
            * additive: voxel colors are added along the view ray until
              the result is saturated.
            * iso: isosurface. Cast a ray until a certain threshold is
              encountered. At that location, lighning calculations are
              performed to give the visual appearance of a surface.
            * attenuated_mip: attenuated maximum intensity projection. Cast a
              ray and attenuate values based on integral of encountered values,
              display the maximum value that was encountered after attenuation.
              This will make nearer objects appear more prominent.
        """
        self._layer.rendering = text
        self._update_rendering_parameter_visibility()

    def changeIsoThreshold(self, value):
        """Change isosurface threshold on the layer model.

        Parameters
        ----------
        value : float
            Threshold for isosurface.
        """
        with self._layer.events.blocker(self._on_iso_threshold_change):
            self._layer.iso_threshold = value

    def changeAttenuation(self, value):
        """Change attenuation rate for attenuated maximum intensity projection.

        Parameters
        ----------
        value : Float
            Attenuation rate for attenuated maximum intensity projection.
        """
        with self._layer.events.blocker(self._on_attenuation_change):
            self._layer.attenuation = value / 200

    def _on_rendering_change(self):
        """Receive layer model rendering change event and update dropdown menu."""
        with self._layer.events.rendering.blocker():
            index = self.renderComboBox.findText(
                self._layer.rendering, Qt.MatchFlag.MatchFixedString
            )
            self.renderComboBox.setCurrentIndex(index)
            self._update_rendering_parameter_visibility()

    def _on_contrast_limits_change(self):
        with self._layer.events.blocker(self._on_iso_threshold_change):
            cmin, cmax = self._layer.contrast_limits_range
            self.isoThresholdSlider.setMinimum(cmin)
            self.isoThresholdSlider.setMaximum(cmax)

    def _on_iso_threshold_change(self):
        """Receive layer model isosurface change event and update the slider."""
        with self._layer.events.iso_threshold.blocker():
            self.isoThresholdSlider.setValue(self._layer.iso_threshold)

    def _on_attenuation_change(self):
        """Receive layer model attenuation change event and update the slider."""
        with self._layer.events.attenuation.blocker():
            self.attenuationSlider.setValue(int(self._layer.attenuation * 200))

    def _on_display_change_hide(self):
        self.isoThresholdSlider.hide()
        self.isoThresholdLabel.hide()
        self.attenuationSlider.hide()
        self.attenuationLabel.hide()
        self.renderComboBox.hide()
        self.renderLabel.hide()

    def _on_display_change_show(self):
        self.renderComboBox.show()
        self.renderLabel.show()
        self._update_rendering_parameter_visibility()

    def _update_rendering_parameter_visibility(self):
        """Hide isosurface rendering parameters if they aren't needed."""
        rendering = ImageRendering(self._layer.rendering)
        iso_threshold_visible = rendering == ImageRendering.ISO
        self.isoThresholdLabel.setVisible(iso_threshold_visible)
        self.isoThresholdSlider.setVisible(iso_threshold_visible)
        attenuation_visible = rendering == ImageRendering.ATTENUATED_MIP
        self.attenuationSlider.setVisible(attenuation_visible)
        self.attenuationLabel.setVisible(attenuation_visible)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.renderLabel, self.renderComboBox),
            (self.isoThresholdLabel, self.isoThresholdSlider),
            (self.attenuationLabel, self.attenuationSlider),
        ]
