from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.widgets._slider_compat import QDoubleSlider
from napari.layers.base._base_constants import BLENDING_TRANSLATIONS, Blending
from napari.layers.base.base import Layer
from napari.utils.translations import trans

# opaque and minimum blending do not support changing alpha (opacity)
NO_OPACITY_BLENDING_MODES = {str(Blending.MINIMUM), str(Blending.OPAQUE)}


class QtOpacityBlendingControls(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between opacity/blending
    layer attributes and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    blendComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select blending mode of layer.
    blendLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the blending combobox widget.
    opacitySlider : qtpy.QtWidgets.QSlider
        Slider controlling opacity of the layer.
    opacityLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the opacity slider widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.blending.connect(self._on_blending_change)
        self._layer.events.opacity.connect(self._on_opacity_change)

        # Setup widgets
        sld = QDoubleSlider(Qt.Orientation.Horizontal, parent=parent)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(1)
        sld.setSingleStep(0.01)
        sld.valueChanged.connect(self.changeOpacity)
        self.opacitySlider = sld
        self.opacityLabel = QtWrappedLabel(trans._('opacity:'))
        self._on_opacity_change()

        blend_comboBox = QComboBox(parent)
        for index, (data, text) in enumerate(BLENDING_TRANSLATIONS.items()):
            data = data.value
            blend_comboBox.addItem(text, data)
            if data == self._layer.blending:
                blend_comboBox.setCurrentIndex(index)

        blend_comboBox.currentTextChanged.connect(self.changeBlending)
        self.blendComboBox = blend_comboBox
        self.blendLabel = QtWrappedLabel(trans._('blending:'))

        # opaque and minimum blending do not support changing alpha
        self.opacitySlider.setEnabled(
            self._layer.blending not in NO_OPACITY_BLENDING_MODES
        )
        self.opacityLabel.setEnabled(
            self._layer.blending not in NO_OPACITY_BLENDING_MODES
        )

    def changeOpacity(self, value: float) -> None:
        """Change opacity value on the layer model.

        Parameters
        ----------
        value : float
            Opacity value for shapes.
            Input range 0 - 100 (transparent to fully opaque).
        """
        with self._layer.events.blocker(self._on_opacity_change):
            self._layer.opacity = value

    def changeBlending(self, text: str) -> None:
        """Change blending mode on the layer model.

        Parameters
        ----------
        text : str
            Name of blending mode, eg: 'translucent', 'additive', 'opaque'.
        """
        self._layer.blending = self.blendComboBox.currentData()
        # opaque and minimum blending do not support changing alpha
        self.opacitySlider.setEnabled(
            self._layer.blending not in NO_OPACITY_BLENDING_MODES
        )
        self.opacityLabel.setEnabled(
            self._layer.blending not in NO_OPACITY_BLENDING_MODES
        )

        blending_tooltip = ''
        if self._layer.blending == str(Blending.MINIMUM):
            blending_tooltip = trans._(
                '`minimum` blending mode works best with inverted colormaps with a white background.',
            )
        self.blendComboBox.setToolTip(blending_tooltip)
        self._layer.help = blending_tooltip

    def _on_opacity_change(self) -> None:
        """
        Receive layer model opacity change event and update opacity slider.
        """
        with self._layer.events.opacity.blocker():
            self.opacitySlider.setValue(self._layer.opacity)

    def _on_blending_change(self) -> None:
        """Receive layer model blending mode change event and update slider."""
        with self._layer.events.blending.blocker():
            self.blendComboBox.setCurrentIndex(
                self.blendComboBox.findData(self._layer.blending)
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.opacityLabel, self.opacitySlider),
            (self.blendLabel, self.blendComboBox),
        ]
