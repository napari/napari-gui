from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QFormLayout, QFrame, QLabel

from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets._slider_compat import QDoubleSlider
from napari.layers.base._base_constants import BLENDING_TRANSLATIONS, Blending
from napari.layers.base.base import Layer
from napari.utils.events import disconnect_events
from napari.utils.translations import trans

# opaque and minimum blending do not support changing alpha (opacity)
NO_OPACITY_BLENDING_MODES = {str(Blending.MINIMUM), str(Blending.OPAQUE)}


class LayerFormLayout(QFormLayout):
    """Reusable form layout for subwidgets in each QtLayerControls class"""

    def __init__(self, QWidget=None) -> None:
        super().__init__(QWidget)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(4)
        self.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)


class QtLayerControls(QFrame):
    """Superclass for all the other LayerControl classes.

    This class is never directly instantiated anywhere.

    Parameters
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    blendComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select blending mode of layer.
    layer : napari.layers.Layer
        An instance of a napari layer.
    opacitySlider : qtpy.QtWidgets.QSlider
        Slider controlling opacity of the layer.
    opacityLabel : qtpy.QtWidgets.QLabel
        Label for the opacity slider widget.
    """

    def __init__(self, layer: Layer) -> None:
        super().__init__()

        self._ndisplay: int = 2

        self.layer = layer
        self.layer.events.blending.connect(self._on_blending_change)
        self.layer.events.opacity.connect(self._on_opacity_change)

        self.setObjectName('layer')
        self.setMouseTracking(True)

        self.setLayout(LayerFormLayout(self))

        sld = QDoubleSlider(Qt.Orientation.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(1)
        sld.setSingleStep(0.01)
        sld.valueChanged.connect(self.changeOpacity)
        self.opacitySlider = sld
        self.opacityLabel = QLabel(trans._('opacity:'))

        self._on_opacity_change()

        blend_comboBox = QComboBox(self)
        for index, (data, text) in enumerate(BLENDING_TRANSLATIONS.items()):
            data = data.value
            blend_comboBox.addItem(text, data)
            if data == self.layer.blending:
                blend_comboBox.setCurrentIndex(index)

        blend_comboBox.currentTextChanged.connect(self.changeBlending)
        self.blendComboBox = blend_comboBox
        # opaque and minimum blending do not support changing alpha
        self.opacitySlider.setEnabled(
            self.layer.blending not in NO_OPACITY_BLENDING_MODES
        )
        self.opacityLabel.setEnabled(
            self.layer.blending not in NO_OPACITY_BLENDING_MODES
        )

        proj_modes = [i.value for i in self.layer._projectionclass]
        if len(proj_modes) > 1:
            self.projectionComboBox = QComboBox(self)
            self.projectionComboBox.addItems(proj_modes)
            self.layer.events.projection_mode.connect(
                self._on_projection_mode_change
            )
            self.projectionComboBox.currentTextChanged.connect(
                self.changeProjectionMode
            )

    def changeOpacity(self, value):
        """Change opacity value on the layer model.

        Parameters
        ----------
        value : float
            Opacity value for shapes.
            Input range 0 - 100 (transparent to fully opaque).
        """
        with self.layer.events.blocker(self._on_opacity_change):
            self.layer.opacity = value

    def changeBlending(self, text):
        """Change blending mode on the layer model.

        Parameters
        ----------
        text : str
            Name of blending mode, eg: 'translucent', 'additive', 'opaque'.
        """
        self.layer.blending = self.blendComboBox.currentData()
        # opaque and minimum blending do not support changing alpha
        self.opacitySlider.setEnabled(
            self.layer.blending not in NO_OPACITY_BLENDING_MODES
        )
        self.opacityLabel.setEnabled(
            self.layer.blending not in NO_OPACITY_BLENDING_MODES
        )

        blending_tooltip = ''
        if self.layer.blending == str(Blending.MINIMUM):
            blending_tooltip = trans._(
                '`minimum` blending mode works best with inverted colormaps with a white background.',
            )
        self.blendComboBox.setToolTip(blending_tooltip)
        self.layer.help = blending_tooltip

    def changeProjectionMode(self, text):
        with self.layer.events.blocker(self._on_projection_mode_change):
            self.layer.projection_mode = text

    def _on_opacity_change(self):
        """Receive layer model opacity change event and update opacity slider."""
        with self.layer.events.opacity.blocker():
            self.opacitySlider.setValue(self.layer.opacity)

    def _on_blending_change(self):
        """Receive layer model blending mode change event and update slider."""
        with self.layer.events.blending.blocker():
            self.blendComboBox.setCurrentIndex(
                self.blendComboBox.findData(self.layer.blending)
            )

    def _on_projection_mode_change(self):
        with qt_signals_blocked(self.projectionComboBox):
            self.projectionComboBox.setCurrentText(
                str(self.layer.projection_mode)
            )

    @property
    def ndisplay(self) -> int:
        """The number of dimensions displayed in the canvas."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay: int) -> None:
        self._ndisplay = ndisplay
        self._on_ndisplay_changed()

    def _on_ndisplay_changed(self) -> None:
        """Respond to a change to the number of dimensions displayed in the viewer.

        This is needed because some layer controls may have options that are specific
        to 2D or 3D visualization only.
        """

    def deleteLater(self):
        disconnect_events(self.layer.events, self)
        super().deleteLater()

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.events, self)
        for child in self.children():
            close_method = getattr(child, 'close', None)
            if close_method is not None:
                close_method()
        return super().close()
