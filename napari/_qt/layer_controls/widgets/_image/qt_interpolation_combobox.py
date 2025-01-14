from typing import Optional

from qtpy.QtWidgets import (
    QComboBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers.base.base import Layer
from napari.layers.image._image_constants import Interpolation
from napari.utils.translations import trans


class QtInterpolationComboBoxControl(QtWidgetControlsBase):
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
    interpComboBox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current shading value of the layer.
    interpComboBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the shading value chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.interpolation2d.connect(
            self._on_interpolation_change
        )
        self._layer.events.interpolation3d.connect(
            self._on_interpolation_change
        )

        # Setup widgets
        self.interpComboBox = QComboBox(parent)
        self.interpComboBox.currentTextChanged.connect(
            self.changeInterpolation
        )
        self.interpComboBox.setToolTip(
            trans._(
                'Texture interpolation for display.\nnearest and linear are most performant.'
            )
        )

        self.interpComboBoxLabel = QtWrappedLabel(trans._('interpolation:'))

    def changeInterpolation(self, text: str) -> None:
        """Change interpolation mode for image display.

        Parameters
        ----------
        text : str
            Interpolation mode used by vispy. Must be one of our supported
            modes:
            'bessel', 'bicubic', 'linear', 'blackman', 'catrom', 'gaussian',
            'hamming', 'hanning', 'hermite', 'kaiser', 'lanczos', 'mitchell',
            'nearest', 'spline16', 'spline36'
        """
        # TODO: Better way to handle the ndisplay value?
        if self.parent().ndisplay == 2:
            self._layer.interpolation2d = text
        else:
            self._layer.interpolation3d = text

    def _on_interpolation_change(self, event) -> None:
        """Receive layer interpolation change event and update dropdown menu.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        interp_string = event.value.value

        with (
            self._layer.events.interpolation.blocker(),
            self._layer.events.interpolation2d.blocker(),
            self._layer.events.interpolation3d.blocker(),
        ):
            if self.interpComboBox.findText(interp_string) == -1:
                self.interpComboBox.addItem(interp_string)
            self.interpComboBox.setCurrentText(interp_string)

    def _update_interpolation_combo(self, ndisplay: int) -> None:
        interp_names = [i.value for i in Interpolation.view_subset()]
        interp = (
            self._layer.interpolation2d
            if ndisplay == 2
            else self._layer.interpolation3d
        )
        with qt_signals_blocked(self.interpComboBox):
            self.interpComboBox.clear()
            self.interpComboBox.addItems(interp_names)
            self.interpComboBox.setCurrentText(interp)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.interpComboBoxLabel, self.interpComboBox)]
