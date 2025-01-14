from typing import TYPE_CHECKING, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.utils.events.event_utils import connect_no_arg, connect_setattr
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.layers import Image


class AutoScaleButtons(QWidget):
    def __init__(self, layer: 'Image', parent=None) -> None:
        super().__init__(parent=parent)

        self.setLayout(QHBoxLayout())
        self.layout().setSpacing(2)
        self.layout().setContentsMargins(0, 0, 0, 0)
        once_btn = QPushButton(trans._('once'))
        once_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        auto_btn = QPushButton(trans._('continuous'))
        auto_btn.setCheckable(True)
        auto_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        once_btn.clicked.connect(lambda: auto_btn.setChecked(False))
        connect_no_arg(once_btn.clicked, layer, 'reset_contrast_limits')
        connect_setattr(auto_btn.toggled, layer, '_keep_auto_contrast')
        connect_no_arg(auto_btn.clicked, layer, 'reset_contrast_limits')

        self.layout().addWidget(once_btn)
        self.layout().addWidget(auto_btn)

        # just for testing
        self._once_btn = once_btn
        self._auto_btn = auto_btn


class QtAutoScaleControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer auto-contrast related
    functionality and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
        autoScaleBar : qtpy.QtWidgets.QWidget
            Widget to wrap push buttons related with the layer auto-contrast funtionality.
        colormapWidgetLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
            Label for the auto-contrast functionality widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)

        # Setup widgets
        self.autoScaleBar = AutoScaleButtons(layer, parent)

        self.autoScaleBarLabel = QtWrappedLabel(trans._('auto-contrast:'))

    def changeColor(self, text):
        """Change colormap on the layer model.

        Parameters
        ----------
        text : str
            Colormap name.
        """
        self._layer.colormap = self.colormapComboBox.currentData()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.autoScaleBarLabel, self.autoScaleBar)]
