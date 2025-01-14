from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QWidget,
)
from superqt import QLabeledDoubleSlider

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.layers.image._image_constants import VolumeDepiction
from napari.utils.action_manager import action_manager
from napari.utils.translations import trans


class PlaneNormalButtons(QWidget):
    """Qt buttons for controlling plane orientation.

    Attributes
    ----------
    xButton : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the x axis.
    yButton : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the y axis.
    zButton : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the z axis.
    obliqueButton : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the camera view direction.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setLayout(QHBoxLayout())
        self.layout().setSpacing(2)
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.xButton = QPushButton('x')
        self.yButton = QPushButton('y')
        self.zButton = QPushButton('z')
        self.obliqueButton = QPushButton(trans._('oblique'))
        action_manager.bind_button(
            'napari:orient_plane_normal_along_z',
            self.zButton,
        )
        action_manager.bind_button(
            'napari:orient_plane_normal_along_y',
            self.yButton,
        )
        action_manager.bind_button(
            'napari:orient_plane_normal_along_x',
            self.xButton,
        )
        action_manager.bind_button(
            'napari:orient_plane_normal_along_view_direction_no_gen',
            self.obliqueButton,
        )

        self.layout().addWidget(self.xButton)
        self.layout().addWidget(self.yButton)
        self.layout().addWidget(self.zButton)
        self.layout().addWidget(self.obliqueButton)


class QtDepictionControl(QtWidgetControlsBase):
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
    depictionComboBox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current depiction value of the layer.
    depictionLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the depiction value chooser widget.
    planeNormalButtons : PlaneNormalButtons
        Buttons controlling plane normal orientation when the `plane` depiction value is choosed.
    planeNormalLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the plane normal value chooser widget.
    planeThicknessSlider : superqt.QLabeledDoubleSlider
        Slider controlling plane normal thickness when the `plane` depiction value is choosed.
    planeThicknessLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the plane normal thickness value chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.depiction.connect(self._on_depiction_change)
        self._layer.plane.events.thickness.connect(
            self._on_plane_thickness_change
        )

        # Setup widgets
        self.depictionComboBox = QComboBox(parent)
        depiction_options = [d.value for d in VolumeDepiction]
        self.depictionComboBox.addItems(depiction_options)
        index = self.depictionComboBox.findText(
            self._layer.depiction, Qt.MatchFlag.MatchFixedString
        )
        self.depictionComboBox.setCurrentIndex(index)
        self.depictionComboBox.currentTextChanged.connect(self.changeDepiction)
        self.depictionLabel = QtWrappedLabel(trans._('depiction:'))

        # plane controls
        self.planeNormalButtons = PlaneNormalButtons(parent)
        self.planeNormalLabel = QtWrappedLabel(trans._('plane normal:'))

        self.planeThicknessSlider = QLabeledDoubleSlider(
            Qt.Orientation.Horizontal, parent
        )
        self.planeThicknessSlider.setFocusPolicy(Qt.NoFocus)
        self.planeThicknessSlider.setMinimum(1)
        self.planeThicknessSlider.setMaximum(50)
        self.planeThicknessSlider.setValue(self._layer.plane.thickness)
        self.planeThicknessSlider.valueChanged.connect(
            self.changePlaneThickness
        )
        self.planeThicknessLabel = QtWrappedLabel(trans._('plane thickness:'))

    def changeDepiction(self, text: str) -> None:
        self._layer.depiction = text
        self._update_plane_parameter_visibility()

    def changePlaneThickness(self, value: float) -> None:
        self._layer.plane.thickness = value

    def _on_depiction_change(self) -> None:
        """Receive layer model depiction change event and update combobox."""
        with self._layer.events.depiction.blocker():
            index = self.depictionComboBox.findText(
                self._layer.depiction, Qt.MatchFlag.MatchFixedString
            )
            self.depictionComboBox.setCurrentIndex(index)
            self._update_plane_parameter_visibility()

    def _on_plane_thickness_change(self) -> None:
        with self._layer.plane.events.blocker():
            self.planeThicknessSlider.setValue(self._layer.plane.thickness)

    def _on_display_change_hide(self) -> None:
        self.depictionComboBox.hide()
        self.depictionLabel.hide()

    def _on_display_change_show(self) -> None:
        self.depictionComboBox.show()
        self.depictionLabel.show()

    def _update_plane_parameter_visibility(self) -> None:
        """Hide plane rendering controls if they aren't needed."""
        depiction = VolumeDepiction(self._layer.depiction)
        # TODO: Better way to handle the ndisplay value?
        visible = (
            depiction == VolumeDepiction.PLANE
            and self.parent().ndisplay == 3
            and self._layer.ndim >= 3
        )
        self.planeNormalButtons.setVisible(visible)
        self.planeNormalLabel.setVisible(visible)
        self.planeThicknessSlider.setVisible(visible)
        self.planeThicknessLabel.setVisible(visible)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.depictionLabel, self.depictionComboBox),
            (self.planeNormalLabel, self.planeNormalButtons),
            (self.planeThicknessLabel, self.planeThicknessSlider),
        ]
