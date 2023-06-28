from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)
from superqt import QLargeIntSpinBox

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.utils import set_widgets_enabled_with_opacity
from napari._qt.widgets._slider_compat import QSlider
from napari._qt.widgets.qt_mode_buttons import (
    QtModePushButton,
    QtModeRadioButton,
)
from napari.layers.labels._labels_constants import (
    LABEL_COLOR_MODE_TRANSLATIONS,
    LabelsRendering,
    Mode,
)
from napari.layers.labels._labels_utils import get_dtype
from napari.utils._dtype import get_dtype_limits
from napari.utils.action_manager import action_manager
from napari.utils.events import disconnect_events
from napari.utils.translations import trans

if TYPE_CHECKING:
    import napari.layers


INT32_MAX = 2**31 - 1


class QtLabelsControls(QtLayerControls):
    """Qt view and controls for the napari Labels layer.

    Parameters
    ----------
    layer : napari.layers.Labels
        An instance of a napari Labels layer.

    Attributes
    ----------
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group of labels layer modes: PAN_ZOOM, PICKER, PAINT, ERASE, or
        FILL.
    colormapUpdate : qtpy.QtWidgets.QPushButton
        Button to update colormap of label layer.
    contigCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if label layer is contiguous.
    fill_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select FILL mode on Labels layer.
    layer : napari.layers.Labels
        An instance of a napari Labels layer.
    ndimSpinBox : qtpy.QtWidgets.QSpinBox
        Spinbox to control the number of editable dimensions of label layer.
    paint_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select PAINT mode on Labels layer.
    panzoom_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select PAN_ZOOM mode on Labels layer.
    pick_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select PICKER mode on Labels layer.
    preserveLabelsCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if existing labels are preserved
    erase_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select ERASE mode on Labels layer.
    selectionSpinBox : superqt.QLargeIntSpinBox
        Widget to select a specific label by its index.
        N.B. cannot represent labels > 2**53.
    selectedColorCheckbox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if only currently selected label is shown.

    Raises
    ------
    ValueError
        Raise error if label mode is not PAN_ZOOM, PICKER, PAINT, ERASE, or
        FILL.
    """

    layer: 'napari.layers.Labels'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        self.layer.events.mode.connect(self._on_mode_change)
        self.layer.events.rendering.connect(self._on_rendering_change)
        self.layer.events.brush_size.connect(self._on_brush_size_change)
        self.layer.events.contiguous.connect(self._on_contiguous_change)
        self.layer.events.n_edit_dimensions.connect(
            self._on_n_edit_dimensions_change
        )
        self.layer.events.contour.connect(self._on_contour_change)
        self.layer.events.editable.connect(self._on_editable_or_visible_change)
        self.layer.events.visible.connect(self._on_editable_or_visible_change)
        self.layer.events.preserve_labels.connect(
            self._on_preserve_labels_change
        )
        self.layer.events.show_selected_label.connect(
            self._on_show_selected_label_change
        )
        self.layer.events.color_mode.connect(self._on_color_mode_change)
        self.layer.events.predefined_labels.connect(
            self._on_predefined_labels_change
        )

        dtype_lims = get_dtype_limits(get_dtype(layer))
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(1)
        sld.setMaximum(40)
        sld.setSingleStep(1)
        sld.valueChanged.connect(self.changeSize)
        self.brushSizeSlider = sld
        self._on_brush_size_change()

        contig_cb = QCheckBox()
        contig_cb.setToolTip(trans._('contiguous editing'))
        contig_cb.stateChanged.connect(self.change_contig)
        self.contigCheckBox = contig_cb
        self._on_contiguous_change()

        ndim_sb = QSpinBox()
        self.ndimSpinBox = ndim_sb
        ndim_sb.setToolTip(trans._('number of dimensions for label editing'))
        ndim_sb.valueChanged.connect(self.change_n_edit_dim)
        ndim_sb.setMinimum(2)
        ndim_sb.setMaximum(self.layer.ndim)
        ndim_sb.setSingleStep(1)
        ndim_sb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._on_n_edit_dimensions_change()

        self.contourSpinBox = QLargeIntSpinBox()
        self.contourSpinBox.setRange(0, dtype_lims[1])
        self.contourSpinBox.setToolTip(trans._('display contours of labels'))
        self.contourSpinBox.valueChanged.connect(self.change_contour)
        self.contourSpinBox.setKeyboardTracking(False)
        self.contourSpinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._on_contour_change()

        preserve_labels_cb = QCheckBox()
        preserve_labels_cb.setToolTip(
            trans._('preserve existing labels while painting')
        )
        preserve_labels_cb.stateChanged.connect(self.change_preserve_labels)
        self.preserveLabelsCheckBox = preserve_labels_cb
        self._on_preserve_labels_change()

        selectedColorCheckbox = QCheckBox()
        selectedColorCheckbox.setToolTip(
            trans._("Display only selected label")
        )
        selectedColorCheckbox.stateChanged.connect(self.toggle_selected_mode)
        self.selectedColorCheckbox = selectedColorCheckbox
        self._on_show_selected_label_change()

        # shuffle colormap button
        self.colormapUpdate = QtModePushButton(
            layer,
            'shuffle',
            slot=self.changeColor,
            tooltip=trans._('shuffle colors'),
        )

        self.panzoom_button = QtModeRadioButton(
            layer,
            'pan',
            Mode.PAN_ZOOM,
            checked=True,
        )
        action_manager.bind_button(
            'napari:activate_labels_pan_zoom_mode', self.panzoom_button
        )

        self.pick_button = QtModeRadioButton(layer, 'picker', Mode.PICK)
        action_manager.bind_button(
            'napari:activate_labels_picker_mode', self.pick_button
        )

        self.paint_button = QtModeRadioButton(layer, 'paint', Mode.PAINT)
        action_manager.bind_button(
            'napari:activate_labels_paint_mode', self.paint_button
        )

        self.fill_button = QtModeRadioButton(
            layer,
            'fill',
            Mode.FILL,
        )
        action_manager.bind_button(
            'napari:activate_labels_fill_mode',
            self.fill_button,
        )

        self.erase_button = QtModeRadioButton(
            layer,
            'erase',
            Mode.ERASE,
        )
        action_manager.bind_button(
            'napari:activate_labels_erase_mode',
            self.erase_button,
        )

        # don't bind with action manager as this would remove "Toggle with {shortcut}"

        self._EDIT_BUTTONS = (
            self.paint_button,
            self.pick_button,
            self.fill_button,
            self.erase_button,
        )

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.panzoom_button)
        self.button_group.addButton(self.paint_button)
        self.button_group.addButton(self.pick_button)
        self.button_group.addButton(self.fill_button)
        self.button_group.addButton(self.erase_button)
        self._on_editable_or_visible_change()

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.colormapUpdate)
        button_row.addWidget(self.erase_button)
        button_row.addWidget(self.paint_button)
        button_row.addWidget(self.fill_button)
        button_row.addWidget(self.pick_button)
        button_row.addWidget(self.panzoom_button)
        button_row.setSpacing(4)
        button_row.setContentsMargins(0, 0, 0, 5)

        renderComboBox = QComboBox(self)
        rendering_options = [i.value for i in LabelsRendering]
        renderComboBox.addItems(rendering_options)
        index = renderComboBox.findText(
            self.layer.rendering, Qt.MatchFlag.MatchFixedString
        )
        renderComboBox.setCurrentIndex(index)
        renderComboBox.currentTextChanged.connect(self.changeRendering)
        self.renderComboBox = renderComboBox
        self.renderLabel = QLabel(trans._('rendering:'))

        self._on_ndisplay_changed()

        color_mode_comboBox = QComboBox(self)
        for index, (data, text) in enumerate(
            LABEL_COLOR_MODE_TRANSLATIONS.items()
        ):
            data = data.value
            color_mode_comboBox.addItem(text, data)

            if self.layer.color_mode == data:
                color_mode_comboBox.setCurrentIndex(index)

        color_mode_comboBox.activated.connect(self.change_color_mode)
        self.colorModeComboBox = color_mode_comboBox
        self._on_color_mode_change()

        self.labelsSpinbox = QtLabelsSpinBox(layer, self)
        self.labelsCombobox = QtLabelsCombobox(layer, self)

        if layer.predefined_labels is None:
            self.labelsCombobox.setVisible(False)
            labels_selection_widget = self.labelsSpinbox
        else:
            self.labelsSpinbox.setVisible(False)
            labels_selection_widget = self.labelsCombobox

        labels_selection_widget.connect_to_layer()

        self.layout().addRow(button_row)
        self.layout().addRow(trans._('label:'), labels_selection_widget)
        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(trans._('brush size:'), self.brushSizeSlider)
        self.layout().addRow(trans._('blending:'), self.blendComboBox)
        self.layout().addRow(self.renderLabel, self.renderComboBox)
        self.layout().addRow(trans._('color mode:'), self.colorModeComboBox)
        self.layout().addRow(trans._('contour:'), self.contourSpinBox)
        self.layout().addRow(trans._('n edit dim:'), self.ndimSpinBox)
        self.layout().addRow(trans._('contiguous:'), self.contigCheckBox)
        self.layout().addRow(
            trans._('preserve\nlabels:'), self.preserveLabelsCheckBox
        )
        self.layout().addRow(
            trans._('show\nselected:'), self.selectedColorCheckbox
        )

    def _on_mode_change(self, event):
        """Receive layer model mode change event and update checkbox ticks.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not PAN_ZOOM, PICK, PAINT, ERASE, or
            FILL
        """
        mode = event.mode
        if mode == Mode.PAN_ZOOM:
            self.panzoom_button.setChecked(True)
        elif mode == Mode.PICK:
            self.pick_button.setChecked(True)
        elif mode == Mode.PAINT:
            self.paint_button.setChecked(True)
        elif mode == Mode.FILL:
            self.fill_button.setChecked(True)
        elif mode == Mode.ERASE:
            self.erase_button.setChecked(True)
        elif mode != Mode.TRANSFORM:
            raise ValueError(trans._("Mode not recognized"))

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
            * iso_categorical: isosurface for categorical data (e.g., labels).
              Cast a ray until a value greater than zero is encountered. At that
              location, lighning calculations are performed to give the visual
              appearance of a surface.
        """
        self.layer.rendering = text

    def changeColor(self):
        """Change colormap of the label layer."""
        self.layer.new_colormap()

    def toggle_selected_mode(self, state):
        """Toggle display of selected label only.

        Parameters
        ----------
        state : int
            Integer value of Qt.CheckState that indicates the check state of selectedColorCheckbox
        """
        self.layer.show_selected_label = (
            Qt.CheckState(state) == Qt.CheckState.Checked
        )

    def changeSize(self, value):
        """Change paint brush size.

        Parameters
        ----------
        value : float
            Size of the paint brush.
        """
        self.layer.brush_size = value

    def change_contig(self, state):
        """Toggle contiguous state of label layer.

        Parameters
        ----------
        state : int
            Integer value of Qt.CheckState that indicates the check state of contigCheckBox
        """
        self.layer.contiguous = Qt.CheckState(state) == Qt.CheckState.Checked

    def change_n_edit_dim(self, value):
        """Change the number of editable dimensions of label layer.

        Parameters
        ----------
        value : int
            The number of editable dimensions to set.
        """
        self.layer.n_edit_dimensions = value
        self.ndimSpinBox.clearFocus()
        self.setFocus()

    def change_contour(self, value):
        """Change contour thickness.

        Parameters
        ----------
        value : int
            Thickness of contour.
        """
        self.layer.contour = value
        self.contourSpinBox.clearFocus()
        self.setFocus()

    def change_preserve_labels(self, state):
        """Toggle preserve_labels state of label layer.

        Parameters
        ----------
        state : int
            Integer value of Qt.CheckState that indicates the check state of preserveLabelsCheckBox
        """
        self.layer.preserve_labels = (
            Qt.CheckState(state) == Qt.CheckState.Checked
        )

    def change_color_mode(self):
        """Change color mode of label layer"""
        self.layer.color_mode = self.colorModeComboBox.currentData()

    def _on_contour_change(self):
        """Receive layer model contour value change event and update spinbox."""
        with self.layer.events.contour.blocker():
            value = self.layer.contour
            self.contourSpinBox.setValue(value)

    def _on_brush_size_change(self):
        """Receive layer model brush size change event and update the slider."""
        with self.layer.events.brush_size.blocker():
            value = self.layer.brush_size
            value = np.maximum(1, int(value))
            if value > self.brushSizeSlider.maximum():
                self.brushSizeSlider.setMaximum(int(value))
            self.brushSizeSlider.setValue(value)

    def _on_n_edit_dimensions_change(self):
        """Receive layer model n-dim mode change event and update the checkbox."""
        with self.layer.events.n_edit_dimensions.blocker():
            value = self.layer.n_edit_dimensions
            self.ndimSpinBox.setValue(int(value))

    def _on_contiguous_change(self):
        """Receive layer model contiguous change event and update the checkbox."""
        with self.layer.events.contiguous.blocker():
            self.contigCheckBox.setChecked(self.layer.contiguous)

    def _on_preserve_labels_change(self):
        """Receive layer model preserve_labels event and update the checkbox."""
        with self.layer.events.preserve_labels.blocker():
            self.preserveLabelsCheckBox.setChecked(self.layer.preserve_labels)

    def _on_show_selected_label_change(self):
        """Receive layer model show_selected_labels event and update the checkbox."""
        with self.layer.events.show_selected_label.blocker():
            self.selectedColorCheckbox.setChecked(
                self.layer.show_selected_label
            )

    def _on_color_mode_change(self):
        """Receive layer model color."""
        with self.layer.events.color_mode.blocker():
            self.colorModeComboBox.setCurrentIndex(
                self.colorModeComboBox.findData(self.layer.color_mode)
            )

    def _on_editable_or_visible_change(self):
        """Receive layer model editable/visible change event & enable/disable buttons."""
        set_widgets_enabled_with_opacity(
            self,
            self._EDIT_BUTTONS,
            self.layer.editable and self.layer.visible,
        )

    def _on_rendering_change(self):
        """Receive layer model rendering change event and update dropdown menu."""
        with self.layer.events.rendering.blocker():
            index = self.renderComboBox.findText(
                self.layer.rendering, Qt.MatchFlag.MatchFixedString
            )
            self.renderComboBox.setCurrentIndex(index)

    def _on_ndisplay_changed(self):
        render_visible = self.ndisplay == 3
        self.renderComboBox.setVisible(render_visible)
        self.renderLabel.setVisible(render_visible)
        self._on_editable_or_visible_change()

    def _on_predefined_labels_change(self):
        labels_combobox_activated = (
            self.layout().indexOf(self.labelsCombobox) != -1
        )
        predefined_labels = self.layer.predefined_labels is not None

        if not predefined_labels and labels_combobox_activated:
            self.layout().replaceWidget(
                self.labelsCombobox, self.labelsSpinbox
            )
            disconnect_events(self.layer.events, self.labelsCombobox)
            self.labelsCombobox.setVisible(False)
            self.labelsSpinbox.setVisible(True)
            self.labelsSpinbox.connect_to_layer()
        elif predefined_labels and not labels_combobox_activated:
            self.layout().replaceWidget(
                self.labelsSpinbox, self.labelsCombobox
            )
            disconnect_events(self.layer.events, self.labelsSpinbox)
            self.labelsSpinbox.setVisible(False)
            self.labelsCombobox.setVisible(True)
            self.labelsCombobox.connect_to_layer()

    def deleteLater(self):
        self.labelsSpinbox.deleteLater()
        self.labelsCombobox.deleteLater()
        super().deleteLater()


class QtLabelsSpinBox(QWidget):
    def __init__(self, layer, parent=None) -> None:
        super().__init__(parent=parent)

        self.layer = layer
        self.colorBox = QtColorBox()

        dtype_lims = get_dtype_limits(get_dtype(layer))
        self.selectionSpinBox = QLargeIntSpinBox()
        self.selectionSpinBox.setRange(*dtype_lims)
        self.selectionSpinBox.setKeyboardTracking(False)
        self.selectionSpinBox.valueChanged.connect(self._on_selection_change)
        self.selectionSpinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QHBoxLayout()
        layout.addWidget(self.colorBox)
        layout.addWidget(self.selectionSpinBox)
        layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(layout)

    def connect_to_layer(self):
        self.layer.events.selected_label.connect(
            self._on_selected_label_change
        )
        self.layer.events.colormap.connect(self._on_colormap_change)
        self._on_selected_label_change()

    def _on_selected_label_change(self):
        """Receive layer model label selection change event and update spinbox."""
        with self.layer.events.selected_label.blocker():
            value = self.layer.selected_label
            self.selectionSpinBox.setValue(value)
        self.colorBox.set_color(self.layer._selected_color)

    def _on_colormap_change(self):
        """Receive label colormap change event & update colorbox."""
        color = self.layer.get_color(self.layer.selected_label)
        self.colorBox.set_color(color)

    def _on_selection_change(self, value):
        """Change currently selected label.

        Parameters
        ----------
        value : int
            Index of label to select.
        """
        self.layer.selected_label = value
        self.selectionSpinBox.clearFocus()
        if self.parent():
            self.parent().setFocus()

    def deleteLater(self):
        disconnect_events(self.layer.events, self)
        super().deleteLater()

    def closeEvent(self, event):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.events, self)
        super().closeEvent(event)


class QtLabelsCombobox(QComboBox):
    def __init__(self, layer, parent=None) -> None:
        super().__init__(parent=parent)

        self.layer = layer
        self._height = 24
        self._last_seed = -1
        self.setFixedHeight(self._height)

        self.currentIndexChanged.connect(self._on_current_index_changed)
        self.activated.connect(self._on_activated)

    def update_items(self):
        if self.layer.predefined_labels is None:
            return
        self.blockSignals(True)
        self._last_seed = self.layer.seed

        self._labels_list = sorted(self.layer.predefined_labels)
        # Initialize color palette
        self.layer.get_color(max(self._labels_list))

        for i, label in enumerate(self._labels_list):
            if i >= self.count():
                self.addItem("")

            color = self.layer.get_color(label)

            color_pixmap = QPixmap(self._height, self._height)

            if color is None:
                paint_checkerboard(QPainter(color_pixmap), self._height)
            else:
                color = np.round(255 * color[:3]).astype(int)
                color_pixmap.fill(QColor(*color.tolist()))

            color_icon = QIcon(color_pixmap)
            name = self.layer.get_label_name(label)
            item_text = str(label) + (": " + name if name else "")

            self.setItemIcon(i, color_icon)
            self.setItemText(i, item_text)

        for _ in range(self.count() - len(self._labels_list)):
            self.removeItem(self.count() - 1)

        self.blockSignals(False)

    def connect_to_layer(self):
        self.layer.events.predefined_labels.connect(self.update_items)
        self.layer.events.colormap.connect(self.update_items)
        self.layer.events.selected_label.connect(
            self._on_selected_label_change
        )
        self._on_selected_label_change()

    def _on_selected_label_change(self):
        if (
            not np.isclose(self._last_seed, self.layer.seed)
            or self.layer.selected_label not in self._labels_list
        ):
            self.update_items()

        item_index = self._labels_list.index(self.layer.selected_label)
        with self.layer.events.selected_label.blocker():
            self.setCurrentIndex(item_index)

    def _on_current_index_changed(self):
        index = self.currentIndex()
        self.layer.selected_label = self._labels_list[index]

    def _on_activated(self):
        self.clearFocus()
        if self.parent():
            self.parent().setFocus()

    def deleteLater(self):
        disconnect_events(self.layer.events, self)
        super().deleteLater()

    def closeEvent(self, event):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.events, self)
        super().closeEvent(event)


class QtColorBox(QWidget):
    """A widget that shows a square with the current label color.

    Parameters
    ----------
    size : int
        A size of the color box.
    """

    def __init__(self, size: int = 24) -> None:
        super().__init__()

        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._height = size
        self.setFixedWidth(self._height)
        self.setFixedHeight(self._height)
        self.setToolTip(trans._('Selected label color'))

        self._color = None

    def set_color(self, color):
        self._color = color
        self.update()

    def paintEvent(self, event):
        """Paint the colorbox.  If no color, display a checkerboard pattern.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        painter = QPainter(self)
        if self._color is None:
            paint_checkerboard(painter, self._height)
        else:
            color = np.round(255 * self._color).astype(int)
            painter.setPen(QColor(*list(color)))
            painter.setBrush(QColor(*list(color)))
            painter.drawRect(0, 0, self._height, self._height)


def paint_checkerboard(painter: QPainter, height: int) -> None:
    for i in range(height // 4):
        for j in range(height // 4):
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                painter.setPen(QColor(230, 230, 230))
                painter.setBrush(QColor(230, 230, 230))
            else:
                painter.setPen(QColor(25, 25, 25))
                painter.setBrush(QColor(25, 25, 25))
            painter.drawRect(i * 4, j * 4, 5, 5)
