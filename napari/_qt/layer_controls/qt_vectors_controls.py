from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLabel

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers.utils._color_manager_constants import ColorMode
from napari.layers.vectors._vectors_constants import VECTORSTYLE_TRANSLATIONS
from napari.utils.translations import trans

if TYPE_CHECKING:
    import napari.layers


class QtVectorsControls(QtLayerControls):
    """Qt view and controls for the napari Vectors layer.

    Parameters
    ----------
    layer : napari.layers.Vectors
        An instance of a napari Vectors layer.

    Attributes
    ----------
    edge_color_label : qtpy.QtWidgets.QLabel
        Label for edgeColorSwatch
    edgeColorEdit : QColorSwatchEdit
        Widget to select display color for vectors.
    vector_style_comboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select vector_style for the vectors.
    color_mode_comboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select edge_color_mode for the vectors.
    color_prop_box : qtpy.QtWidgets.QComboBox
        Dropdown widget to select _edge_color_property for the vectors.
    edge_prop_label : qtpy.QtWidgets.QLabel
        Label for color_prop_box
    layer : napari.layers.Vectors
        An instance of a napari Vectors layer.
    outOfSliceCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to indicate whether to render out of slice.
    lengthSpinBox : qtpy.QtWidgets.QDoubleSpinBox
        Spin box widget controlling line length of vectors.
        Multiplicative factor on projections for length of all vectors.
    widthSpinBox : qtpy.QtWidgets.QDoubleSpinBox
        Spin box widget controlling edge line width of vectors.
    vector_style_comboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select vector_style for the vectors.
    """

    layer: 'napari.layers.Vectors'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        # dropdown to select the property for mapping edge_color
        color_properties = self._get_property_values()
        self.color_prop_box = QComboBox(self)
        self.color_prop_box.currentTextChanged.connect(
            self.change_edge_color_property
        )
        self.color_prop_box.addItems(color_properties)

        self.edge_prop_label = QLabel(trans._('edge property:'))

        # vector direct color mode adjustment and widget
        self.edgeColorEdit = QColorSwatchEdit(
            initial_color=self.layer.edge_color,
            tooltip=trans._(
                'click to set current edge color',
            ),
        )
        self.edgeColorEdit.color_changed.connect(self.change_edge_color_direct)
        self.edge_color_label = QLabel(trans._('edge color:'))
        self._on_edge_color_change()

        # dropdown to select the edge display vector_style
        vector_style_comboBox = QComboBox(self)
        for index, (data, text) in enumerate(VECTORSTYLE_TRANSLATIONS.items()):
            data = data.value
            vector_style_comboBox.addItem(text, data)
            if data == self.layer.vector_style:
                vector_style_comboBox.setCurrentIndex(index)

        self.vector_style_comboBox = vector_style_comboBox
        self.vector_style_comboBox.currentTextChanged.connect(
            self.change_vector_style
        )

        # dropdown to select the edge color mode
        self.color_mode_comboBox = QComboBox(self)
        color_modes = [e.value for e in ColorMode]
        self.color_mode_comboBox.addItems(color_modes)
        self.color_mode_comboBox.currentTextChanged.connect(
            self.change_edge_color_mode
        )
        self._on_edge_color_mode_change()

        # line width in pixels
        self.widthSpinBox = QDoubleSpinBox()
        self.widthSpinBox.setKeyboardTracking(False)
        self.widthSpinBox.setSingleStep(0.1)
        self.widthSpinBox.setMinimum(0.1)
        self.widthSpinBox.setMaximum(np.inf)
        self.widthSpinBox.setValue(self.layer.edge_width)
        self.widthSpinBox.valueChanged.connect(self.change_width)

        # line length
        self.lengthSpinBox = QDoubleSpinBox()
        self.lengthSpinBox.setKeyboardTracking(False)
        self.lengthSpinBox.setSingleStep(0.1)
        self.lengthSpinBox.setValue(self.layer.length)
        self.lengthSpinBox.setMinimum(0.1)
        self.lengthSpinBox.setMaximum(np.inf)
        self.lengthSpinBox.valueChanged.connect(self.change_length)

        out_of_slice_cb = QCheckBox()
        out_of_slice_cb.setToolTip(trans._('Out of slice display'))
        out_of_slice_cb.setChecked(self.layer.out_of_slice_display)
        out_of_slice_cb.stateChanged.connect(self.change_out_of_slice)
        self.outOfSliceCheckBox = out_of_slice_cb

        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(trans._('width:'), self.widthSpinBox)
        self.layout().addRow(trans._('length:'), self.lengthSpinBox)
        self.layout().addRow(trans._('blending:'), self.blendComboBox)
        self.layout().addRow(
            trans._('projection mode:'), self.projectionComboBox
        )
        self.layout().addRow(
            trans._('vector style:'), self.vector_style_comboBox
        )
        self.layout().addRow(
            trans._('edge color mode:'), self.color_mode_comboBox
        )
        self.layout().addRow(self.edge_color_label, self.edgeColorEdit)
        self.layout().addRow(self.edge_prop_label, self.color_prop_box)
        self.layout().addRow(trans._('out of slice:'), self.outOfSliceCheckBox)

        self.layer.events.edge_width.connect(self._on_edge_width_change)
        self.layer.events.length.connect(self._on_length_change)
        self.layer.events.out_of_slice_display.connect(
            self._on_out_of_slice_display_change
        )
        self.layer.events.vector_style.connect(self._on_vector_style_change)
        self.layer.events.edge_color_mode.connect(
            self._on_edge_color_mode_change
        )
        self.layer.events.edge_color.connect(self._on_edge_color_change)

    def change_edge_color_property(self, property_name: str):
        """Change edge_color_property of vectors on the layer model.
        This property is the property the edge color is mapped to.

        Parameters
        ----------
        property_name : str
            property to map the edge color to
        """
        mode = self.layer.edge_color_mode
        try:
            self.layer.edge_color = property_name
            self.layer.edge_color_mode = mode
        except TypeError:
            # if the selected property is the wrong type for the current color mode
            # the color mode will be changed to the appropriate type, so we must update
            self._on_edge_color_mode_change()
            raise

    def change_vector_style(self, vector_style: str):
        """Change vector style of vectors on the layer model.

        Parameters
        ----------
        vector_style : str
            Name of vectors style, eg: 'line', 'triangle' or 'arrow'.
        """
        with self.layer.events.vector_style.blocker():
            self.layer.vector_style = vector_style

    def change_edge_color_mode(self, mode: str):
        """Change edge color mode of vectors on the layer model.

        Parameters
        ----------
        mode : str
            Edge color for vectors. Must be: 'direct', 'cycle', or 'colormap'
        """
        old_mode = self.layer.edge_color_mode
        with self.layer.events.edge_color_mode.blocker():
            try:
                self.layer.edge_color_mode = mode
                self._update_edge_color_gui(mode)

            except ValueError:
                # if the color mode was invalid, revert to the old mode (layer and GUI)
                self.layer.edge_color_mode = old_mode
                self.color_mode_comboBox.setCurrentText(old_mode)
                raise

    def change_edge_color_direct(self, color: np.ndarray):
        """Change edge color of vectors on the layer model.

        Parameters
        ----------
        color : np.ndarray
            Edge color for vectors, in an RGBA array
        """
        self.layer.edge_color = color

    def change_width(self, value):
        """Change edge line width of vectors on the layer model.

        Parameters
        ----------
        value : float
            Line width of vectors.
        """
        self.layer.edge_width = value
        self.widthSpinBox.clearFocus()
        self.setFocus()

    def change_length(self, value):
        """Change length of vectors on the layer model.

        Multiplicative factor on projections for length of all vectors.

        Parameters
        ----------
        value : float
            Length of vectors.
        """
        self.layer.length = value
        self.lengthSpinBox.clearFocus()
        self.setFocus()

    def change_out_of_slice(self, state):
        """Toggle out of slice display of vectors layer.

        Parameters
        ----------
        state : int
            Integer value of Qt.CheckState that indicates the check state of outOfSliceCheckBox
        """
        self.layer.out_of_slice_display = (
            Qt.CheckState(state) == Qt.CheckState.Checked
        )

    def _update_edge_color_gui(self, mode: str):
        """Update the GUI element associated with edge_color.
        This is typically used when edge_color_mode changes

        Parameters
        ----------
        mode : str
            The new edge_color mode the GUI needs to be updated for.
            Should be: 'direct', 'cycle', 'colormap'
        """
        if mode in {'cycle', 'colormap'}:
            self.edgeColorEdit.setHidden(True)
            self.edge_color_label.setHidden(True)
            self.color_prop_box.setHidden(False)
            self.edge_prop_label.setHidden(False)

        elif mode == 'direct':
            self.edgeColorEdit.setHidden(False)
            self.edge_color_label.setHidden(False)
            self.color_prop_box.setHidden(True)
            self.edge_prop_label.setHidden(True)

    def _get_property_values(self):
        """Get the current property values from the Vectors layer

        Returns
        -------
        property_values : np.ndarray
            array of all of the union of the property names (keys)
            in Vectors.properties and Vectors.property_choices

        """
        property_choices = [*self.layer.property_choices]
        properties = [*self.layer.properties]
        property_values = np.union1d(property_choices, properties)

        return property_values

    def _on_length_change(self):
        """Change length of vectors."""
        with self.layer.events.length.blocker():
            self.lengthSpinBox.setValue(self.layer.length)

    def _on_out_of_slice_display_change(self, event):
        """Receive layer model out_of_slice_display change event and update checkbox."""
        with self.layer.events.out_of_slice_display.blocker():
            self.outOfSliceCheckBox.setChecked(self.layer.out_of_slice_display)

    def _on_edge_width_change(self):
        """Receive layer model width change event and update width spinbox."""
        with self.layer.events.edge_width.blocker():
            self.widthSpinBox.setValue(self.layer.edge_width)

    def _on_vector_style_change(self):
        """Receive layer model vector style change event & update dropdown."""
        with self.layer.events.vector_style.blocker():
            vector_style = self.layer.vector_style
            index = self.vector_style_comboBox.findText(
                vector_style, Qt.MatchFixedString
            )
            self.vector_style_comboBox.setCurrentIndex(index)

    def _on_edge_color_mode_change(self):
        """Receive layer model edge color mode change event & update dropdown."""
        with qt_signals_blocked(self.color_mode_comboBox):
            mode = self.layer._edge.color_mode
            index = self.color_mode_comboBox.findText(
                mode, Qt.MatchFixedString
            )
            self.color_mode_comboBox.setCurrentIndex(index)

            self._update_edge_color_gui(mode)

    def _on_edge_color_change(self):
        """Receive layer model edge color  change event & update dropdown."""
        if (
            self.layer._edge.color_mode == ColorMode.DIRECT
            and len(self.layer.data) > 0
        ):
            with qt_signals_blocked(self.edgeColorEdit):
                self.edgeColorEdit.setColor(self.layer.edge_color[0])
        elif self.layer._edge.color_mode in (
            ColorMode.CYCLE,
            ColorMode.COLORMAP,
        ):
            with qt_signals_blocked(self.color_prop_box):
                prop = self.layer._edge.color_properties.name
                index = self.color_prop_box.findText(prop, Qt.MatchFixedString)
                self.color_prop_box.setCurrentIndex(index)
