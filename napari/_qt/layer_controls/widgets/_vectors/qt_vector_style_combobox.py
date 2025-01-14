from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.layers.vectors._vectors_constants import VECTORSTYLE_TRANSLATIONS
from napari.utils.translations import trans


class QtVectorStyleComboBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer edge style
    value attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    vector_style_comboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select vector_style for the vectors.
    vector_style_comboBox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for vector_style value chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.vector_style.connect(self._on_vector_style_change)

        # Setup widgets
        # dropdown to select the edge display vector_style
        vector_style_comboBox = QComboBox(parent)
        for index, (data, text) in enumerate(VECTORSTYLE_TRANSLATIONS.items()):
            data = data.value
            vector_style_comboBox.addItem(text, data)
            if data == self._layer.vector_style:
                vector_style_comboBox.setCurrentIndex(index)

        self.vector_style_comboBox = vector_style_comboBox
        self.vector_style_comboBox.currentTextChanged.connect(
            self.change_vector_style
        )

        self.vector_style_comboBox_label = QtWrappedLabel(
            trans._('vector style:')
        )

    def change_vector_style(self, vector_style: str) -> None:
        """Change vector style of vectors on the layer model.

        Parameters
        ----------
        vector_style : str
            Name of vectors style, eg: 'line', 'triangle' or 'arrow'.
        """
        with self._layer.events.vector_style.blocker():
            self._layer.vector_style = vector_style

    def _on_vector_style_change(self) -> None:
        """Receive layer model vector style change event & update dropdown."""
        with self._layer.events.vector_style.blocker():
            vector_style = self._layer.vector_style
            index = self.vector_style_comboBox.findText(
                vector_style, Qt.MatchFlag.MatchFixedString
            )
            self.vector_style_comboBox.setCurrentIndex(index)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.vector_style_comboBox_label, self.vector_style_comboBox)]
