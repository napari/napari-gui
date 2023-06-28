import numpy as np
import pytest
from qtpy.QtGui import QColor

from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari.layers import Labels
from napari.utils.colormaps import colormap_utils

np.random.seed(0)
_LABELS = np.random.randint(5, size=(10, 15))
_COLOR = {1: 'white', 2: 'blue', 3: 'green', 4: 'red', 5: 'yellow'}


@pytest.fixture
def make_labels_controls(qtbot, color=None, predefined_labels=None):
    def _make_labels_controls(
        color=color, predefined_labels=predefined_labels
    ):
        layer = Labels(
            _LABELS, color=color, predefined_labels=predefined_labels
        )
        qtctrl = QtLabelsControls(layer)
        qtbot.add_widget(qtctrl)
        return layer, qtctrl

    return _make_labels_controls


def test_changing_layer_color_mode_updates_combo_box(make_labels_controls):
    """Updating layer color mode changes the combo box selection"""
    layer, qtctrl = make_labels_controls(color=_COLOR)

    original_color_mode = layer.color_mode
    assert original_color_mode == qtctrl.colorModeComboBox.currentText()

    layer.color_mode = 'auto'
    assert layer.color_mode == qtctrl.colorModeComboBox.currentText()


def test_rendering_combobox(make_labels_controls):
    """Changing the model attribute should update the view"""
    layer, qtctrl = make_labels_controls()
    combo = qtctrl.renderComboBox
    opts = {combo.itemText(i) for i in range(combo.count())}
    rendering_options = {'translucent', 'iso_categorical'}
    assert opts == rendering_options
    # programmatically updating rendering mode updates the combobox
    new_mode = 'iso_categorical'
    layer.rendering = new_mode
    assert combo.findText(new_mode) == combo.currentIndex()


def test_changing_colormap_updates_colorbox(make_labels_controls):
    """Test that changing the colormap on a layer will update color swatch in the combo box"""
    layer, qtctrl = make_labels_controls(color=_COLOR)
    color_box = qtctrl.labelsSpinbox.colorBox

    layer.selected_label = 1

    # For a paint event, which does not occur in a headless qtbot
    color_box.paintEvent(None)

    np.testing.assert_equal(color_box._color, layer._selected_color)

    layer.colormap = colormap_utils.label_colormap(num_colors=5)

    # For a paint event, which does not occur in a headless qtbot
    color_box.paintEvent(None)

    np.testing.assert_equal(color_box._color, layer._selected_color)


def test_selected_color_checkbox(make_labels_controls):
    """Tests that the 'selected color' checkbox sets the 'show_selected_label' property properly."""
    layer, qtctrl = make_labels_controls()
    qtctrl.selectedColorCheckbox.setChecked(True)
    assert layer.show_selected_label
    qtctrl.selectedColorCheckbox.setChecked(False)
    assert not layer.show_selected_label
    qtctrl.selectedColorCheckbox.setChecked(True)
    assert layer.show_selected_label


def test_contiguous_labels_checkbox(make_labels_controls):
    """Tests that the 'contiguous' checkbox sets the 'contiguous' property properly."""
    layer, qtctrl = make_labels_controls()
    qtctrl.contigCheckBox.setChecked(True)
    assert layer.contiguous
    qtctrl.contigCheckBox.setChecked(False)
    assert not layer.contiguous
    qtctrl.contigCheckBox.setChecked(True)
    assert layer.contiguous


def test_preserve_labels_checkbox(make_labels_controls):
    """Tests that the 'preserve labels' checkbox sets the 'preserve_labels' property properly."""
    layer, qtctrl = make_labels_controls()
    qtctrl.preserveLabelsCheckBox.setChecked(True)
    assert layer.preserve_labels
    qtctrl.preserveLabelsCheckBox.setChecked(False)
    assert not layer.preserve_labels
    qtctrl.preserveLabelsCheckBox.setChecked(True)
    assert layer.preserve_labels


def test_labels_combobox(make_labels_controls):
    """Tests that QtLabelsCombobox interacts correctly with the Labels layer."""
    predefined_labels = [10, 20, 30, 40, 50]
    layer, qtctrl = make_labels_controls(predefined_labels=predefined_labels)

    qtctrl.labelsCombobox.setCurrentIndex(2)
    assert layer.selected_label == 20

    # Check that selected labels matches the correct combobox item
    # and that all the combobox items are created properly
    for label_id in predefined_labels:
        layer.selected_label = label_id

        assert qtctrl.labelsCombobox.currentText() == str(label_id)

        icon = qtctrl.labelsCombobox.itemIcon(
            qtctrl.labelsCombobox.currentIndex()
        )
        icon_image = icon.pixmap(qtctrl.labelsCombobox._height).toImage()
        color = QColor(icon_image.pixel(5, 5)).getRgbF()
        assert np.allclose(color[:3], layer.get_color(label_id)[:3], atol=0.05)

    # Check if the icons are updated after setting a new colormap
    layer.new_colormap()
    for label_id in predefined_labels:
        layer.selected_label = label_id
        icon = qtctrl.labelsCombobox.itemIcon(
            qtctrl.labelsCombobox.currentIndex()
        )
        icon_image = icon.pixmap(qtctrl.labelsCombobox._height).toImage()
        color = QColor(icon_image.pixel(5, 5)).getRgbF()
        assert np.allclose(color[:3], layer.get_color(label_id)[:3], atol=0.05)

    layer.selected_label = layer._background_label
    assert qtctrl.labelsCombobox.currentText().endswith(': background')

    layer.selected_label = 5
    assert qtctrl.labelsCombobox.currentText() == '5: unspecified'


def test_switching_labels_selection_widget(make_labels_controls):
    """Tests changing the labels selection widget."""
    predefined_labels = [1, 2, 3]
    layer, qtctrl = make_labels_controls(predefined_labels=[1, 2, 3])

    assert qtctrl.layout().indexOf(qtctrl.labelsSpinbox) == -1
    assert qtctrl.layout().indexOf(qtctrl.labelsCombobox) != -1

    layer.predefined_labels = None
    assert qtctrl.layout().indexOf(qtctrl.labelsCombobox) == -1
    assert qtctrl.layout().indexOf(qtctrl.labelsSpinbox) != -1

    qtctrl.labelsSpinbox.selectionSpinBox.setValue(3)
    assert layer.selected_label == 3

    layer.selected_label = 2
    assert qtctrl.labelsSpinbox.selectionSpinBox.value() == 2

    layer.predefined_labels = predefined_labels
    assert qtctrl.layout().indexOf(qtctrl.labelsSpinbox) == -1
    assert qtctrl.layout().indexOf(qtctrl.labelsCombobox) != -1

    layer.selected_label = 3
    assert qtctrl.labelsCombobox.currentText().startswith("3")
    assert qtctrl.labelsSpinbox.selectionSpinBox.value() != 3

    qtctrl.labelsCombobox.setCurrentIndex(0)
    assert layer.selected_label != 3
