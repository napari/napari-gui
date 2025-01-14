import numpy as np
import pytest

from napari._qt.layer_controls.qt_points_controls import QtPointsControls
from napari.layers import Points


def test_out_of_slice_display_checkbox(qtbot):
    """Changing the model attribute should update the view"""
    layer = Points(np.random.rand(10, 2))
    qtctrl = QtPointsControls(layer)
    qtbot.addWidget(qtctrl)
    combo = qtctrl.qtOutSliceCheckBoxControl.outOfSliceCheckBox

    assert layer.out_of_slice_display is False
    combo.setChecked(True)
    assert layer.out_of_slice_display is True


def test_current_size_display_in_range(qtbot):
    """Changing the model attribute should update the view"""
    layer = Points(np.random.rand(10, 2))
    qtctrl = QtPointsControls(layer)
    qtbot.addWidget(qtctrl)
    slider = qtctrl.qtCurrentSizeSliderControl.sizeSlider
    slider.setValue(10)

    # Initial values
    assert slider.maximum() == 100
    assert slider.minimum() == 1
    assert slider.value() == 10
    assert layer.current_size == 10

    # Size event needs to be triggered manually, because no points are selected.
    layer.current_size = 5
    layer.events.size()
    assert slider.maximum() == 100
    assert slider.minimum() == 1
    assert slider.value() == 5
    assert layer.current_size == 5

    # Size event needs to be triggered manually, because no points are selected.
    layer.current_size = 100
    layer.events.size()
    assert slider.maximum() == 100
    assert slider.minimum() == 1
    assert slider.value() == 100
    assert layer.current_size == 100

    # Size event needs to be triggered manually, because no points are selected.
    layer.current_size = 200
    layer.events.size()
    assert slider.maximum() == 201
    assert slider.minimum() == 1
    assert slider.value() == 200
    assert layer.current_size == 200

    # Size event needs to be triggered manually, because no points are selected.
    with pytest.raises(ValueError, match='must be positive'):
        layer.current_size = -1000
    layer.events.size()
    assert slider.maximum() == 201
    assert slider.minimum() == 1
    assert slider.value() == 200
    assert layer.current_size == 200

    layer.current_size = 20
    layer.events.size()
    assert slider.maximum() == 201
    assert slider.minimum() == 1
    assert slider.value() == 20
    assert layer.current_size == 20

    with pytest.warns(DeprecationWarning):
        layer.current_size = [10, 10]
    layer.events.size()
    assert slider.maximum() == 201
    assert slider.minimum() == 1
    assert slider.value() == 10
    assert layer.current_size == 10


def test_current_size_slider_properly_initialized(qtbot):
    """Changing the model attribute should update the view"""
    layer = Points(np.random.rand(10, 2), size=np.linspace(-2, 200, 10))
    qtctrl = QtPointsControls(layer)
    qtbot.addWidget(qtctrl)
    slider = qtctrl.qtCurrentSizeSliderControl.sizeSlider
    assert slider.maximum() == 201
    assert slider.minimum() == 1
    assert slider.value() == 10
    assert layer.current_size == 10

    layer = Points(np.random.rand(10, 2), size=np.linspace(-2, 50, 10))
    qtctrl = QtPointsControls(layer)
    qtbot.addWidget(qtctrl)
    slider = qtctrl.qtCurrentSizeSliderControl.sizeSlider
    assert slider.maximum() == 100
    assert slider.minimum() == 1
    assert slider.value() == 10
    assert layer.current_size == 10


def test_size_slider_represents_current_size(qtbot):
    """Changing the current_size attribute should update the slider"""
    layer = Points(np.random.rand(10, 2))
    qtctrl = QtPointsControls(layer)
    qtbot.addWidget(qtctrl)
    slider = qtctrl.qtCurrentSizeSliderControl.sizeSlider
    slider.setValue(10)

    # Initial value
    assert slider.value() == 10
    assert layer.current_size == 10

    # Size event needs to be triggered manually, because no points are selected.
    layer.current_size = 5
    layer.events.current_size()
    assert slider.value() == 5
    assert layer.current_size == 5
