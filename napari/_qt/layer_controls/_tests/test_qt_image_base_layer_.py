import os
from unittest.mock import patch

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton

from napari._qt.layer_controls.qt_image_controls_base import (
    QtBaseImageControls,
    QtLayerControls,
)
from napari._qt.layer_controls.widgets.qt_contrast_limits import (
    QContrastLimitsPopup,
    QRangeSliderPopup,
    range_to_decimals,
)
from napari.components.dims import Dims
from napari.layers import Image, Surface

_IMAGE = np.arange(100).astype(np.uint16).reshape((10, 10))
_SURF = (
    np.random.random((10, 2)),
    np.random.randint(10, size=(6, 3)),
    np.arange(100).astype(float),
)


@pytest.mark.parametrize('layer', [Image(_IMAGE), Surface(_SURF)])
def test_base_controls_creation(qtbot, layer):
    """Check basic creation of QtBaseImageControls works"""
    qtctrl = QtBaseImageControls(layer)
    qtbot.addWidget(qtctrl)
    original_clims = tuple(layer.contrast_limits)
    slider_clims = (
        qtctrl.qtContrastLimitsSliderControl.contrastLimitsSlider.value()
    )
    assert slider_clims[0] == 0
    assert slider_clims[1] == 99
    assert tuple(slider_clims) == original_clims


@patch.object(QRangeSliderPopup, 'show')
@pytest.mark.parametrize('layer', [Image(_IMAGE), Surface(_SURF)])
def test_clim_right_click_shows_popup(mock_show, qtbot, layer):
    """Right clicking on the contrast limits slider should show a popup."""
    qtctrl = QtBaseImageControls(layer)
    qtbot.addWidget(qtctrl)
    qtbot.mousePress(
        qtctrl.qtContrastLimitsSliderControl.contrastLimitsSlider,
        Qt.RightButton,
    )
    assert hasattr(qtctrl.qtContrastLimitsSliderControl, 'clim_popup')
    # this mock doesn't seem to be working on cirrus windows
    # but it works on local windows tests...
    if not (os.name == 'nt' and os.getenv('CI')):
        mock_show.assert_called_once()


@pytest.mark.parametrize('layer', [Image(_IMAGE), Surface(_SURF)])
def test_changing_model_updates_view(qtbot, layer):
    """Changing the model attribute should update the view"""
    qtctrl = QtBaseImageControls(layer)
    qtbot.addWidget(qtctrl)
    new_clims = (20, 40)
    layer.contrast_limits = new_clims
    assert (
        tuple(
            qtctrl.qtContrastLimitsSliderControl.contrastLimitsSlider.value()
        )
        == new_clims
    )


@patch.object(QRangeSliderPopup, 'show')
@pytest.mark.parametrize(
    'layer', [Image(_IMAGE), Image(_IMAGE.astype(np.int32)), Surface(_SURF)]
)
def test_range_popup_clim_buttons(mock_show, qtbot, qapp, layer):
    """The buttons in the clim_popup should adjust the contrast limits value"""
    # this test relies implicitly on ndisplay=3 which is now a broken assumption?
    layer._slice_dims(Dims(ndim=3, ndisplay=3))
    qtctrl = QtBaseImageControls(layer)
    qtbot.addWidget(qtctrl)
    original_clims = tuple(layer.contrast_limits)
    layer.contrast_limits = (20, 40)
    qtbot.mousePress(
        qtctrl.qtContrastLimitsSliderControl.contrastLimitsSlider,
        Qt.RightButton,
    )

    # pressing the reset button returns the clims to the default values
    reset_button = qtctrl.qtContrastLimitsSliderControl.clim_popup.findChild(
        QPushButton, 'reset_clims_button'
    )
    reset_button.click()
    qapp.processEvents()
    assert (
        tuple(
            qtctrl.qtContrastLimitsSliderControl.contrastLimitsSlider.value()
        )
        == original_clims
    )

    rangebtn = qtctrl.qtContrastLimitsSliderControl.clim_popup.findChild(
        QPushButton, 'full_clim_range_button'
    )
    # data in this test is uint16 or int32 for Image, and float for Surface.
    # Surface will not have a "full range button"
    if np.issubdtype(layer.dtype, np.integer):
        info = np.iinfo(layer.dtype)
        rangebtn.click()
        qapp.processEvents()
        assert tuple(layer.contrast_limits_range) == (info.min, info.max)
        min_ = (
            qtctrl.qtContrastLimitsSliderControl.contrastLimitsSlider.minimum()
        )
        max_ = (
            qtctrl.qtContrastLimitsSliderControl.contrastLimitsSlider.maximum()
        )
        assert (min_, max_) == (info.min, info.max)
    else:
        assert rangebtn is None


@pytest.mark.parametrize('mag', list(range(-16, 16, 4)))
def test_clim_slider_step_size_and_precision(qtbot, mag):
    """Make sure the slider has a reasonable step size and precision.

    ...across a broad range of orders of magnitude.
    """
    layer = Image(np.random.rand(20, 20) * 10**mag)
    popup = QContrastLimitsPopup(layer)
    qtbot.addWidget(popup)

    # scale precision with the log of the data range order of magnitude
    # eg.   0 - 1   (0 order of mag)  -> 3 decimal places
    #       0 - 10  (1 order of mag)  -> 2 decimals
    #       0 - 100 (2 orders of mag) -> 1 decimal
    #       ≥ 3 orders of mag -> no decimals
    # no more than 64 decimals
    decimals = range_to_decimals(layer.contrast_limits, layer.dtype)
    assert popup.slider.decimals() == decimals

    # the slider step size should also be inversely proportional to the data
    # range, with 1000 steps across the data range
    assert popup.slider.singleStep() == 10**-decimals


def test_qt_image_controls_change_contrast(qtbot):
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtBaseImageControls(layer)
    qtbot.addWidget(qtctrl)
    qtctrl.qtContrastLimitsSliderControl.contrastLimitsSlider.setValue(
        (0.1, 0.8)
    )
    assert tuple(layer.contrast_limits) == (0.1, 0.8)


def test_tensorstore_clim_popup(qtbot):
    """Regression to test, makes sure it works with tensorstore dtype"""
    ts = pytest.importorskip('tensorstore')
    layer = Image(ts.array(np.random.rand(20, 20)))
    qtbot.addWidget(QContrastLimitsPopup(layer))


def test_blending_opacity_slider(qtbot):
    """Tests whether opacity slider is disabled for minimum and opaque blending."""
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtLayerControls(layer)
    qtbot.addWidget(qtctrl)
    assert layer.blending == 'translucent'
    # check that the opacity slider is present by default
    assert qtctrl.qtOpacityBlendingControls.opacitySlider.isEnabled()
    # set minimum blending, the opacity slider should be disabled
    layer.blending = 'minimum'
    assert not qtctrl.qtOpacityBlendingControls.opacitySlider.isEnabled()
    # set the blending to 'additive' confirm the slider is enabled
    layer.blending = 'additive'
    assert layer.blending == 'additive'
    assert qtctrl.qtOpacityBlendingControls.opacitySlider.isEnabled()
    # set opaque blending, the opacity slider should be disabled
    layer.blending = 'opaque'
    assert layer.blending == 'opaque'
    assert not qtctrl.qtOpacityBlendingControls.opacitySlider.isEnabled()
    # set the blending back to 'translucent' confirm the slider is enabled
    layer.blending = 'translucent'
    assert layer.blending == 'translucent'
    assert qtctrl.qtOpacityBlendingControls.opacitySlider.isEnabled()
