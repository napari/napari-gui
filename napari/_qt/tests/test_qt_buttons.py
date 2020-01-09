from qtpy.QtCore import Qt

from napari._qt.qt_mode_buttons import QtModePushButton, QtModeRadioButton
from napari.layers import Points
from napari.layers.points._constants import Mode


def test_radio_button(qtbot):
    """Make sure the QtModeRadioButton works to change layer modes"""
    layer = Points()
    assert layer.mode != Mode.ADD

    btn = QtModeRadioButton(layer, 'test_button', Mode.ADD, tool_tip='tooltip')
    assert btn.property('mode') == 'test_button'
    assert btn.toolTip() == 'tooltip'

    qtbot.mouseClick(btn, Qt.LeftButton)
    assert layer.mode == 'add'


def test_push_button(qtbot):
    """Make sure the QtModePushButton works with callbacks"""
    layer = Points()

    def set_test_prop():
        layer.test_prop = True

    btn = QtModePushButton(
        layer, 'test_button', set_test_prop, tool_tip='tooltip'
    )
    assert btn.property('mode') == 'test_button'
    assert btn.toolTip() == 'tooltip'

    qtbot.mouseClick(btn, Qt.LeftButton)
    assert layer.test_prop
