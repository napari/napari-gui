import numpy as np

from napari._qt.layer_controls.qt_vectors_controls import QtVectorsControls
from napari.layers import Vectors

_VECTORS = np.zeros((2, 2, 2))


def test_out_of_slice_display_checkbox(qtbot):
    layer = Vectors(_VECTORS)
    qtctrl = QtVectorsControls(layer)
    qtbot.addWidget(qtctrl)
    qtctrl.qtOutSliceCheckBoxControl.outOfSliceCheckBox.setChecked(True)
    assert layer.out_of_slice_display
    qtctrl.qtOutSliceCheckBoxControl.outOfSliceCheckBox.setChecked(False)
    assert not layer.out_of_slice_display
