import numpy as np

from napari._qt.layer_controls.qt_surface_controls import QtSurfaceControls
from napari.layers import Surface
from napari.layers.surface._surface_constants import SHADING_TRANSLATION

data = np.array([[0, 0], [0, 20], [10, 0], [10, 10]])
faces = np.array([[0, 1, 2], [1, 2, 3]])
values = np.linspace(0, 1, len(data))
_SURFACE = (data, faces, values)


def test_shading_combobox(qtbot):
    layer = Surface(_SURFACE)
    qtctrl = QtSurfaceControls(layer)
    qtbot.addWidget(qtctrl)
    assert (
        qtctrl.qtShadingComboBoxControl.shadingComboBox.currentText()
        == layer.shading
    )

    for display, shading in SHADING_TRANSLATION.items():
        qtctrl.qtShadingComboBoxControl.shadingComboBox.setCurrentText(display)
        assert layer.shading == shading

    for display, shading in SHADING_TRANSLATION.items():
        layer.shading = shading
        assert (
            qtctrl.qtShadingComboBoxControl.shadingComboBox.currentText()
            == display
        )
