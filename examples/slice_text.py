"""
Slice text overlay
=========

Display a 3D timelapse and the slice text overlay.

.. tags:: experimental
"""
import numpy as np
from skimage import data

import napari

cells = data.cells3d()

cells_timelapse = np.stack([cells] * 5)

viewer = napari.Viewer(axis_labels=('t', 'z', 'y', 'x'))

viewer.add_image(
    cells_timelapse,
    name=('membrane', 'nuclei'),
    channel_axis=2,
    scale=(30, 0.29, 0.26, 0.26),
)
# didn't work before adding the image
viewer.dims.axis_labels = ('t', 'z', 'y', 'x')

viewer.slice_text.visible = True

# also showing scale bar to view colision
viewer.scale_bar.visible = True
viewer.scale_bar.unit = 'um'

if __name__ == '__main__':
    napari.run()
