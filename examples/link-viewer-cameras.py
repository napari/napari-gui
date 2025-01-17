"""
Link cameras between two viewers
================================

This lets you zoom in on different layers side by side instead of overlaying
them. [1]_

.. [1] https://forum.image.sc/t/zooming-in-on-the-same-region-on-two-different-layers/95659/5

.. tags:: gui, layers, visualization-advanced, multicanvas
"""
from functools import partial

from skimage import data, filters

import napari

# example data
coins = data.coins()
binary = coins > filters.threshold_otsu(coins)

# make two viewers
viewer0 = napari.Viewer()
viewer1 = napari.Viewer()

# add a layer to each
viewer0.add_image(coins)
viewer1.add_labels(binary)

# note: check available events with:
# viewer0.camera.events.emitters

# reusable functions to set camera zoom and center
def set_zoom(camera, event):
    camera.zoom = event.value

def set_center(camera, event):
    camera.center = event.value

# hook everything up — viewer0's camera's zoom and center to viewer1's, and
# vice-versa
viewer0.camera.events.zoom.connect(
    partial(set_zoom, viewer1.camera)
)
viewer1.camera.events.zoom.connect(
    partial(set_zoom, viewer0.camera)
)
viewer0.camera.events.center.connect(
    partial(set_center, viewer1.camera)
)
viewer1.camera.events.center.connect(
    partial(set_center, viewer0.camera)
)

if __name__ == '__main__':
    napari.run()
