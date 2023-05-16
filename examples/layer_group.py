import os

import numpy as np
from skimage import data

import napari

os.environ['ALLOW_LAYERGROUPS'] = '1'


points = np.random.rand(10, 3) * (1, 2, 3) * 100
colors = np.random.rand(10, 3)

blobs = data.binary_blobs(length=100, volume_fraction=0.05, n_dim=2)

viewer = napari.Viewer(ndisplay=3)

pl = napari.layers.Points(points, face_color=colors)
il = napari.layers.Image(blobs, scale=(1, 2), translate=(20, 15))

viewer.add_layer([pl, il])

napari.run()
