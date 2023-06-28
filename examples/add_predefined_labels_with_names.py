"""
Add a closed set of labels with names
========================

Display a labels layer with a closed predefined set of labels

.. tags:: layers, analysis
"""


import numpy as np
import pandas as pd
from skimage import data
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import closing, remove_small_objects, square
from skimage.segmentation import clear_border

import napari

image = data.coins()[50:-50, 50:-50]

# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, square(4))

# remove artifacts connected to image border
cleared = remove_small_objects(clear_border(bw), 20)
ignored_area = cleared != bw

# label image regions
label_image = label(cleared)

# initialise viewer with coins image
viewer = napari.view_image(image, name='coins', rgb=False)

# get the size of each coin (first element is background area)
label_areas = np.bincount(label_image.ravel())[1:]

# split coins into small or large
size_range = max(label_areas) - min(label_areas)
small_threshold = min(label_areas) + (size_range / 2)
label_mapping = np.zeros(len(label_areas) + 1, dtype=int)
label_mapping[1:][label_areas < small_threshold] = 1
label_mapping[1:][label_areas >= small_threshold] = 10

label_image = label_mapping[label_image]
label_image[ignored_area] = 255

labels_df = pd.DataFrame.from_dict({
    1: ['small coin', 'green'],
    10: ['big coin', 'orange'],
    255: ['ignore', 'blue']
}, orient='index', columns=['name', 'color'])

# add the labels
label_layer = viewer.add_labels(
    label_image,
    name='segmentation',
    predefined_labels=labels_df['name'].to_dict(),
    color=labels_df['color'].to_dict(),
)

if __name__ == '__main__':
    napari.run()
