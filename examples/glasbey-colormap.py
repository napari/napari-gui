"""
Glasbey colormap
================

A long-requested feature [1]_ for napari was to display labels/segmentation
layers using a well-known discrete colormap such as Glasbey [2]_.

In this example, we demonstrate displaying segmentations using custom colormaps
with the help of the glasbey Python library [3]_. We display a segmentation
using the napari built-in labels colormap, the original Glasbey colormap, a
more modern version produced by limiting the lightness and chroma and
optimizing for colorblind-safety, and finally with the matplotlib tab10
colormap.

.. [1] https://github.com/napari/napari/issues/454
.. [2] Colour displays for categorical images. Chris Glasbey, Gerie van der
       Heijden, Vivian F. K. Toh, and Alision Gray. (2007)
       :DOI:`10.1002/col.20327`
.. [3] https://github.com/lmcinnes/glasbey

.. tags:: layers, visualization-basic
"""
import glasbey
from skimage import data
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import closing, remove_small_objects, square
from skimage.segmentation import clear_border

import napari

image = data.coins()[50:-50, 50:-50]

###############################################################################
# First, we segment the image.

# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, square(4))

# remove artifacts connected to image border
cleared = remove_small_objects(clear_border(bw), 20)

# label image regions
label_image = label(cleared).astype("uint8")

###############################################################################
# Then, we create two color palettes using the glasbey library. One with the
# original glasbey parameters and 256 colors, and a more modern one with better
# lightness and chroma bounds for a less glary look.

# original glasbey
glas = glasbey.create_palette(256)

# more optimized glasbey
glas19mid = glasbey.create_palette(
        19,
        lightness_bounds=(20, 60), chroma_bounds=(40, 50),
        colorblind_safe=True,
        )

###############################################################################
# Finally, we display the coins image and the overlaid segmentation. We do this
# twice to show both colormaps.

viewer, image_layer = napari.imshow(image, name='coins')

# add the labels
label_layer_glas = viewer.add_labels(
        label_image, name='segmentation', colormap=glas
        )

viewer2, image_layer2 = napari.imshow(image, name='coins')

label_layer_modern = viewer2.add_labels(
        label_image, name='segmentation-glasbey-19-mid-chroma', colormap=glas19mid,
        )


if __name__ == '__main__':
    napari.run()
