"""
magicgui parameter sweep
========================

Example showing how to accomplish a napari parameter sweep with magicgui.

It demonstrates:
1. overriding the default widget type with a custom class
2. the `auto_call` option, which calls the function whenever a parameter changes

.. tags:: gui
"""
from typing import Annotated

import skimage.data
import skimage.filters

import napari


# Define our gaussian_blur function.
# Note that we can use forward references for the napari type annotations.
# You can read more about them here:
# https://peps.python.org/pep-0484/#forward-references
# In this example, because we have already imported napari anyway, it doesn't
# really matter. But this syntax would let you specify that a parameter is a
# napari object type without actually importing or depending on napari.
# We also use the `Annotated` type to pass an additional dictionary that can be used
# to aid widget generation. The keys of the dictionary are keyword arguments to
# the corresponding magicgui widget type. For more information see
# https://napari.org/magicgui/api/widgets.html.
def gaussian_blur(
    layer: 'napari.layers.Image',
    sigma: Annotated[float, {'widget_type': 'FloatSlider', 'max': 6}] = 1.0,
    mode: Annotated[str, {'choices': ['reflect', 'constant', 'nearest', 'mirror', 'wrap']}]='nearest',
) -> 'napari.types.ImageData | None':
    """Apply a gaussian blur to ``layer``."""
    if layer:
        return skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode)
    return None


# create a viewer and add some images
viewer = napari.Viewer()
viewer.add_image(skimage.data.astronaut().mean(-1), name='astronaut')
viewer.add_image(skimage.data.grass().astype('float'), name='grass')

# Add our magic function to napari
viewer.window.add_function_widget(gaussian_blur)

if __name__ == '__main__':
    napari.run()
