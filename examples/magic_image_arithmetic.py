"""
magicgui Image Arithmetic
=========================

Basic example of using magicgui to create an Image Arithmetic GUI in napari.

.. tags:: gui
"""

import enum

import numpy as np

import napari


# Enums are a convenient way to get a dropdown menu
class Operation(enum.Enum):
    """A set of valid arithmetic operations for image_arithmetic."""

    add = np.add
    subtract = np.subtract
    multiply = np.multiply
    divide = np.divide


# Define our image_arithmetic function.
# Note that we can use forward references for the napari type annotations.
# You can read more about them here:
# https://peps.python.org/pep-0484/#forward-references
# In this example, because we have already imported napari anyway, it doesn't
# really matter. But this syntax would let you specify that a parameter is a
# napari object type without actually importing or depending on napari.
# Note: here we use `napari.types.ImageData` as our parameter annotations,
# which means our function will be passed layer.data instead of
# the full layer instance
def image_arithmetic(
    layerA: 'napari.types.ImageData',
    operation: Operation,
    layerB: 'napari.types.ImageData',
) -> 'napari.types.ImageData':
    """Adds, subtracts, multiplies, or divides two same-shaped image layers."""
    if layerA is not None and layerB is not None:
        return operation.value(layerA, layerB)
    return None


# create a new viewer with a couple image layers
viewer = napari.Viewer()
viewer.add_image(np.random.rand(20, 20), name='Layer 1', scale=(2, 2))
viewer.add_image(np.random.rand(20, 20), name='Layer 2', scale=(2, 2))

# Add our magic function to napari
viewer.window.add_function_widget(image_arithmetic)

if __name__ == '__main__':
    napari.run()
