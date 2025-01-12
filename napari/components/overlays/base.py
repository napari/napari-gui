from napari.components._viewer_constants import CanvasPosition
from napari.layers.base._base_constants import Blending
from napari.utils.events import EventedModel


class Overlay(EventedModel):
    """
    Overlay evented model.

    An overlay is a renderable entity meant to display additional information
    on top of the layer data, but is not data per se.
    For example: a scale bar, a color bar, axes, bounding boxes, etc.

    Attributes
    ----------
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    visible: bool = False
    opacity: float = 1
    order: int = 10**6
    blending: Blending

    def __hash__(self):
        return id(self)


class CanvasOverlay(Overlay):
    """
    Canvas overlay model.

    Canvas overlays live in canvas space; they do not live in the 2- or 3-dimensional scene being rendered, but in the 2D space of the screen.
    For example: scale bars, colormap bars, etc.

    Attributes
    ----------
    position : CanvasPosition
        The position of the overlay in the canvas.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    position: CanvasPosition | tuple[int, int] = CanvasPosition.BOTTOM_RIGHT
    blending: Blending = Blending.TRANSLUCENT_NO_DEPTH


class SceneOverlay(Overlay):
    """
    Scene overlay model.

    Scene overlays live in the 2- or 3-dimensional space of the rendered data.
    For example: bounding boxes, data grids, etc.

    Attributes
    ----------
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    blending: Blending = Blending.TRANSLUCENT
