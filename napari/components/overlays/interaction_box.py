from napari.components.overlays.base import SceneOverlay
from napari.layers.utils.interaction_box import (
    InteractionBoxHandle,
    calculate_bounds_from_contained_points,
)


class SelectionBoxOverlay(SceneOverlay):
    """A box that can be used to select and transform objects.

    Attributes
    ----------
    bounds : 2-tuple of 2-tuples
        Corners at top left and bottom right in layer coordinates.
    handles : bool
        Whether to show the handles for transfomation or just the box.
    selected_handle : Optional[InteractionBoxHandle]
        The currently selected handle.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    bounds: tuple[tuple[float, float], tuple[float, float]] = ((0, 0), (0, 0))
    handles: bool = False
    selected_handle: InteractionBoxHandle | None = None

    def update_from_points(self, points):
        """Create as a bounding box of the given points"""
        self.bounds = calculate_bounds_from_contained_points(points)


class TransformBoxOverlay(SceneOverlay):
    """A box that can be used to transform layers.

    Attributes
    ----------
    selected_handle : Optional[InteractionBoxHandle]
        The currently selected handle.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    selected_handle: InteractionBoxHandle | None = None
