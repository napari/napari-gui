from collections.abc import Callable

from napari.layers.base._base_constants import Mode
from napari.layers.surface.surface import Surface
from napari.layers.utils.layer_utils import (
    register_layer_action,
    register_layer_attr_action,
)
from napari.utils.translations import trans


def register_surface_action(
    description: str, repeatable: bool = False
) -> Callable[[Callable], Callable]:
    return register_layer_action(Surface, description, repeatable)


def register_surface_mode_action(
    description: str,
) -> Callable[[Callable], Callable]:
    return register_layer_attr_action(Surface, description, 'mode')


@register_surface_mode_action(trans._('Transform'))
def activate_surface_transform_mode(layer: Surface) -> None:
    layer.mode = str(Mode.TRANSFORM)


@register_surface_mode_action(trans._('Pan/zoom'))
def activate_surface_pan_zoom_mode(layer: Surface) -> None:
    layer.mode = str(Mode.PAN_ZOOM)


surface_fun_to_mode = [
    (activate_surface_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_surface_transform_mode, Mode.TRANSFORM),
]
