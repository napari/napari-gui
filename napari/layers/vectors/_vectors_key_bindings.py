from collections.abc import Callable

from napari.layers.base._base_constants import Mode
from napari.layers.utils.layer_utils import (
    register_layer_action,
    register_layer_attr_action,
)
from napari.layers.vectors.vectors import Vectors
from napari.utils.translations import trans


def register_vectors_action(
    description: str, repeatable: bool = False
) -> Callable[[Callable], Callable]:
    return register_layer_action(Vectors, description, repeatable)


def register_vectors_mode_action(
    description: str,
) -> Callable[[Callable], Callable]:
    return register_layer_attr_action(Vectors, description, 'mode')


@register_vectors_mode_action(trans._('Transform'))
def activate_vectors_transform_mode(layer: Vectors) -> None:
    layer.mode = str(Mode.TRANSFORM)


@register_vectors_mode_action(trans._('Pan/zoom'))
def activate_vectors_pan_zoom_mode(layer: Vectors) -> None:
    layer.mode = str(Mode.PAN_ZOOM)


vectors_fun_to_mode = [
    (activate_vectors_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_vectors_transform_mode, Mode.TRANSFORM),
]
