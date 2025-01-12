from typing import TYPE_CHECKING

from vispy.scene.visuals import create_visual_node
from vispy.visuals.image import ImageVisual
from vispy.visuals.shaders import Function, FunctionChain

from napari._vispy.visuals.util import TextureMixin

if TYPE_CHECKING:
    from vispy.visuals.visual import VisualView


class LabelVisual(TextureMixin, ImageVisual):
    """Visual subclass displaying a 2D array of labels."""

    def _build_color_transform(self) -> FunctionChain:
        """Build the color transform function chain."""
        funcs = [
            Function(self._func_templates['red_to_luminance']),
            Function(self.cmap.glsl_map),
        ]

        return FunctionChain(
            funcs=funcs,
        )


BaseLabel = create_visual_node(LabelVisual)


class LabelNode(BaseLabel):  # type: ignore [valid-type,misc]
    def _compute_bounds(
        self, axis: int, view: 'VisualView'
    ) -> tuple[float, float] | None:
        if self._data is None:
            return None
        elif axis > 1:  # noqa: RET505
            return 0, 0
        else:
            return 0, self.size[axis]
