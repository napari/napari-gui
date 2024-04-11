"""Scale bar model."""

from typing import TYPE_CHECKING, Callable

from napari._pydantic_compat import Field
from napari.components.overlays.base import CanvasOverlay
from napari.utils.color import ColorValue

if TYPE_CHECKING:
    from napari import Viewer


def _default_slice_parse_function(viewer: 'Viewer', dim_idx: int) -> str:
    num = viewer.dims.point[dim_idx]
    name = viewer.dims.axis_labels[dim_idx]
    formatted_num = format(num, '.5f').rstrip('0').rstrip('.')
    return f'{name}={formatted_num}\n'


class SliceTextOverlay(CanvasOverlay):
    """Slice bar indicating size in world coordinates.

    Attributes
    ----------
    colored : bool
        If scale bar are colored or not. If colored then
        default color is magenta. If not colored than
        scale bar color is the opposite of the canvas
        background or the background box.
    color : ColorValue
        Scalebar and text color.
        See ``ColorValue.validate`` for supported values.
    background_color : np.ndarray
        Background color of canvas. If scale bar is not colored
        then it has the color opposite of this color.
    font_size : float
        The font size (in points) of the text.
    box : bool
        If background box is visible or not.
    box_color : Optional[str | array-like]
        Background box color.
        See ``ColorValue.validate`` for supported values.
    text_prefix : str
        The prefix to the slice text overlay, by default 'Current slice:\n'.
    slice_parse_function : Callable[[Viewer, int], str]
        Function to parse the slice information from the viewer.
        The function should take the viewer and the dimension index
        and return a string to be displayed in the overlay.
        By default, the function will display the axis label and the
        current slice value.
        This function is responsible for the line breaks and formatting.
        To ignore a dimension simply return an empty string, ''.
    position : CanvasPosition
        The position of the overlay in the canvas.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    colored: bool = False
    color: ColorValue = Field(default_factory=lambda: ColorValue([1, 0, 1, 1]))
    font_size: float = 10
    box: bool = False
    box_color: ColorValue = Field(
        default_factory=lambda: ColorValue([0, 0, 0, 0.6])
    )
    text_prefix: str = 'Current slice:\n'
    slice_parse_function: Callable[['Viewer', int], str] = Field(
        default_factory=lambda: _default_slice_parse_function
    )
