"""Scale Bar overlay."""
import bisect

import numpy as np

from ...components._viewer_constants import Position
from ...settings import get_settings
from ...utils._units import PREFERRED_VALUES, get_unit_registry
from ...utils.colormaps.standardize_color import transform_color
from ...utils.theme import get_theme
from ..visuals.scale_bar import ScaleBar
from .base import VispyBaseOverlay


class VispyScaleBarOverlay(VispyBaseOverlay):
    """Scale bar in world coordinates."""

    def __init__(self, overlay):
        node = ScaleBar()
        super().__init__(overlay, node)

        self.overlay.events.colored.connect(self._on_data_change)
        self.overlay.events.ticks.connect(self._on_data_change)
        self.overlay.events.box_color.connect(self._on_data_change)
        self.overlay.events.color.connect(self._on_data_change)
        self.overlay.events.zoom.connect(self._on_zoom_change)
        self.overlay.events.font_size.connect(self._on_text_change)
        self.overlay.events.unit.connect(self._on_unit_change)
        self.overlay.events.box.connect(self._on_box_change)

        get_settings().appearance.events.theme.connect(self._on_data_change)

        self._on_visible_change()
        self._on_data_change()
        self._on_unit_change()
        self._on_position_change()

    def _on_unit_change(self):
        unit = self.overlay.unit
        self._quantity = get_unit_registry()(unit)
        self._on_zoom_change(force=True)

    def _calculate_best_length(self, desired_length: float):
        """Calculate new quantity based on the pixel length of the bar.

        Parameters
        ----------
        desired_length : float
            Desired length of the scale bar in world size.

        Returns
        -------
        new_length : float
            New length of the scale bar in world size based
            on the preferred scale bar value.
        new_quantity : pint.Quantity
            New quantity with abbreviated base unit.
        """
        current_quantity = self._quantity * desired_length
        # convert the value to compact representation
        new_quantity = current_quantity.to_compact()
        # calculate the scaling factor taking into account any conversion
        # that might have occurred (e.g. um -> cm)
        factor = current_quantity / new_quantity

        # select value closest to one of our preferred values
        index = bisect.bisect_left(PREFERRED_VALUES, new_quantity.magnitude)
        if index > 0:
            # When we get the lowest index of the list, removing -1 will
            # return the last index.
            index -= 1
        new_value = PREFERRED_VALUES[index]

        # get the new pixel length utilizing the user-specified units
        new_length = (
            (new_value * factor) / self._quantity.magnitude
        ).magnitude
        new_quantity = new_value * new_quantity.units
        return new_length, new_quantity

    def _on_zoom_change(self, *, force: bool = False):
        """Update axes length based on zoom scale."""

        # If scale has not changed, do not redraw
        scale = 1 / self.overlay.zoom
        if abs(np.log10(self._scale) - np.log10(scale)) < 1e-4 and not force:
            return
        self._scale = scale

        scale_canvas2world = self._scale
        target_canvas_pixels = self._target_length
        # convert desired length to world size
        target_world_pixels = scale_canvas2world * target_canvas_pixels

        # calculate the desired length as well as update the value and units
        target_world_pixels_rounded, new_dim = self._calculate_best_length(
            target_world_pixels
        )
        target_canvas_pixels_rounded = (
            target_world_pixels_rounded / scale_canvas2world
        )
        scale = target_canvas_pixels_rounded

        sign = (
            -1
            if self.overlay.position
            in [Position.TOP_RIGHT, Position.BOTTOM_RIGHT]
            else 1
        )

        # Update scalebar and text
        self.node.line.transform.scale = [sign * scale, 1, 1, 1]
        self.node.text.text = f'{new_dim:~}'

    def _on_data_change(self):
        """Change color and data of scale bar and box."""
        color = self.overlay.color
        box_color = self.overlay.box_color

        if not self.overlay.colored:
            if self.overlay.box:
                # The box is visible - set the scale bar color to the negative of the
                # box color.
                color = 1 - box_color
                color[-1] = 1
            else:
                # set scale color negative of theme background.
                # the reason for using the `as_hex` here is to avoid
                # `UserWarning` which is emitted when RGB values are above 1
                background_color = get_theme(
                    get_settings().appearance.theme, False
                ).canvas.as_hex()
                background_color = transform_color(background_color)[0]
                color = np.subtract(1, background_color)
                color[-1] = background_color[-1]

        self.node.set_data(color, box_color, self.overlay.ticks)

    def _on_visible_change(self):
        """Change visibility of scale bar."""
        self.node.visible = self.overlay.visible

    def _on_box_change(self):
        self.node.text.visible = self.overlay.box

    def _on_text_change(self):
        """Update text information"""
        self.node.text.font_size = self.overlay.font_size
