import bisect
from decimal import Decimal
from math import floor, log

import numpy as np
import pint

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.scale_bar import ScaleBar
from napari.settings import get_settings
from napari.utils._units import PREFERRED_VALUES, get_unit_registry
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.theme import get_theme


class VispyScaleBarOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    """Scale bar in world coordinates."""

    def __init__(self, *, viewer, overlay, parent=None) -> None:
        self._target_length = 150.0
        self._scale = 1
        self._unit: pint.Unit

        super().__init__(
            node=ScaleBar(), viewer=viewer, overlay=overlay, parent=parent
        )
        self.x_size = 150  # will be updated on zoom anyways
        # need to change from defaults because the anchor is in the center
        self.y_offset = 20
        # TODO: perhaps change name as y_size does not indicate bottom offset.
        self.y_size = 5

        # In the super().__init__ we see node is scale bar, need to connect its parent, canvas
        self.node.events.parent_change.connect(self._on_parent_change)

        self.overlay.events.box.connect(self._on_box_change)
        self.overlay.events.box_color.connect(self._on_data_change)
        self.overlay.events.color.connect(self._on_data_change)
        self.overlay.events.colored.connect(self._on_data_change)
        self.overlay.events.font_size.connect(self._on_text_change)
        self.overlay.events.ticks.connect(self._on_data_change)
        self.overlay.events.unit.connect(self._on_unit_change)
        self.overlay.events.length.connect(self._on_length_change)

        self.viewer.events.theme.connect(self._on_data_change)
        self.viewer.camera.events.zoom.connect(self._on_zoom_change)

        self.reset()

    def _on_parent_change(self, event):
        """Connect the canvas resize event to scale bar callback function(s)."""
        if event.new and self.node.canvas:
            event.new.canvas.events.resize.connect(
                self._scale_scalebar_on_canvas_resize
            )
            event.new.canvas.events.resize.connect(self._scale_font_size)

    def _scale_font_size(self, event):
        """Scale the font size in response to a canvas resize"""
        self.node.text.font_size = (
            event.source.size[1]
            / get_settings().experimental.scale_bar_font_size
        )

    def _scale_scalebar_on_canvas_resize(self, event):
        self._target_length = (
            event.source.size[0] / get_settings().experimental.scale_bar_length
        )
        self.y_size = event.source.size[1] / 40
        self.x_size = event.source.size[0] / 20
        self.node.line._width = event.source.size[1] / 100
        self._on_zoom_change(force=True)

    def _on_unit_change(self):
        self._unit = get_unit_registry()(self.overlay.unit)
        self._on_zoom_change(force=True)

    def _on_length_change(self):
        self._on_zoom_change(force=True)

    def _calculate_best_length(
        self, desired_length: float
    ) -> tuple[float, pint.Quantity]:
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
        current_quantity = self._unit * desired_length
        # convert the value to compact representation
        new_quantity = current_quantity.to_compact()
        # calculate the scaling factor taking into account any conversion
        # that might have occurred (e.g. um -> cm)
        factor = current_quantity / new_quantity

        # select value closest to one of our preferred values and also
        # validate if quantity is dimensionless and lower than 1 to prevent
        # the scale bar to extend beyond the canvas when zooming.
        # If the value falls in those conditions, we use the corresponding
        # preferred value but scaled to take into account the actual value
        # magnitude. See https://github.com/napari/napari/issues/5914
        magnitude_1000 = floor(log(new_quantity.magnitude, 1000))
        scaled_magnitude = new_quantity.magnitude * 1000 ** (-magnitude_1000)
        index = bisect.bisect_left(PREFERRED_VALUES, scaled_magnitude)
        if index > 0:
            # When we get the lowest index of the list, removing -1 will
            # return the last index.
            index -= 1
        new_value: float = PREFERRED_VALUES[index]
        if new_quantity.dimensionless and new_quantity.magnitude < 1:
            # using Decimal is necessary to avoid `4.999999e-6`
            # at really small scale.
            new_value = float(
                Decimal(new_value) * Decimal(1000) ** magnitude_1000
            )

        # get the new pixel length utilizing the user-specified units
        new_length = (
            (new_value * factor) / (1 * self._unit).magnitude
        ).magnitude
        new_quantity = new_value * new_quantity.units
        return new_length, new_quantity

    def _on_zoom_change(self, *, force: bool = False):
        """Update axes length based on zoom scale."""

        # If scale has not changed, do not redraw
        scale = 1 / self.viewer.camera.zoom
        if abs(np.log10(self._scale) - np.log10(scale)) < 1e-4 and not force:
            return
        self._scale = scale

        scale_canvas2world = self._scale
        target_canvas_pixels = self._target_length
        # convert desired length to world size
        target_world_pixels = scale_canvas2world * target_canvas_pixels

        # If length is set, use that value to calculate the scale bar length
        if self.overlay.length is not None:
            target_canvas_pixels = self.overlay.length / scale_canvas2world
            new_dim = self.overlay.length * self._unit.units
        else:
            # calculate the desired length as well as update the value and units
            target_world_pixels_rounded, new_dim = self._calculate_best_length(
                target_world_pixels
            )
            target_canvas_pixels = (
                target_world_pixels_rounded / scale_canvas2world
            )

        scale = target_canvas_pixels

        # Update scalebar and text
        self.node.transform.scale = [scale, 1, 1, 1]
        self.node.text.text = f'{new_dim:g~#P}'
        self.x_size = scale  # needed to offset properly
        super()._on_position_change()

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
                if (
                    self.node.parent is not None
                    and self.node.parent.canvas.bgcolor
                ):
                    background_color = self.node.parent.canvas.bgcolor.rgba
                else:
                    background_color = get_theme(
                        self.viewer.theme
                    ).canvas.as_hex()
                    background_color = transform_color(background_color)[0]
                color = np.subtract(1, background_color)
                color[-1] = background_color[-1]

        self.node.set_data(color, self.overlay.ticks)
        self.node.box.color = box_color

    def _on_box_change(self):
        self.node.box.visible = self.overlay.box

    def _on_text_change(self):
        """Update text information"""
        # update the dpi scale factor to account for screen dpi
        # because vispy scales pixel height of text by screen dpi
        if self.node.text.transforms.dpi:
            # use 96 as the napari reference dpi for historical reasons
            dpi_scale_factor = 96 / self.node.text.transforms.dpi
        else:
            dpi_scale_factor = 1

        self.node.text.font_size = self.overlay.font_size * dpi_scale_factor
        # ensure we recalculate the y_offset from the text size when at top of canvas
        if 'top' in self.overlay.position:
            self._on_position_change()

    def _on_position_change(self, event=None):
        # prevent the text from being cut off by shifting down
        if 'top' in self.overlay.position:
            # convert font_size to logical pixels as vispy does
            # in vispy/visuals/text/text.py
            # 72 is the vispy reference dpi
            # 96 dpi is used as the napari reference dpi
            font_logical_pix = self.overlay.font_size * 96 / 72
            # 7 is base value for the default 10 font size
            self.y_offset = 7 + font_logical_pix
        else:
            self.y_offset = 20
        super()._on_position_change()

    def _on_visible_change(self):
        # ensure that dpi is updated when the scale bar is visible
        self._on_text_change()
        return super()._on_visible_change()

    def reset(self):
        super().reset()
        self._on_unit_change()
        self._on_data_change()
        self._on_box_change()
        self._on_text_change()
        self._on_length_change()
