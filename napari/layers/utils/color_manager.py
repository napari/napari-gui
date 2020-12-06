from dataclasses import field

import numpy as np

from ...utils.colormaps import CategoricalColormap
from ...utils.colormaps.color_transformations import (
    ColorType,
    normalize_and_broadcast_colors,
    transform_color_with_defaults,
)
from ...utils.events.dataclass import Property, evented_dataclass
from ._color_manager_constants import ColorMode
from ._color_manager_utils import is_color_mapped
from .layer_utils import guess_continuous


def create_color_cycle(color_cycle):

    return CategoricalColormap(
        color_cycle, use_cycle=True, fallback_color='black'
    )


@evented_dataclass(events=True, properties=True)
class ColorManager:
    """Colors for a display property

    Parameters
    ----------
    colors : np.ndarray
        The RGBA color for each data entry
    mode : ColorMode
        Color setting mode. Should be one of the following:
        DIRECT (default mode) allows each point to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
    color_cycle_map : dict
        Mapping of categorical property values to colors

    """

    colors: np.ndarray = np.empty((0, 4))
    mode: Property[ColorMode, str, None] = ColorMode.DIRECT
    color_property: str = ''
    color_cycle: Property[
        CategoricalColormap, None, CategoricalColormap
    ] = np.array([[0, 0, 0, 1]])
    color_cycle_map: dict = field(default_factory=dict)

    def set_color(self, color: ColorType, n_colors: int, properties: dict):
        """ Set the face_color or edge_color property

        Parameters
        ----------
        color : (N, 4) array or str
            The new color. If an array, color should be an
            Nx4 RGBA array for N colors or a 1x4 RGBA array
            that gets broadcast to N colors.
        n_colors:
            The total number of colors that should be created.
        """
        if is_color_mapped(color, properties):
            if guess_continuous(properties[color]):
                self._color_mode = ColorMode.COLORMAP
            else:
                self._color_mode = ColorMode.CYCLE
                self._color_property = color
            self.refresh_colors(properties=properties)
        else:
            transformed_color = transform_color_with_defaults(
                num_entries=n_colors,
                colors=color,
                elem_name="color",
                default="white",
            )
            colors = normalize_and_broadcast_colors(
                n_colors, transformed_color
            )
            self.colors = colors

    def refresh_colors(
        self, properties: dict, update_color_mapping: bool = False
    ):
        """Calculate and update face or edge colors if using a cycle or color map

        Parameters
        ----------
        properties : dict
            The layer properties to map the colors against.
        update_color_mapping : bool
            If set to True, the function will recalculate the color cycle map
            or colormap (whichever is being used). If set to False, the function
            will use the current color cycle map or color map. For example, if you
            are adding/modifying points and want them to be colored with the same
            mapping as the other points (i.e., the new points shouldn't affect
            the color cycle map or colormap), set update_color_mapping=False.
            Default value is False.
        """

        if self._color_mode == ColorMode.CYCLE:
            color_properties = properties[self.color_property]

            colors = self.color_cycle.map(color_properties)
            if len(colors) == 0:
                colors = np.empty((0, 4))
            self.colors = colors

        # elif self._color_mode == ColorMode.COLORMAP:
        #     color_property = getattr(self, f'_{attribute}_color_property')
        #     color_properties = self.properties[color_property]
        #     if len(color_properties) > 0:
        #         contrast_limits = getattr(self, f'{attribute}_contrast_limits')
        #         colormap = getattr(self, f'{attribute}_colormap')
        #         if update_color_mapping or contrast_limits is None:
        #
        #             colors, contrast_limits = map_property(
        #                 prop=color_properties, colormap=colormap
        #             )
        #             setattr(
        #                 self, f'{attribute}_contrast_limits', contrast_limits,
        #             )
        #         else:
        #
        #             colors, _ = map_property(
        #                 prop=color_properties,
        #                 colormap=colormap,
        #                 contrast_limits=contrast_limits,
        #             )
        #     else:
        #         colors = np.empty((0, 4))
        #     setattr(self, f'_{attribute}_color', colors)

        self.events.colors()
