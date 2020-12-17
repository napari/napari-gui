from dataclasses import InitVar
from typing import Optional, Tuple, Union

import numpy as np

from ...utils.colormaps import CategoricalColormap, Colormap, ensure_colormap
from ...utils.colormaps.color_transformations import (
    ColorType,
    normalize_and_broadcast_colors,
    transform_color_with_defaults,
)
from ...utils.colormaps.standardize_color import transform_single_color
from ...utils.events.dataclass import Property, evented_dataclass
from ._color_manager_constants import ColorMode
from ._color_manager_utils import is_color_mapped
from .layer_utils import guess_continuous, map_property


@evented_dataclass
class ColorManager:
    """Colors for a display property

    Parameters
    ----------
    values : np.ndarray
        The RGBA color for each data entry
    current_color : np.ndarray
        The value of the next color to be added
    mode : ColorMode
        Color setting mode. Should be one of the following:
        DIRECT (default mode) allows each point to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
    color_property : str
        The name of the property to be used to set colors. This is not used when mode is ColorMode.DIRECT.
    continuous_colormap : Colormap
        The colormap to be used to map continuous property values to conlors.
        This is used when mode is ColorMode.COLORMAP. The default value is 'viridis'
    continuous_contrast_limits : Optional[Tuple[float, float]]
        The contrast limits used when scaling the property values for use with the continuous_colormap.
        This is used when mode is ColorMode.COLORMAP. When set to None, the limits are set to the
        min and max property values. The default value is None.
    categorical_colormap : CategoricalColormap
        This is the colormap used to map categorical property values to colors. The colormap can be
        provided as an array, which will be used as a color cycle or a dictionary where the keys
        are the property values and the values are the colors the property values should be mapped to.
        This is used when mode is ColorMode.CYCLE. The default value maps all property values to black.
    """

    colors: InitVar[ColorType] = 'black'
    n_colors: InitVar[int] = 0
    properties: InitVar[Optional[dict]] = None

    values: Optional[np.ndarray] = None
    current_color: Optional[np.ndarray] = None
    mode: Property[ColorMode, str, None] = ColorMode.DIRECT
    color_property: str = ''
    continuous_colormap: Property[Colormap, None, ensure_colormap] = 'viridis'
    continuous_contrast_limits: Optional[Tuple[float, float]] = None
    categorical_colormap: Property[
        CategoricalColormap, None, CategoricalColormap
    ] = np.array([[0, 0, 0, 1]])

    def __post_init__(self, colors, n_colors, properties):
        if colors is None:
            colors = np.empty((0, 4))
        self.set_color(color=colors, n_colors=n_colors, properties=properties)

        if self._current_color is None:
            self._initialize_current_color(colors, n_colors, properties)

    def _initialize_current_color(
        self, colors: ColorType, n_colors: int, properties: dict
    ):
        """Set the current color based on the number of colors and mode

        Parameters
        ----------
        colors : ColorType
            The colors that the ColorManager was initialized with
        n_colors : int
            The number of colors in the ColorManager
        properties : dict
            The layer properties that were used to initialize the ColorManager
        """
        if n_colors > 0:
            self._current_color = self.values[-1]
        elif n_colors == 0 and properties:
            if self._mode == ColorMode.DIRECT:
                curr_color = transform_color_with_defaults(
                    num_entries=1,
                    colors=colors,
                    elem_name='color',
                    default="white",
                )
            elif self._mode == ColorMode.CYCLE:
                prop_value = properties[self.color_property][0]
                curr_color = self.categorical_colormap.map(prop_value)
            elif self._mode == ColorMode.COLORMAP:
                prop_value = properties[self.color_property][0]
                curr_color = self.continuous_colormap.map(prop_value)
            self._current_color = curr_color
        else:
            self._current_color = transform_single_color(colors)

    def set_color(self, color: ColorType, n_colors: int, properties: dict):
        """ Set the color values

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
                self._mode = ColorMode.COLORMAP
            else:
                self._mode = ColorMode.CYCLE
            self._color_property = color
            if n_colors == 0:
                self.values = np.empty((0, 4))
            else:
                self.refresh_colors(properties=properties)
        else:
            if n_colors == 0:
                self.values = np.empty((0, 4))
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
                self.values = colors
            self._color_mode = ColorMode.DIRECT

    def add(self, color: Optional[ColorType] = None, n_colors: int = 1):
        """Add colors

        Parameters
        ----------
        color : Optional[ColorType]
            The color to add. If set to None, the value of self.current_color will be used.
            The default value is None.
        n_colors : int
            The number of colors to add. The default value is 1.
        """
        if color is None:
            new_color = self.current_color
        else:
            if self._mode == ColorMode.DIRECT:
                new_color = color
            elif self._mode == ColorMode.CYCLE:
                if isinstance(color, str):
                    color = [color]
                new_color = self.categorical_colormap.map(color)
            elif self._mode == ColorMode.COLORMAP:
                new_color = self.continuous_colormap.map(color)
        transformed_color = transform_color_with_defaults(
            num_entries=n_colors,
            colors=new_color,
            elem_name="color",
            default="white",
        )
        broadcasted_colors = normalize_and_broadcast_colors(
            n_colors, transformed_color
        )
        self._values = np.concatenate((self.values, broadcasted_colors))

    def remove(self, indices_to_remove: Union[set, list, np.ndarray]):
        """Remove the indicated color elements

        Parameters
        ----------
        indices_to_remove : set, list, np.ndarray
            The indices of the text elements to remove.
        """
        selected_indices = list(indices_to_remove)
        if len(selected_indices) > 0:
            self._values = np.delete(self.values, selected_indices, axis=0)

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
        if self._mode in [ColorMode.CYCLE, ColorMode.COLORMAP]:
            if self._mode == ColorMode.CYCLE:
                color_properties = properties[self.color_property]
                colors = self.categorical_colormap.map(color_properties)

            elif self._mode == ColorMode.COLORMAP:
                color_properties = properties[self.color_property]
                if len(color_properties) > 0:
                    if (
                        update_color_mapping
                        or self.continuous_contrast_limits is None
                    ):
                        colors, contrast_limits = map_property(
                            prop=color_properties,
                            colormap=self.continuous_colormap,
                        )
                        self.continuous_contrast_limits = contrast_limits
                    else:
                        colors, _ = map_property(
                            prop=color_properties,
                            colormap=self.continuous_colormap,
                            contrast_limits=self.continuous_contrast_limits,
                        )
                else:
                    colors = np.empty((0, 4))
            if len(colors) == 0:
                colors = np.empty((0, 4))
            self.values = colors
            self.events.values()

    def on_current_properties_update(self, current_properties):
        """Updates the current_color value when the current_properties attribute is
        updated on the layer.

        Parameters:
        ----------
        current_properties : dict
            The new value of current_properties.
        """
        if self._mode == ColorMode.CYCLE:
            new_prop_values = current_properties[self.color_property]
            new_color = self.categorical_colormap.map(new_prop_values)
            self.current_color = new_color
        if self._mode == ColorMode.COLORMAP:
            new_prop_values = current_properties[self.color_property]
            new_color = self.continuous_colormap.map(new_prop_values)
            self.current_color = new_color
