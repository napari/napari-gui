from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from napari._pydantic_compat import Field, root_validator, validator
from napari.layers.utils._color_manager_constants import ColorMode
from napari.layers.utils.color_manager_utils import (
    _validate_colormap_mode,
    _validate_cycle_mode,
    guess_continuous,
    is_color_mapped,
)
from napari.layers.utils.color_transformations import (
    normalize_and_broadcast_colors,
    transform_color,
    transform_color_with_defaults,
)
from napari.utils.colormaps import Colormap
from napari.utils.colormaps.categorical_colormap import CategoricalColormap
from napari.utils.colormaps.colormap_utils import ColorType, ensure_colormap
from napari.utils.events import EventedModel
from napari.utils.events.custom_types import Array
from napari.utils.translations import trans


@dataclass
class ColorProperties:
    """The property values that are used for setting colors in ColorMode.COLORMAP
    and ColorMode.CYCLE.

    Attributes
    ----------
    name : str
        The name of the property being used.
    values : np.ndarray
        The array containing the property values.
    current_value : Optional[Any]
        the value for the next item to be added.
    """

    name: str
    values: np.ndarray
    current_value: Any | None = None

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        if val is None:
            color_properties = val
        elif isinstance(val, dict):
            if len(val) == 0:
                color_properties = None
            else:
                try:
                    # ensure the values are a numpy array
                    val['values'] = np.asarray(val['values'])
                    color_properties = cls(**val)
                except ValueError as e:
                    raise ValueError(
                        trans._(
                            'color_properties dictionary should have keys: name, values, and optionally current_value',
                            deferred=True,
                        )
                    ) from e

        elif isinstance(val, cls):
            color_properties = val
        else:
            raise TypeError(
                trans._(
                    'color_properties should be None, a dict, or ColorProperties object',
                    deferred=True,
                )
            )

        return color_properties

    def _json_encode(self):
        return {
            'name': self.name,
            'values': self.values.tolist(),
            'current_value': self.current_value,
        }

    def __eq__(self, other):
        if isinstance(other, ColorProperties):
            name_eq = self.name == other.name
            values_eq = np.array_equal(self.values, other.values)
            current_value_eq = np.array_equal(
                self.current_value, other.current_value
            )

            return np.all([name_eq, values_eq, current_value_eq])

        return False


class ColorManager(EventedModel):
    """A class for controlling the display colors for annotations in napari.

    Attributes
    ----------
    current_color : Optional[np.ndarray]
        A (4,) color array for the color of the next items to be added.
    mode : ColorMode
        The mode for setting colors.

        ColorMode.DIRECT: colors are set by passing color values to ColorManager.colors
        ColorMode.COLORMAP: colors are set via the continuous_colormap applied to the
                            color_properties
        ColorMode.CYCLE: colors are set vie the categorical_colormap appied to the
                         color_properties. This should be used for categorical
                         properties only.
     color_properties : Optional[ColorProperties]
        The property values that are used for setting colors in ColorMode.COLORMAP
        and ColorMode.CYCLE. The ColorProperties dataclass has 3 fields: name,
        values, and current_value. name (str) is the name of the property being used.
        values (np.ndarray) is an array containing the property values.
        current_value contains the value for the next item to be added. color_properties
        can be set as either a ColorProperties object or a dictionary where the keys are
        the field values and the values are the field values (i.e., a dictionary that would
        be valid in ColorProperties(**input_dictionary) ).
    continuous_colormap : Colormap
        The napari colormap object used in ColorMode.COLORMAP mode. This can also be set
        using the name of a known colormap as a string.
    contrast_limits : Tuple[float, float]
        The min and max value for the colormap being applied to the color_properties
        in ColorMonde.COLORMAP mode. Set as a tuple (min, max).
    categorical_colormap : CategoricalColormap
        The napari CategoricalColormap object used in ColorMode.CYCLE mode.
        To set a direct mapping between color_property values and colors,
        pass a dictionary where the keys are the property values and the
        values are colors (either string names or (4,) color arrays).
        To use a color cycle, pass a list or array of colors. You can also
        pass the CategoricalColormap keyword arguments as a dictionary.
    colors : np.ndarray
        The colors in a Nx4 color array, where N is the number of colors.
    """

    # fields
    current_color: Array[float, (4,)] | None = None
    color_mode: ColorMode = ColorMode.DIRECT
    color_properties: ColorProperties | None = None
    continuous_colormap: Colormap = ensure_colormap('viridis')
    contrast_limits: tuple[float, float] | None = None
    categorical_colormap: CategoricalColormap = CategoricalColormap.from_array(
        [0, 0, 0, 1]
    )
    colors: Array[float, (-1, 4)] = Field(
        default_factory=lambda: np.empty((0, 4))
    )

    # validators
    @validator('continuous_colormap', pre=True, allow_reuse=True)
    def _ensure_continuous_colormap(cls, v):
        return ensure_colormap(v)

    @validator('colors', pre=True, allow_reuse=True)
    def _ensure_color_array(cls, v):
        if len(v) > 0:
            return transform_color(v)

        return np.empty((0, 4))

    @validator('current_color', pre=True, allow_reuse=True)
    def _coerce_current_color(cls, v):
        if v is None:
            return v
        if len(v) == 0:
            return None

        return transform_color(v)[0]

    @root_validator(allow_reuse=True)
    def _validate_colors(cls, values):
        color_mode = values['color_mode']
        if color_mode == ColorMode.CYCLE:
            colors, values = _validate_cycle_mode(values)
        elif color_mode == ColorMode.COLORMAP:
            colors, values = _validate_colormap_mode(values)
        else:  # color_mode == ColorMode.DIRECT:
            colors = values['colors']

        # set the current color to the last color/property value
        # if it wasn't already set
        if values.get('current_color') is None and len(colors) > 0:
            values['current_color'] = colors[-1]
            if color_mode in [ColorMode.CYCLE, ColorMode.COLORMAP]:
                property_values = values['color_properties']
                property_values.current_value = property_values.values[-1]
                values['color_properties'] = property_values

        values['colors'] = colors
        return values

    def _set_color(
        self,
        color: ColorType,
        n_colors: int,
        properties: dict[str, np.ndarray],
        current_properties: dict[str, np.ndarray],
    ):
        """Set a color property. This is convenience function

        Parameters
        ----------
        color : (N, 4) array or str
            The value for setting edge or face_color
        n_colors : int
            The number of colors that needs to be set. Typically len(data).
        properties : Dict[str, np.ndarray]
            The layer property values
        current_properties : Dict[str, np.ndarray]
            The layer current property values
        """
        # if the provided color is a string, first check if it is a key in the properties.
        # otherwise, assume it is the name of a color
        if is_color_mapped(color, properties):
            # note that we set ColorProperties.current_value by indexing rather than
            # np.squeeze since the current_property values have shape (1,) and
            # np.squeeze would return an array with shape ().
            # see https://github.com/napari/napari/pull/3110#discussion_r680680779
            self.color_properties = ColorProperties(
                name=color,
                values=properties[color],
                current_value=current_properties[color][0],
            )
            if guess_continuous(properties[color]):
                self.color_mode = ColorMode.COLORMAP
            else:
                self.color_mode = ColorMode.CYCLE
        else:
            transformed_color = transform_color_with_defaults(
                num_entries=n_colors,
                colors=color,
                elem_name='color',
                default='white',
            )
            colors = normalize_and_broadcast_colors(
                n_colors, transformed_color
            )
            self.color_mode = ColorMode.DIRECT
            self.colors = colors

    def _refresh_colors(
        self,
        properties: dict[str, np.ndarray],
        update_color_mapping: bool = False,
    ):
        """Calculate and update colors if using a cycle or color map
        Parameters
        ----------
        properties : Dict[str, np.ndarray]
           The layer properties to use to update the colors.
        update_color_mapping : bool
           If set to True, the function will recalculate the color cycle map
           or colormap (whichever is being used). If set to False, the function
           will use the current color cycle map or color map. For example, if you
           are adding/modifying points and want them to be colored with the same
           mapping as the other points (i.e., the new points shouldn't affect
           the color cycle map or colormap), set update_color_mapping=False.
           Default value is False.
        """
        if self.color_mode in [ColorMode.CYCLE, ColorMode.COLORMAP]:
            property_name = self.color_properties.name
            current_value = self.color_properties.current_value
            property_values = properties[property_name]
            self.color_properties = ColorProperties(
                name=property_name,
                values=property_values,
                current_value=current_value,
            )

            if update_color_mapping is True:
                self.contrast_limits = None
            self.events.color_properties()

    def _add(
        self,
        color: ColorType | None = None,
        n_colors: int = 1,
        update_clims: bool = False,
    ):
        """Add colors
        Parameters
        ----------
        color : Optional[ColorType]
            The color to add. If set to None, the value of self.current_color will be used.
            The default value is None.
        n_colors : int
            The number of colors to add. The default value is 1.
        update_clims : bool
            If in colormap mode, update the contrast limits when adding the new values
            (i.e., reset the range to 0-new_max_value).
        """
        if self.color_mode == ColorMode.DIRECT:
            new_color = self.current_color if color is None else color
            transformed_color = transform_color_with_defaults(
                num_entries=n_colors,
                colors=new_color,
                elem_name='color',
                default='white',
            )
            broadcasted_colors = normalize_and_broadcast_colors(
                n_colors, transformed_color
            )
            self.colors = np.concatenate((self.colors, broadcasted_colors))
        else:
            # add the new value color_properties
            color_property_name = self.color_properties.name
            current_value = self.color_properties.current_value
            if color is None:
                color = current_value
            new_color_property_values = np.concatenate(
                (self.color_properties.values, np.repeat(color, n_colors)),
                axis=0,
            )
            self.color_properties = ColorProperties(
                name=color_property_name,
                values=new_color_property_values,
                current_value=current_value,
            )

            if update_clims and self.color_mode == ColorMode.COLORMAP:
                self.contrast_limits = None

    def _remove(self, indices_to_remove: set | list | np.ndarray):
        """Remove the indicated color elements
        Parameters
        ----------
        indices_to_remove : set, list, np.ndarray
            The indices of the text elements to remove.
        """
        selected_indices = list(indices_to_remove)
        if len(selected_indices) > 0:
            if self.color_mode == ColorMode.DIRECT:
                self.colors = np.delete(self.colors, selected_indices, axis=0)
            else:
                # remove the color_properties
                color_property_name = self.color_properties.name
                current_value = self.color_properties.current_value
                new_color_property_values = np.delete(
                    self.color_properties.values, selected_indices
                )
                self.color_properties = ColorProperties(
                    name=color_property_name,
                    values=new_color_property_values,
                    current_value=current_value,
                )

    def _paste(self, colors: np.ndarray, properties: dict[str, np.ndarray]):
        """Append colors to the ColorManager. Uses the color values if
        in direct mode and the properties in colormap or cycle mode.

        This method is for compatibility with the paste functionality
        in the layers.

        Parameters
        ----------
        colors : np.ndarray
            The (Nx4) color array of color values to add. These values are
            only used if the color mode is direct.
        properties : Dict[str, np.ndarray]
            The property values to add. These are used if the color mode
            is colormap or cycle.
        """
        if self.color_mode == ColorMode.DIRECT:
            self.colors = np.concatenate(
                (self.colors, transform_color(colors))
            )
        else:
            color_property_name = self.color_properties.name
            current_value = self.color_properties.current_value
            old_properties = self.color_properties.values
            values_to_add = properties[color_property_name]
            new_color_property_values = np.concatenate(
                (old_properties, values_to_add),
                axis=0,
            )

            self.color_properties = ColorProperties(
                name=color_property_name,
                values=new_color_property_values,
                current_value=current_value,
            )

    def _update_current_properties(
        self, current_properties: dict[str, np.ndarray]
    ):
        """This is updates the current_value of the color_properties when the
        layer current_properties is updated.

        This is a convenience method that is generally only called by the layer.

        Parameters
        ----------
        current_properties : Dict[str, np.ndarray]
            The new current property values
        """
        if self.color_properties is not None:
            current_property_name = self.color_properties.name
            current_property_values = self.color_properties.values
            if current_property_name in current_properties:
                # note that we set ColorProperties.current_value by indexing rather than
                # np.squeeze since the current_property values have shape (1,) and
                # np.squeeze would return an array with shape ().
                # see https://github.com/napari/napari/pull/3110#discussion_r680680779
                new_current_value = current_properties[current_property_name][
                    0
                ]

                if new_current_value != self.color_properties.current_value:
                    self.color_properties = ColorProperties(
                        name=current_property_name,
                        values=current_property_values,
                        current_value=new_current_value,
                    )

    def _update_current_color(
        self, current_color: np.ndarray, update_indices: list | None = None
    ):
        """Update the current color and update the colors if requested.

        This is a convenience method and is generally called by the layer.

        Parameters
        ----------
        current_color : np.ndarray
            The new current color value.
        update_indices : list
            The indices of the color elements to update.
            If the list has length 0, no colors are updated.
            If the ColorManager is not in DIRECT mode, updating the values
            will change the mode to DIRECT.
        """
        if update_indices is None:
            update_indices = []
        self.current_color = transform_color(current_color)[0]
        if update_indices:
            self.color_mode = ColorMode.DIRECT
            cur_colors = self.colors.copy()
            cur_colors[update_indices] = self.current_color
            self.colors = cur_colors

    @classmethod
    def _from_layer_kwargs(
        cls,
        colors: dict | str | np.ndarray,
        properties: dict[str, np.ndarray],
        n_colors: int | None = None,
        continuous_colormap: str | Colormap | None = None,
        contrast_limits: tuple[float, float] | None = None,
        categorical_colormap: CategoricalColormap
        | list
        | np.ndarray
        | None = None,
        color_mode: ColorMode | str | None = None,
        current_color: np.ndarray | None = None,
        default_color_cycle: ColorType = None,
    ):
        """Initialize a ColorManager object from layer kwargs. This is a convenience
        function to coerce possible inputs into ColorManager kwargs

        """
        if default_color_cycle is None:
            default_color_cycle = np.array([1, 1, 1, 1])

        properties = {k: np.asarray(v) for k, v in properties.items()}
        if isinstance(colors, dict):
            # if the kwargs are passed as a dictionary, unpack them
            color_values = colors.get('colors', None)
            current_color = colors.get('current_color', current_color)
            color_mode = colors.get('color_mode', color_mode)
            color_properties = colors.get('color_properties', None)
            continuous_colormap = colors.get(
                'continuous_colormap', continuous_colormap
            )
            contrast_limits = colors.get('contrast_limits', contrast_limits)
            categorical_colormap = colors.get(
                'categorical_colormap', categorical_colormap
            )

            if isinstance(color_properties, str):
                # if the color properties were given as a property name,
                # coerce into ColorProperties
                try:
                    prop_values = properties[color_properties]
                    prop_name = color_properties
                    color_properties = ColorProperties(
                        name=prop_name, values=prop_values
                    )
                except KeyError as e:
                    raise KeyError(
                        trans._(
                            'if color_properties is a string, it should be a property name',
                            deferred=True,
                        )
                    ) from e
        else:
            color_values = colors
            color_properties = None

        if categorical_colormap is None:
            categorical_colormap = deepcopy(default_color_cycle)

        color_kwargs = {
            'categorical_colormap': categorical_colormap,
            'continuous_colormap': continuous_colormap,
            'contrast_limits': contrast_limits,
            'current_color': current_color,
            'n_colors': n_colors,
        }

        if color_properties is None:
            if is_color_mapped(color_values, properties):
                if n_colors == 0:
                    color_properties = ColorProperties(
                        name=color_values,
                        values=np.empty(
                            0, dtype=properties[color_values].dtype
                        ),
                        current_value=properties[color_values][0],
                    )
                else:
                    color_properties = ColorProperties(
                        name=color_values, values=properties[color_values]
                    )
                if color_mode is None:
                    if guess_continuous(color_properties.values):
                        color_mode = ColorMode.COLORMAP
                    else:
                        color_mode = ColorMode.CYCLE

                color_kwargs.update(
                    {
                        'color_mode': color_mode,
                        'color_properties': color_properties,
                    }
                )

            else:
                # direct mode
                if n_colors == 0:
                    if current_color is None:
                        current_color = transform_color(color_values)[0]
                    color_kwargs.update(
                        {
                            'color_mode': ColorMode.DIRECT,
                            'current_color': current_color,
                        }
                    )
                else:
                    transformed_color = transform_color_with_defaults(
                        num_entries=n_colors,
                        colors=color_values,
                        elem_name='colors',
                        default='white',
                    )
                    colors = normalize_and_broadcast_colors(
                        n_colors, transformed_color
                    )
                    color_kwargs.update(
                        {'color_mode': ColorMode.DIRECT, 'colors': colors}
                    )
        else:
            color_kwargs.update(
                {
                    'color_mode': color_mode,
                    'color_properties': color_properties,
                }
            )

        return cls(**color_kwargs)
