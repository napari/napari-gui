import warnings
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Union

import numpy as np
import pandas as pd

from napari._pydantic_compat import PositiveInt, validator
from napari.layers.base._base_constants import Blending
from napari.layers.utils._text_constants import Anchor
from napari.layers.utils._text_utils import get_text_anchors
from napari.layers.utils.color_encoding import (
    ColorArray,
    ColorEncoding,
    ConstantColorEncoding,
)
from napari.layers.utils.layer_utils import _validate_features
from napari.layers.utils.string_encoding import (
    ConstantStringEncoding,
    StringArray,
    StringEncoding,
)
from napari.layers.utils.style_encoding import _get_style_values
from napari.utils.events import Event, EventedModel
from napari.utils.events.custom_types import Array
from napari.utils.translations import trans


class TextManager(EventedModel):
    """Manages properties related to text displayed in conjunction with the layer.

    Parameters
    ----------
    features : Any
        The features table of a layer.
    values : array-like
        The array of strings manually specified.
        .. deprecated:: 0.4.16
            `values` is deprecated. Use `string` instead.
    text : str
        A a property name or a format string containing property names.
        This will be used to fill out string values n_text times using the
        data in properties.
        .. deprecated:: 0.4.16
            `text` is deprecated. Use `string` instead.
    n_text : int
        The number of text elements to initially display, which should match
        the number of elements (e.g. points) in a layer.
        .. deprecated:: 0.4.16
            `n_text` is deprecated. Its value is implied by `features` instead.
    properties: dict
        Stores properties data that will be used to generate strings from the
        given text. Typically comes from a layer.
        .. deprecated:: 0.4.16
            `properties` is deprecated. Use `features` instead.

    Attributes
    ----------
    string : StringEncoding
        Defines the string for each text element.
    values : np.ndarray
        The encoded string values.
    visible : bool
        True if the text should be displayed, false otherwise.
    size : float
        Font size of the text, which must be positive. Default value is 12.
    color : ColorEncoding
        Defines the color for each text element.
    blending : Blending
        The blending mode that determines how RGB and alpha values of the layer
        visual get mixed. Allowed values are 'translucent' and 'additive'.
        Note that 'opaque' blending is not allowed, as it colors the bounding box
        surrounding the text, and if given, 'translucent' will be used instead.
    anchor : Anchor
        The location of the text origin relative to the bounding box.
        Should be 'center', 'upper_left', 'upper_right', 'lower_left', or 'lower_right'.
    translation : np.ndarray
        Offset from the anchor point in data coordinates.
    rotation : float
        Angle of the text elements around the anchor point. Default value is 0.
    """

    string: StringEncoding = ConstantStringEncoding(constant='')
    color: ColorEncoding = ConstantColorEncoding(constant='cyan')
    visible: bool = True
    size: PositiveInt = 12
    blending: Blending = Blending.TRANSLUCENT
    anchor: Anchor = Anchor.CENTER
    # Use a scalar default translation to broadcast to any dimensionality.
    translation: Array[float] = 0
    rotation: float = 0

    def __init__(
        self, text=None, properties=None, n_text=None, features=None, **kwargs
    ) -> None:
        if n_text is not None:
            _warn_about_deprecated_n_text_parameter()
        if properties is not None:
            _warn_about_deprecated_properties_parameter()
            features = _validate_features(properties, num_data=n_text)
        else:
            features = _validate_features(features)
        if 'values' in kwargs:
            _warn_about_deprecated_values_parameter()
            values = kwargs.pop('values')
            if 'string' not in kwargs:
                kwargs['string'] = values
        if text is not None:
            _warn_about_deprecated_text_parameter()
            kwargs['string'] = text
        super().__init__(**kwargs)
        self.apply(features)

    @property
    def values(self):
        return self.string._values

    def __setattr__(self, key, value):
        if key == 'values':
            self.string = value
        else:
            super().__setattr__(key, value)

    def refresh(self, features: Any) -> None:
        """Refresh all encoded values using new layer features.

        Parameters
        ----------
        features : Any
            The features table of a layer.
        """
        self.string._clear()
        self.color._clear()
        self.string._apply(features)
        self.events.values()
        self.color._apply(features)
        # Trigger the main event for vispy layers.
        self.events(Event(type_name='refresh'))

    def refresh_text(self, properties: dict[str, np.ndarray]):
        """Refresh all of the current text elements using updated properties values

        Parameters
        ----------
        properties : Dict[str, np.ndarray]
            The new properties from the layer
        """
        warnings.warn(
            trans._(
                'TextManager.refresh_text is deprecated since 0.4.16. Use TextManager.refresh instead.'
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        features = _validate_features(properties)
        self.refresh(features)

    def add(self, properties: dict, n_text: int):
        """Adds a number of a new text elements.

        Parameters
        ----------
        properties : dict
            The properties to draw the text from
        n_text : int
            The number of text elements to add
        """
        warnings.warn(
            trans._(
                'TextManager.add is deprecated since 0.4.16. Use TextManager.apply instead.'
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        features = pd.DataFrame(
            {
                name: np.repeat(value, n_text, axis=0)
                for name, value in properties.items()
            }
        )
        values = self.string(features)
        self.string._append(values)
        self.events.values()
        colors = self.color(features)
        self.color._append(colors)

    def remove(self, indices_to_remove: range | set | list | np.ndarray):
        """Remove the indicated text elements

        Parameters
        ----------
        indices_to_remove : set, list, np.ndarray
            The indices of the text elements to remove.
        """
        if isinstance(indices_to_remove, set):
            indices_to_remove = list(indices_to_remove)
        self.string._delete(indices_to_remove)
        self.events.values()
        self.color._delete(indices_to_remove)

    def apply(self, features: Any):
        """Applies any encodings to be the same length as the given features,
        generating new values or removing extra values only as needed.

        Parameters
        ----------
        features : Any
            The features table of a layer.
        """
        self.string._apply(features)
        self.events.values()
        self.color._apply(features)

    def _copy(self, indices: list[int]) -> dict:
        """Copies all encoded values at the given indices."""
        return {
            'string': _get_style_values(self.string, indices),
            'color': _get_style_values(self.color, indices, value_ndim=1),
        }

    def _paste(self, *, string: StringArray, color: ColorArray):
        """Pastes encoded values to the end of the existing values."""
        self.string._append(string)
        self.events.values()
        self.color._append(color)

    def compute_text_coords(
        self,
        view_data: np.ndarray,
        ndisplay: int,
        order: tuple[int, ...] | None = None,
    ) -> tuple[np.ndarray, str, str]:
        """Calculate the coordinates for each text element in view

        Parameters
        ----------
        view_data : np.ndarray
            The in view data from the layer
        ndisplay : int
            The number of dimensions being displayed in the viewer
        order : tuple of ints, optional
            The display order of the dimensions in the layer.
            If None, implies ``range(ndisplay)``.

        Returns
        -------
        text_coords : np.ndarray
            The coordinates of the text elements
        anchor_x : str
            The vispy text anchor for the x axis
        anchor_y : str
            The vispy text anchor for the y axis
        """
        anchor_coords, anchor_x, anchor_y = get_text_anchors(
            view_data, ndisplay, self.anchor
        )
        # The translation should either be a scalar or be as long as
        # the dimensionality of the associated layer.
        # We do not have direct knowledge of that dimensionality, but
        # can infer enough information to get the translation coordinates
        # that need to offset the anchor coordinates.
        ndim_coords = min(ndisplay, anchor_coords.shape[1])
        translation = self.translation
        if translation.size > 1:
            if order is None:
                translation = self.translation[-ndim_coords:]
            else:
                order_displayed = list(order[-ndim_coords:])
                translation = self.translation[order_displayed]
        text_coords = anchor_coords + translation
        return text_coords, anchor_x, anchor_y

    def view_text(self, indices_view: np.ndarray) -> np.ndarray:
        """Get the values of the text elements in view

        Parameters
        ----------
        indices_view : (N x 1) np.ndarray
            Indices of the text elements in view

        Returns
        -------
        text : (N x 1) np.ndarray
            Array of text strings for the N text elements in view
        """
        values = _get_style_values(self.string, indices_view)
        return (
            np.broadcast_to(values, len(indices_view))
            if values.ndim == 0
            else values
        )

    def _view_color(self, indices_view: np.ndarray) -> np.ndarray:
        """Get the colors of the text elements at the given indices."""
        return _get_style_values(self.color, indices_view, value_ndim=1)

    @classmethod
    def _from_layer(
        cls,
        *,
        text: Union['TextManager', dict, str, Sequence[str], None],
        features: Any,
    ) -> 'TextManager':
        """Create a TextManager from a layer.

        Parameters
        ----------
        text : Union[TextManager, dict, str, Sequence[str], None]
            An instance of TextManager, a dict that contains some of its state,
            a string that may be a format string or a feature name, or a
            sequence of strings specified manually.
        features : Any
            The features table of a layer.

        Returns
        -------
        TextManager
        """
        if isinstance(text, TextManager):
            kwargs = text.dict()
        elif isinstance(text, dict):
            kwargs = deepcopy(text)
        elif text is None:
            kwargs = {'string': ConstantStringEncoding(constant='')}
        else:
            kwargs = {'string': text}
        kwargs['features'] = features
        return cls(**kwargs)

    def _update_from_layer(
        self,
        *,
        text: Union['TextManager', dict, str, None],
        features: Any,
    ):
        """Updates this in-place from a layer.

        This will effectively overwrite all existing state, but in-place
        so that there is no need for any external components to reconnect
        to any useful events. For this reason, only fields that change in
        value will emit their corresponding events.

        Parameters
        ----------
        See :meth:`TextManager._from_layer`.
        """
        # Create a new instance from the input to populate all fields.
        new_manager = TextManager._from_layer(text=text, features=features)

        # Update a copy of this so that any associated errors are raised
        # before actually making the update. This does not need to be a
        # deep copy because update will only try to reassign fields and
        # should not mutate any existing fields in-place.
        # Avoid recursion because some fields are also models that may
        # not share field names/types (e.g. string).
        current_manager = self.copy()
        current_manager.update(new_manager, recurse=False)

        # If we got here, then there were no errors, so update for real.
        # Connected callbacks may raise errors, but those are bugs.
        self.update(new_manager, recurse=False)

        # Some of the encodings may have changed, so ensure they encode new
        # values if needed.
        self.apply(features)

    @validator('blending', pre=True, always=True, allow_reuse=True)
    def _check_blending_mode(cls, blending):
        blending_mode = Blending(blending)

        # The opaque blending mode is not allowed for text.
        # See: https://github.com/napari/napari/pull/600#issuecomment-554142225
        if blending_mode == Blending.OPAQUE:
            blending_mode = Blending.TRANSLUCENT
            warnings.warn(
                trans._(
                    'opaque blending mode is not allowed for text. setting to translucent.',
                    deferred=True,
                ),
                category=RuntimeWarning,
            )

        return blending_mode


def _warn_about_deprecated_text_parameter():
    warnings.warn(
        trans._(
            'text is a deprecated parameter since 0.4.16. Use string instead.'
        ),
        DeprecationWarning,
        stacklevel=2,
    )


def _warn_about_deprecated_properties_parameter():
    warnings.warn(
        trans._(
            'properties is a deprecated parameter since 0.4.16. Use features instead.'
        ),
        DeprecationWarning,
        stacklevel=2,
    )


def _warn_about_deprecated_n_text_parameter():
    warnings.warn(
        trans._(
            'n_text is a deprecated parameter since 0.4.16. Use features instead.'
        ),
        DeprecationWarning,
        stacklevel=2,
    )


def _warn_about_deprecated_values_parameter():
    warnings.warn(
        trans._(
            'values is a deprecated parameter since 0.4.16. Use string instead.'
        ),
        DeprecationWarning,
        stacklevel=2,
    )
