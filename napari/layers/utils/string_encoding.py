from collections.abc import Sequence
from string import Formatter
from typing import Any, Literal, Protocol, Union, runtime_checkable

import numpy as np

from napari._pydantic_compat import parse_obj_as
from napari.layers.utils.style_encoding import (
    StyleEncoding,
    _ConstantStyleEncoding,
    _DerivedStyleEncoding,
    _ManualStyleEncoding,
)
from napari.utils.events.custom_types import Array
from napari.utils.translations import trans

"""A scalar array that represents one string value."""
StringValue = Array[str, ()]

"""An Nx1 array where each element represents one string value."""
StringArray = Array[str, (-1,)]


"""The default string value, which may also be used a safe fallback string."""
DEFAULT_STRING = np.array('', dtype='<U1')


@runtime_checkable
class StringEncoding(StyleEncoding[StringValue, StringArray], Protocol):
    """Encodes strings from layer features."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(
        cls, value: Union['StringEncoding', dict, str, Sequence[str]]
    ) -> 'StringEncoding':
        """Validates and coerces a value to a StringEncoding.

        Parameters
        ----------
        value : StringEncodingArgument
            The value to validate and coerce.
            If this is already a StringEncoding, it is returned as is.
            If this is a dict, then it should represent one of the built-in string encodings.
            If this a valid format string, then a FormatStringEncoding is returned.
            If this is any other string, a DirectStringEncoding is returned.
            If this is a sequence of strings, a ManualStringEncoding is returned.

        Returns
        -------
        StringEncoding

        Raises
        ------
        TypeError
            If the value is not a supported type.
        ValidationError
            If the value cannot be parsed into a StringEncoding.
        """
        if isinstance(value, StringEncoding):
            return value
        if isinstance(value, dict):
            return parse_obj_as(
                Union[
                    ConstantStringEncoding,
                    ManualStringEncoding,
                    DirectStringEncoding,
                    FormatStringEncoding,
                ],
                value,
            )
        if isinstance(value, str):
            if _is_format_string(value):
                return FormatStringEncoding(format=value)
            return DirectStringEncoding(feature=value)
        if isinstance(value, Sequence):
            return ManualStringEncoding(array=value, default=DEFAULT_STRING)
        raise TypeError(
            trans._(
                'value should be a StringEncoding, a dict, a string, a sequence of strings, or None',
                deferred=True,
            )
        )


class ConstantStringEncoding(_ConstantStyleEncoding[StringValue, StringArray]):
    """Encodes color values from a single constant color.

    Attributes
    ----------
    constant : StringValue
        The constant string value.
    encoding_type : Literal['ConstantStringEncoding']
        The type of encoding this specifies, which is useful for distinguishing
        this from other encodings when passing this as a dictionary.
    """

    constant: StringValue
    encoding_type: Literal['ConstantStringEncoding'] = 'ConstantStringEncoding'


class ManualStringEncoding(_ManualStyleEncoding[StringValue, StringArray]):
    """Encodes string values manually in an array.

    Attributes
    ----------
    array : StringArray
        The array of string values.
    default : StringValue
        The default string value that is used when requesting a value that
        is out of bounds in the array attribute.
    encoding_type : Literal['ManualStringEncoding']
        The type of encoding this specifies, which is useful for distinguishing
        this from other encodings when passing this as a dictionary.
    """

    array: StringArray
    default: StringValue = DEFAULT_STRING
    encoding_type: Literal['ManualStringEncoding'] = 'ManualStringEncoding'


class DirectStringEncoding(_DerivedStyleEncoding[StringValue, StringArray]):
    """Encodes strings directly from a feature column.

    Attributes
    ----------
    feature : str
        The name of the feature that contains the desired strings.
    fallback : StringValue
        The safe constant fallback string to use if the feature column
        does not contain valid string values.
    encoding_type : Literal['DirectStringEncoding']
        The type of encoding this specifies, which is useful for distinguishing
        this from other encodings when passing this as a dictionary.
    """

    feature: str
    fallback: StringValue = DEFAULT_STRING
    encoding_type: Literal['DirectStringEncoding'] = 'DirectStringEncoding'

    def __call__(self, features: Any) -> StringArray:
        return np.array(features[self.feature], dtype=str)


class FormatStringEncoding(_DerivedStyleEncoding[StringValue, StringArray]):
    """Encodes string values by formatting feature values.

    Attributes
    ----------
    format : str
        A format string with the syntax supported by :func:`str.format`,
        where all format fields should be feature names.
    fallback : StringValue
        The safe constant fallback string to use if the format string
        is not valid or contains fields other than feature names.
    encoding_type : Literal['FormatStringEncoding']
        The type of encoding this specifies, which is useful for distinguishing
        this from other encodings when passing this as a dictionary.
    """

    format: str
    fallback: StringValue = DEFAULT_STRING
    encoding_type: Literal['FormatStringEncoding'] = 'FormatStringEncoding'

    def __call__(self, features: Any) -> StringArray:
        feature_names = features.columns.to_list()
        # Expose the dataframe index to the format string keys
        # unless a column exists with the name "index", which takes precedence.
        with_index = False
        if 'index' not in feature_names:
            feature_names = ['index'] + feature_names
            with_index = True
        values = [
            self.format.format(**dict(zip(feature_names, row, strict=False)))
            for row in features.itertuples(index=with_index, name=None)
        ]
        return np.array(values, dtype=str)


def _is_format_string(string: str) -> bool:
    """Returns True if a string is a valid format string with at least one field, False otherwise."""
    try:
        fields = tuple(
            field
            for _, field, _, _ in Formatter().parse(string)
            if field is not None
        )
    except ValueError:
        return False
    return len(fields) > 0
