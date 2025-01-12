from collections.abc import Callable, Generator
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
)

import numpy as np

from napari._pydantic_compat import errors, types

if TYPE_CHECKING:
    from decimal import Decimal

    from napari._pydantic_compat import ModelField

    Number = Union[int, float, Decimal]

# In numpy 2, the semantics of the copy argument in np.array changed
# so that copy=False errors if a copy is needed:
# https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
#
# In numpy 1, copy=False meant that a copy was avoided unless necessary,
# but would not error.
#
# In most usage like this use np.asarray instead, but sometimes we need
# to use some of the unique arguments of np.array (e.g. ndmin).
#
# This solution assumes numpy 1 by default, and switches to the numpy 2
# value for any release of numpy 2 on PyPI (including betas and RCs).
copy_if_needed: bool | None = False
if np.lib.NumpyVersion(np.__version__) >= '2.0.0b1':
    copy_if_needed = None


class Array(np.ndarray):
    def __class_getitem__(cls, t):
        return type('Array', (Array,), {'__dtype__': t})

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        dtype = getattr(cls, '__dtype__', None)
        if isinstance(dtype, tuple):
            dtype, shape = dtype
        else:
            shape = ()

        result = np.array(
            val, dtype=dtype, copy=copy_if_needed, ndmin=len(shape)
        )

        if any(
            (shape[i] != -1 and shape[i] != result.shape[i])
            for i in range(len(shape))
        ):
            result = result.reshape(shape)
        return result


class NumberNotEqError(errors.PydanticValueError):
    code = 'number.not_eq'
    msg_template = 'ensure this value is not equal to {prohibited}'

    def __init__(self, *, prohibited: 'Number') -> None:
        super().__init__(prohibited=prohibited)


class ConstrainedInt(types.ConstrainedInt):
    """ConstrainedInt extension that adds not-equal"""

    ne: int | list[int] | None = None

    @classmethod
    def __modify_schema__(cls, field_schema: dict[str, Any]) -> None:
        super().__modify_schema__(field_schema)
        if cls.ne is not None:
            f = 'const' if isinstance(cls.ne, int) else 'enum'
            field_schema['not'] = {f: cls.ne}

    @classmethod
    def __get_validators__(cls) -> Generator[Callable[..., Any], None, None]:
        yield from super().__get_validators__()
        yield cls.validate_ne

    @staticmethod
    def validate_ne(v: 'Number', field: 'ModelField') -> 'Number':
        field_type: ConstrainedInt = field.type_
        _ne = field_type.ne
        if _ne is not None and v in (_ne if isinstance(_ne, list) else [_ne]):
            raise NumberNotEqError(prohibited=field_type.ne)
        return v


def conint(
    *,
    strict: bool = False,
    gt: int | None = None,
    ge: int | None = None,
    lt: int | None = None,
    le: int | None = None,
    multiple_of: int | None = None,
    ne: int | None = None,
) -> type[int]:
    """Extended version of `pydantic.types.conint` that includes not-equal."""
    # use kwargs then define conf in a dict to aid with IDE type hinting
    namespace = {
        'strict': strict,
        'gt': gt,
        'ge': ge,
        'lt': lt,
        'le': le,
        'multiple_of': multiple_of,
        'ne': ne,
    }
    return type('ConstrainedIntValue', (ConstrainedInt,), namespace)
