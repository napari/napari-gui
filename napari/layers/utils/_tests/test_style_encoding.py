"""
These tests cover and help explain the implementations of different
types of generic encodings, like constant, manual, and derived encodings,
rather than the types of values they encode like strings and colors
or the ways those are encoded.

In particular, these cover the stateful part of the StyleEncoding, which
is important to napari at the time of writing, but may be removed in the future.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from napari._pydantic_compat import Field
from napari.layers.utils.style_encoding import (
    _ConstantStyleEncoding,
    _DerivedStyleEncoding,
    _ManualStyleEncoding,
)
from napari.utils.events.custom_types import Array


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'scalar': [1, 2, 3],
            'vector': [[1, 1], [2, 2], [3, 3]],
        }
    )


Scalar = Array[int, ()]
ScalarArray = Array[int, (-1,)]


class ScalarConstantEncoding(_ConstantStyleEncoding[Scalar, ScalarArray]):
    constant: Scalar


def test_scalar_constant_encoding_apply(features):
    encoding = ScalarConstantEncoding(constant=0)
    encoding._apply(features)
    np.testing.assert_array_equal(encoding._values, 0)


def test_scalar_constant_encoding_append():
    encoding = ScalarConstantEncoding(constant=0)
    encoding._append(Vector.validate_type([4, 5]))
    np.testing.assert_array_equal(encoding._values, 0)


def test_scalar_constant_encoding_delete():
    encoding = ScalarConstantEncoding(constant=0)
    encoding._delete([0, 2])
    np.testing.assert_array_equal(encoding._values, 0)


def test_scalar_constant_encoding_clear():
    encoding = ScalarConstantEncoding(constant=0)
    encoding._clear()
    np.testing.assert_array_equal(encoding._values, 0)


class ScalarManualEncoding(_ManualStyleEncoding[Scalar, ScalarArray]):
    array: ScalarArray
    default: Scalar = np.array(-1)


def test_scalar_manual_encoding_apply_with_shorter(features):
    encoding = ScalarManualEncoding(array=[1, 2, 3, 4])
    encoding._apply(features)
    np.testing.assert_array_equal(encoding._values, [1, 2, 3])


def test_scalar_manual_encoding_apply_with_equal_length(features):
    encoding = ScalarManualEncoding(array=[1, 2, 3])
    encoding._apply(features)
    np.testing.assert_array_equal(encoding._values, [1, 2, 3])


def test_scalar_manual_encoding_apply_with_longer(features):
    encoding = ScalarManualEncoding(array=[1, 2], default=-1)
    encoding._apply(features)
    np.testing.assert_array_equal(encoding._values, [1, 2, -1])


def test_scalar_manual_encoding_append():
    encoding = ScalarManualEncoding(array=[1, 2, 3])
    encoding._append(Vector.validate_type([4, 5]))
    np.testing.assert_array_equal(encoding._values, [1, 2, 3, 4, 5])


def test_scalar_manual_encoding_delete():
    encoding = ScalarManualEncoding(array=[1, 2, 3])
    encoding._delete([0, 2])
    np.testing.assert_array_equal(encoding._values, [2])


def test_scalar_manual_encoding_clear():
    encoding = ScalarManualEncoding(array=[1, 2, 3])
    encoding._clear()
    np.testing.assert_array_equal(encoding._values, [1, 2, 3])


class ScalarDirectEncoding(_DerivedStyleEncoding[Scalar, ScalarArray]):
    feature: str
    fallback: Scalar = np.array(-1)

    def __call__(self, features: Any) -> ScalarArray:
        return ScalarArray.validate_type(features[self.feature])


def test_scalar_derived_encoding_apply(features):
    encoding = ScalarDirectEncoding(feature='scalar')

    encoding._apply(features)

    expected_values = features['scalar']
    np.testing.assert_array_equal(encoding._values, expected_values)


def test_scalar_derived_encoding_apply_with_failure(features):
    encoding = ScalarDirectEncoding(feature='not_a_column', fallback=-1)

    with pytest.warns(RuntimeWarning):
        encoding._apply(features)

    np.testing.assert_array_equal(encoding._values, [-1] * len(features))


def test_scalar_derived_encoding_append():
    encoding = ScalarDirectEncoding(feature='scalar')
    encoding._cached = ScalarArray.validate_type([1, 2, 3])

    encoding._append(ScalarArray.validate_type([4, 5]))

    np.testing.assert_array_equal(encoding._values, [1, 2, 3, 4, 5])


def test_scalar_derived_encoding_delete():
    encoding = ScalarDirectEncoding(feature='scalar')
    encoding._cached = ScalarArray.validate_type([1, 2, 3])

    encoding._delete([0, 2])

    np.testing.assert_array_equal(encoding._values, [2])


def test_scalar_derived_encoding_clear():
    encoding = ScalarDirectEncoding(feature='scalar')
    encoding._cached = ScalarArray.validate_type([1, 2, 3])

    encoding._clear()

    np.testing.assert_array_equal(encoding._values, [])


Vector = Array[int, (2,)]
VectorArray = Array[int, (-1, 2)]


class VectorConstantEncoding(_ConstantStyleEncoding[Vector, VectorArray]):
    constant: Vector


def test_vector_constant_encoding_apply(features):
    encoding = VectorConstantEncoding(constant=[0, 0])
    encoding._apply(features)
    np.testing.assert_array_equal(encoding._values, [0, 0])


def test_vector_constant_encoding_append():
    encoding = VectorConstantEncoding(constant=[0, 0])
    encoding._append(Vector.validate_type([4, 5]))
    np.testing.assert_array_equal(encoding._values, [0, 0])


def test_vector_constant_encoding_delete():
    encoding = VectorConstantEncoding(constant=[0, 0])
    encoding._delete([0, 2])
    np.testing.assert_array_equal(encoding._values, [0, 0])


def test_vector_constant_encoding_clear():
    encoding = VectorConstantEncoding(constant=[0, 0])
    encoding._clear()
    np.testing.assert_array_equal(encoding._values, [0, 0])


class VectorManualEncoding(_ManualStyleEncoding[Vector, VectorArray]):
    array: VectorArray
    default: Vector = Field(default_factory=lambda: np.array([-1, -1]))


def test_vector_manual_encoding_apply_with_shorter(features):
    encoding = VectorManualEncoding(array=[[1, 1], [2, 2], [3, 3], [4, 4]])
    encoding._apply(features)
    np.testing.assert_array_equal(encoding._values, [[1, 1], [2, 2], [3, 3]])


def test_vector_manual_encoding_apply_with_equal_length(features):
    encoding = VectorManualEncoding(array=[[1, 1], [2, 2], [3, 3]])
    encoding._apply(features)
    np.testing.assert_array_equal(encoding._values, [[1, 1], [2, 2], [3, 3]])


def test_vector_manual_encoding_apply_with_longer(features):
    encoding = VectorManualEncoding(array=[[1, 1], [2, 2]], default=[-1, -1])
    encoding._apply(features)
    np.testing.assert_array_equal(encoding._values, [[1, 1], [2, 2], [-1, -1]])


def test_vector_manual_encoding_append():
    encoding = VectorManualEncoding(array=[[1, 1], [2, 2], [3, 3]])
    encoding._append(Vector.validate_type([[4, 4], [5, 5]]))
    np.testing.assert_array_equal(
        encoding._values, [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    )


def test_vector_manual_encoding_delete():
    encoding = VectorManualEncoding(array=[[1, 1], [2, 2], [3, 3]])
    encoding._delete([0, 2])
    np.testing.assert_array_equal(encoding._values, [[2, 2]])


def test_vector_manual_encoding_clear():
    encoding = VectorManualEncoding(array=[[1, 1], [2, 2], [3, 3]])
    encoding._clear()
    np.testing.assert_array_equal(encoding._values, [[1, 1], [2, 2], [3, 3]])


class VectorDirectEncoding(_DerivedStyleEncoding[Vector, VectorArray]):
    feature: str
    fallback: Vector = Field(default_factory=lambda: np.array([-1, -1]))

    def __call__(self, features: Any) -> Vector | VectorArray:
        return VectorArray.validate_type(list(features[self.feature]))


def test_vector_derived_encoding_apply(features):
    encoding = VectorDirectEncoding(feature='vector')

    encoding._apply(features)

    expected_values = list(features['vector'])
    np.testing.assert_array_equal(encoding._values, expected_values)


def test_vector_derived_encoding_apply_with_failure(features):
    encoding = VectorDirectEncoding(feature='not_a_column', fallback=[-1, -1])

    with pytest.warns(RuntimeWarning):
        encoding._apply(features)

    np.testing.assert_array_equal(encoding._values, [[-1, -1]] * len(features))


def test_vector_derived_encoding_append():
    encoding = VectorDirectEncoding(feature='vector')
    encoding._cached = VectorArray.validate_type([[1, 1], [2, 2], [3, 3]])

    encoding._append(VectorArray.validate_type([[4, 4], [5, 5]]))

    np.testing.assert_array_equal(
        encoding._values, [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    )


def test_vector_derived_encoding_delete():
    encoding = VectorDirectEncoding(feature='vector')
    encoding._cached = VectorArray.validate_type([[1, 1], [2, 2], [3, 3]])

    encoding._delete([0, 2])

    np.testing.assert_array_equal(encoding._values, [[2, 2]])


def test_vector_derived_encoding_clear():
    encoding = VectorDirectEncoding(feature='vector')
    encoding._cached = VectorArray.validate_type([[1, 1], [2, 2], [3, 3]])

    encoding._clear()

    np.testing.assert_array_equal(encoding._values, np.empty((0, 2)))
