import inspect
import operator
from collections.abc import Sequence
from enum import auto
from typing import ClassVar, Protocol, runtime_checkable
from unittest.mock import Mock

import dask.array as da
import numpy as np
import pytest
from dask import delayed
from dask.delayed import Delayed

from napari._pydantic_compat import Field, ValidationError
from napari.utils.events import EmitterGroup, EventedModel
from napari.utils.events.custom_types import Array
from napari.utils.misc import StringEnum


def test_creating_empty_evented_model():
    """Test creating an empty evented pydantic model."""
    model = EventedModel()
    assert model is not None
    assert model.events is not None


def test_evented_model():
    """Test creating an evented pydantic model."""

    class User(EventedModel):
        """Demo evented model.

        Parameters
        ----------
        id : int
            User id.
        name : str, optional
            User name.
        """

        id: int
        name: str = 'Alex'
        age: ClassVar[int] = 100

    user = User(id=0)
    # test basic functionality
    assert user.id == 0
    assert user.name == 'Alex'

    user.id = 2
    assert user.id == 2

    # test event system
    assert isinstance(user.events, EmitterGroup)
    assert 'id' in user.events
    assert 'name' in user.events

    # ClassVars are excluded from events
    assert 'age' not in user.events
    # mocking EventEmitters to spy on events
    user.events.id = Mock(user.events.id)
    user.events.name = Mock(user.events.name)
    # setting an attribute should, by default, emit an event with the value
    user.id = 4
    user.events.id.assert_called_with(value=4)
    user.events.name.assert_not_called()
    # and event should only be emitted when the value has changed.
    user.events.id.reset_mock()
    user.id = 4
    user.events.id.assert_not_called()
    user.events.name.assert_not_called()


def test_evented_model_with_array():
    """Test creating an evented pydantic model with an array."""

    def make_array():
        return np.array([[4, 3]])

    class Model(EventedModel):
        """Demo evented model."""

        int_values: Array[int]
        any_values: Array
        shaped1_values: Array[float, (-1,)]
        shaped2_values: Array[int, (1, 2)] = Field(default_factory=make_array)
        shaped3_values: Array[float, (4, -1)]
        shaped4_values: Array[float, (-1, 4)]

    model = Model(
        int_values=[1, 2.2, 3],
        any_values=[1, 2.2],
        shaped1_values=np.array([1.1, 2.0]),
        shaped3_values=np.array([1.1, 2.0, 2.0, 3.0]),
        shaped4_values=np.array([1.1, 2.0, 2.0, 3.0]),
    )
    # test basic functionality
    np.testing.assert_almost_equal(model.int_values, np.array([1, 2, 3]))
    np.testing.assert_almost_equal(model.any_values, np.array([1, 2.2]))
    np.testing.assert_almost_equal(model.shaped1_values, np.array([1.1, 2.0]))
    np.testing.assert_almost_equal(model.shaped2_values, np.array([[4, 3]]))
    np.testing.assert_almost_equal(
        model.shaped3_values, np.array([[1.1, 2.0, 2.0, 3.0]]).T
    )
    np.testing.assert_almost_equal(
        model.shaped4_values, np.array([[1.1, 2.0, 2.0, 3.0]])
    )

    # try changing shape to something impossible to correctly reshape
    with pytest.raises(ValidationError, match='cannot reshape'):
        model.shaped2_values = [1]


def test_evented_model_array_updates():
    """Test updating an evented pydantic model with an array."""

    class Model(EventedModel):
        """Demo evented model."""

        values: Array[int]

    model = Model(values=[1, 2, 3])

    # Mock events
    model.events.values = Mock(model.events.values)

    np.testing.assert_almost_equal(model.values, np.array([1, 2, 3]))

    # Updating with new data
    model.values = [1, 2, 4]
    assert model.events.values.call_count == 1
    np.testing.assert_almost_equal(
        model.events.values.call_args[1]['value'], np.array([1, 2, 4])
    )
    model.events.values.reset_mock()

    # Updating with same data, no event should be emitted
    model.values = [1, 2, 4]
    model.events.values.assert_not_called()


def test_evented_model_array_equality():
    """Test checking equality with an evented model with custom array."""

    class Model(EventedModel):
        """Demo evented model."""

        values: Array[int]

    model1 = Model(values=[1, 2, 3])
    model2 = Model(values=[1, 5, 6])

    assert model1 == model1
    assert model1 != model2

    model2.values = [1, 2, 3]
    assert model1 == model2


def test_evented_model_np_array_equality():
    """Test checking equality with an evented model with direct numpy."""

    class Model(EventedModel):
        values: np.ndarray

    model1 = Model(values=np.array([1, 2, 3]))
    model2 = Model(values=np.array([1, 5, 6]))

    assert model1 == model1
    assert model1 != model2

    model2.values = np.array([1, 2, 3])
    assert model1 == model2


def test_evented_model_da_array_equality():
    """Test checking equality with an evented model with direct dask."""

    class Model(EventedModel):
        values: da.Array

    r = da.ones((64, 64))
    model1 = Model(values=r)
    model2 = Model(values=da.ones((64, 64)))

    assert model1 == model1
    # dask arrays will only evaluate as equal if they are the same object.
    assert model1 != model2

    model2.values = r
    assert model1 == model2


def test_values_updated():
    class User(EventedModel):
        """Demo evented model.

        Parameters
        ----------
        id : int
            User id.
        name : str, optional
            User name.
        """

        id: int
        name: str = 'A'
        age: ClassVar[int] = 100

    user1 = User(id=0)
    user2 = User(id=1, name='K')

    # Add mocks
    user1_events = Mock(user1.events)
    user1.events.connect(user1_events)
    user1.events.id = Mock(user1.events.id)
    user2.events.id = Mock(user2.events.id)

    # Check user1 and user2 dicts
    assert user1.dict() == {'id': 0, 'name': 'A'}
    assert user2.dict() == {'id': 1, 'name': 'K'}

    # Update user1 from user2
    user1.update(user2)
    assert user1.dict() == {'id': 1, 'name': 'K'}

    user1.events.id.assert_called_with(value=1)
    user2.events.id.assert_not_called()
    assert user1_events.call_count == 1
    user1.events.id.reset_mock()
    user2.events.id.reset_mock()
    user1_events.reset_mock()

    # Update user1 from user2 again, no event emission expected
    user1.update(user2)
    assert user1.dict() == {'id': 1, 'name': 'K'}

    user1.events.id.assert_not_called()
    user2.events.id.assert_not_called()
    assert user1_events.call_count == 0


def test_update_with_inner_model_union():
    class Inner(EventedModel):
        w: str

    class AltInner(EventedModel):
        x: str

    class Outer(EventedModel):
        y: int
        z: Inner | AltInner

    original = Outer(y=1, z=Inner(w='a'))
    updated = Outer(y=2, z=AltInner(x='b'))

    original.update(updated, recurse=False)

    assert original == updated


def test_update_with_inner_model_protocol():
    @runtime_checkable
    class InnerProtocol(Protocol):
        def string(self) -> str: ...

        # Protocol fields are not successfully set without explicit validation.
        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, v):
            return v

    class Inner(EventedModel):
        w: str

        def string(self) -> str:
            return self.w

    class AltInner(EventedModel):
        x: str

        def string(self) -> str:
            return self.x

    class Outer(EventedModel):
        y: int
        z: InnerProtocol

    original = Outer(y=1, z=Inner(w='a'))
    updated = Outer(y=2, z=AltInner(x='b'))

    original.update(updated, recurse=False)

    assert original == updated


def test_evented_model_signature():
    class T(EventedModel):
        x: int
        y: str = 'yyy'
        z = b'zzz'

    assert isinstance(T.__signature__, inspect.Signature)
    sig = inspect.signature(T)
    assert str(sig) == "(*, x: int, y: str = 'yyy', z: bytes = b'zzz') -> None"


class MyObj:
    def __init__(self, a: int, b: str) -> None:
        self.a = a
        self.b = b

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        # turn a generic dict into object
        if isinstance(val, dict):
            a = val.get('a')
            b = val.get('b')
        elif isinstance(val, MyObj):
            return val
        # perform additional validation here
        return cls(a, b)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def _json_encode(self):
        return self.__dict__


def test_evented_model_serialization():
    class Model(EventedModel):
        """Demo evented model."""

        obj: MyObj
        shaped: Array[float, (-1,)]

    m = Model(obj=MyObj(1, 'hi'), shaped=[1, 2, 3])
    raw = m.json()
    assert raw == '{"obj": {"a": 1, "b": "hi"}, "shaped": [1.0, 2.0, 3.0]}'
    deserialized = Model.parse_raw(raw)
    assert deserialized == m


def test_nested_evented_model_serialization():
    """Test that encoders on nested sub-models can be used by top model."""

    class NestedModel(EventedModel):
        obj: MyObj

    class Model(EventedModel):
        nest: NestedModel

    m = Model(nest={'obj': {'a': 1, 'b': 'hi'}})
    raw = m.json()
    assert raw == r'{"nest": {"obj": {"a": 1, "b": "hi"}}}'
    deserialized = Model.parse_raw(raw)
    assert deserialized == m


def test_evented_model_dask_delayed():
    """Test that evented models work with dask delayed objects"""

    class MyObject(EventedModel):
        attribute: Delayed

    @delayed
    def my_function():
        pass

    o1 = MyObject(attribute=my_function)

    # check that equality checking works as expected
    assert o1 == o1


# The following tests ensure that StringEnum field values can be
# compared against the enum constants and not their string value.
# For more context see the GitHub issue:
# https://github.com/napari/napari/issues/3062
class SomeStringEnum(StringEnum):
    NONE = auto()
    SOME_VALUE = auto()
    ANOTHER_VALUE = auto()


class ModelWithStringEnum(EventedModel):
    enum_field: SomeStringEnum = SomeStringEnum.NONE


def test_evented_model_with_string_enum_default():
    model = ModelWithStringEnum()
    assert model.enum_field == SomeStringEnum.NONE


def test_evented_model_with_string_enum_parameter():
    model = ModelWithStringEnum(enum_field=SomeStringEnum.SOME_VALUE)
    assert model.enum_field == SomeStringEnum.SOME_VALUE


def test_evented_model_with_string_enum_parameter_as_str():
    model = ModelWithStringEnum(enum_field='some_value')
    assert model.enum_field == SomeStringEnum.SOME_VALUE


def test_evented_model_with_string_enum_setter():
    model = ModelWithStringEnum()
    model.enum_field = SomeStringEnum.SOME_VALUE
    assert model.enum_field == SomeStringEnum.SOME_VALUE


def test_evented_model_with_string_enum_setter_as_str():
    model = ModelWithStringEnum()
    model.enum_field = 'some_value'
    assert model.enum_field == SomeStringEnum.SOME_VALUE


def test_evented_model_with_string_enum_parse_raw():
    model = ModelWithStringEnum(enum_field=SomeStringEnum.SOME_VALUE)
    deserialized_model = ModelWithStringEnum.parse_raw(model.json())
    assert deserialized_model.enum_field == model.enum_field


def test_evented_model_with_string_enum_parse_obj():
    model = ModelWithStringEnum(enum_field=SomeStringEnum.SOME_VALUE)
    deserialized_model = ModelWithStringEnum.parse_obj(model.dict())
    assert deserialized_model.enum_field == model.enum_field


class T(EventedModel):
    a: int = 1
    b: int = 1

    @property
    def c(self) -> list[int]:
        return [self.a, self.b]

    @c.setter
    def c(self, val: Sequence[int]):
        self.a, self.b = val

    @property
    def d(self) -> int:
        return sum(self.c)

    @d.setter
    def d(self, val: int):
        # note that d only uses c, which in turns affects in a and b
        self.c = [val // 2, val // 2]

    @property
    def e(self) -> int:
        # should also work without setter
        return self.a * 10


def test_evented_model_with_property_setters():
    t = T()

    assert list(T.__properties__) == ['c', 'd', 'e']
    # the metaclass should have figured out that both a and b affect c
    assert T.__field_dependents__ == {'a': {'c', 'd', 'e'}, 'b': {'c', 'd'}}

    # all the fields and properties behave as expected
    assert t.c == [1, 1]
    t.a = 4
    assert t.c == [4, 1]
    t.c = [2, 3]
    assert t.c == [2, 3]
    assert t.a == 2
    assert t.b == 3
    t.d = 4
    assert t.a == 2
    assert t.b == 2

    with pytest.raises(AttributeError):
        t.e = 100


@pytest.fixture
def mocked_object():
    t = T()
    t.events.a = Mock(t.events.a)
    t.events.b = Mock(t.events.b)
    t.events.c = Mock(t.events.c)
    t.events.d = Mock(t.events.d)
    t.events.e = Mock(t.events.e)
    return t


@pytest.mark.parametrize(
    ('attribute', 'value', 'expected_event_values'),
    [
        ('a', 5, {'a': 5, 'b': None, 'c': [5, 1], 'd': 6, 'e': 50}),
        ('b', 5, {'a': None, 'b': 5, 'c': [1, 5], 'd': 6, 'e': None}),
        ('c', [10, 20], {'a': 10, 'b': 20, 'c': [10, 20], 'd': 30, 'e': 100}),
        ('d', 8, {'a': 4, 'b': 4, 'c': [4, 4], 'd': 8, 'e': 40}),
    ],
)
def test_evented_model_with_property_setter_events(
    mocked_object, attribute, value, expected_event_values
):
    """Test that setting connected fields and properties fires the right events.

    For each field and property, set a new value and check that all the
    dependent fields/properties fire events with the correct value,
    and that non-connected properties fire no event.
    """
    assert attribute in mocked_object.events

    setattr(mocked_object, attribute, value)
    for attr, val in expected_event_values.items():
        emitter = getattr(mocked_object.events, attr)
        if val is None:
            emitter.assert_not_called()
        else:
            emitter.assert_called_with(value=val)


def test_evented_model_with_property_without_setter(mocked_object):
    with pytest.raises(AttributeError):
        # no setter provided for T.e
        mocked_object.e = 2


def test_evented_model_with_provided_dependencies():
    class T(EventedModel):
        a: int = 1

        @property
        def b(self):
            return self.a * 2

        class Config:
            dependencies = {'b': ['a']}

    t = T()
    t.events.a = Mock(t.events.a)
    t.events.b = Mock(t.events.b)

    t.a = 2
    t.events.a.assert_called_with(value=2)
    t.events.b.assert_called_with(value=4)

    # should fail if property does not exist
    with pytest.raises(
        ValueError, match='Fields with dependencies must be properties'
    ):

        class T(EventedModel):
            a: int = 1

            @property
            def b(self):  # pragma: no cover
                return self.a * 2

            class Config:
                dependencies = {'x': ['a']}

    # should warn if field does not exist
    with pytest.warns(match='Unrecognized field dependency'):

        class T(EventedModel):
            a: int = 1

            @property
            def b(self):  # pragma: no cover
                return self.a * 2

            class Config:
                dependencies = {'b': ['x']}


def test_property_get_eq_operator():
    """Test if the __eq_operators__ for properties are properly recognized"""

    class Tt(EventedModel):
        a: int = 1

        @property
        def b(self) -> float:  # pragma: no cover
            return self.a * 2

        @property
        def c(self):  # pragma: no cover
            return self.a * 3

    assert Tt.__eq_operators__ == {'a': operator.eq, 'b': operator.eq}


def test_property_str_annotation():
    """Test if the __str_annotations__ for properties are properly recognized"""

    class Tt(EventedModel):
        a: int = 1

        @property
        def b(self) -> 'np.ndarray':  # pragma: no cover
            return np.ndarray([self.a, self.a])

        @property
        def c(self):  # pragma: no cover
            return self.a * 3

    assert Tt.__eq_operators__ == {'a': operator.eq}


def test_events_are_fired_only_if_necessary(monkeypatch):
    class Tt(EventedModel):
        a: int = 1

        @property
        def b(self) -> float:
            return self.a * 2

        @property
        def c(self):
            return self.a * 3

    eq_op_get = Mock(return_value=operator.eq)
    monkeypatch.setattr(
        'napari.utils.events.evented_model.pick_equality_operator', eq_op_get
    )

    t = Tt()

    a_eq = Mock(return_value=False)
    b_eq = Mock(return_value=False)

    t.__eq_operators__['a'] = a_eq
    t.__eq_operators__['b'] = b_eq

    t.a = 2
    a_eq.assert_not_called()
    b_eq.assert_not_called()

    call1 = Mock()
    t.events.a.connect(call1)

    t.a = 3

    call1.assert_called_once()
    assert call1.call_args.args[0].value == 3
    a_eq.assert_called_once()
    b_eq.assert_not_called()
    eq_op_get.assert_not_called()

    call2 = Mock()
    t.events.b.connect(call2)
    call1.reset_mock()
    a_eq.reset_mock()

    t.a = 4
    call1.assert_called_once()
    call2.assert_called_once()
    assert call1.call_args.args[0].value == 4
    assert call2.call_args.args[0].value == 8
    a_eq.assert_called_once()
    b_eq.assert_called_once()
    eq_op_get.assert_not_called()

    call3 = Mock()
    t.events.c.connect(call3)
    call1.reset_mock()
    call2.reset_mock()
    a_eq.reset_mock()
    b_eq.reset_mock()

    t.a = 3
    call1.assert_called_once()
    call2.assert_called_once()
    call3.assert_called_once()
    assert call1.call_args.args[0].value == 3
    assert call2.call_args.args[0].value == 6
    assert call3.call_args.args[0].value == 9
    a_eq.assert_called_once()
    b_eq.assert_called_once()
    eq_op_get.assert_called_once_with(9)


def _reset_mocks(*args):
    for el in args:
        el.reset_mock()


def test_single_emit():
    class SampleClass(EventedModel):
        a: int = 1
        b: int = 2

        @property
        def c(self):
            return self.a

        @c.setter
        def c(self, value):
            self.a = value

        @property
        def d(self):
            return self.a + self.b

        @d.setter
        def d(self, value):
            self.a = value // 2
            self.b = value - self.a

        @property
        def e(self):
            return self.a - self.b

    s = SampleClass()
    a_m = Mock()
    c_m = Mock()
    d_m = Mock()
    s.events.a.connect(a_m)
    s.events.c.connect(c_m)
    s.events.d.connect(d_m)

    s.a = 4
    a_m.assert_called_once()
    c_m.assert_called_once()
    d_m.assert_called_once()

    _reset_mocks(a_m, c_m, d_m)

    s.c = 6
    a_m.assert_called_once()
    c_m.assert_called_once()
    d_m.assert_called_once()

    _reset_mocks(a_m, c_m, d_m)

    e_m = Mock()
    s.events.e.connect(e_m)

    s.d = 21
    a_m.assert_called_once()
    c_m.assert_called_once()
    d_m.assert_called_once()
    e_m.assert_called_once()
