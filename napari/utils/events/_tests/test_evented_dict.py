from unittest.mock import Mock

import pytest

from napari.utils.events import EmitterGroup
from napari.utils.events.containers import EventedDict


@pytest.fixture
def regular_dict():
    return {'A': 0, 'B': 1, 'C': 2, False: '3', 4: 5}


@pytest.fixture(params=[EventedDict])
def test_dict(request, regular_dict):
    test_dict = request.param(regular_dict)
    test_dict.events = Mock(wraps=test_dict.events)
    return test_dict


@pytest.mark.parametrize(
    'meth',
    [
        # METHOD, ARGS, EXPECTED EVENTS
        # primary interface
        ('__getitem__', ('A',), ()),  # read
        ('__setitem__', ('A', 3), ('changed',)),  # update
        ('__setitem__', ('D', 3), ('adding', 'added')),  # add new entry
        ('__delitem__', ('A',), ('removing', 'removed')),  # delete
        # inherited interface
        ('key', (3,), ()),
        ('clear', (), ('removing', 'removed') * 3),
        ('pop', ('B',), ('removing', 'removed')),
    ],
    ids=lambda x: x[0],
)
def test_dict_interface_parity(test_dict, regular_dict, meth):
    method_name, args, expected = meth
    test_dict_method = getattr(test_dict, method_name)
    assert test_dict == regular_dict
    if hasattr(regular_dict, method_name):
        regular_dict_method = getattr(regular_dict, method_name)
        assert test_dict_method(*args) == regular_dict_method(*args)
        assert test_dict == regular_dict
    else:
        test_dict_method(*args)  # smoke test

    for c, expect in zip(
        test_dict.events.call_args_list, expected, strict=False
    ):
        event = c.args[0]
        assert event.type == expect


def test_copy(test_dict, regular_dict):
    """Copying an evented dict should return a same-class evented dict."""
    new_test = test_dict.copy()
    assert (
        len({type(k) for k in new_test}) >= 2
    ), 'We want at least non-string keys to test edge cases'

    new_reg = regular_dict.copy()
    assert id(new_test) != id(test_dict)
    assert new_test == test_dict
    assert tuple(new_test) == tuple(test_dict) == tuple(new_reg)
    test_dict.events.assert_not_called()


class E:
    def __init__(self) -> None:
        self.events = EmitterGroup(test=None)


def test_child_events():
    """Test that evented dicts bubble child events."""
    # create a random object that emits events
    e_obj = E()
    root = EventedDict()
    observed = []
    root.events.connect(lambda e: observed.append(e))
    root['A'] = e_obj
    e_obj.events.test(value='hi')
    obs = [(e.type, e.key, getattr(e, 'value', None)) for e in observed]
    expected = [
        ('adding', 'A', None),  # before we adding b into root
        ('added', 'A', e_obj),  # after b was added into root
        ('test', 'A', 'hi'),  # when e_obj emitted an event called "test"
    ]
    for o, e in zip(obs, expected, strict=False):
        assert o == e


def test_evented_dict_subclass():
    """Test that multiple inheritance maintains events from superclass."""

    class A:
        events = EmitterGroup(boom=None)

    class B(A, EventedDict):
        pass

    dct = B({'A': 1, 'B': 2})
    assert hasattr(dct, 'events')
    assert 'boom' in dct.events.emitters
    assert dct == {'A': 1, 'B': 2}
