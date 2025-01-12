from collections.abc import MutableSequence
from unittest.mock import Mock, call

import numpy as np
import pytest

from napari.utils.events import EmitterGroup, EventedList, NestableEventedList


@pytest.fixture
def regular_list():
    return list(range(5))


@pytest.fixture(params=[EventedList, NestableEventedList])
def test_list(request, regular_list):
    test_list = request.param(regular_list)
    test_list.events = Mock(wraps=test_list.events)
    return test_list


@pytest.mark.parametrize(
    'meth',
    [
        # METHOD, ARGS, EXPECTED EVENTS
        # primary interface
        ('insert', (2, 10), ('inserting', 'inserted')),  # create
        ('__getitem__', (2,), ()),  # read
        ('__setitem__', (2, 3), ('changed',)),  # update
        ('__setitem__', (slice(2), [1, 2]), ('changed',)),  # update slice
        ('__setitem__', (slice(2, 2), [1, 2]), ('changed',)),  # update slice
        ('__delitem__', (2,), ('removing', 'removed')),  # delete
        (
            '__delitem__',
            (slice(2),),
            ('removing', 'removed') * 2,
        ),
        ('__delitem__', (slice(0, 0),), ('removing', 'removed')),
        (
            '__delitem__',
            (slice(-3),),
            ('removing', 'removed') * 2,
        ),
        (
            '__delitem__',
            (slice(-2, None),),
            ('removing', 'removed') * 2,
        ),
        # inherited interface
        ('append', (3,), ('inserting', 'inserted')),
        ('clear', (), ('removing', 'removed') * 5),
        ('count', (3,), ()),
        ('extend', ([7, 8, 9],), ('inserting', 'inserted') * 3),
        ('index', (3,), ()),
        ('pop', (-2,), ('removing', 'removed')),
        ('remove', (3,), ('removing', 'removed')),
        ('reverse', (), ('reordered',)),
        ('__add__', ([7, 8, 9],), ()),
        ('__iadd__', ([7, 9],), ('inserting', 'inserted') * 2),
        ('__radd__', ([7, 9],), ('inserting', 'inserted') * 2),
        # sort?
    ],
    ids=lambda x: x[0],
)
def test_list_interface_parity(test_list, regular_list, meth):
    method_name, args, expected = meth
    test_list_method = getattr(test_list, method_name)
    assert tuple(test_list) == tuple(regular_list)
    if hasattr(regular_list, method_name):
        regular_list_method = getattr(regular_list, method_name)
        assert test_list_method(*args) == regular_list_method(*args)
        assert tuple(test_list) == tuple(regular_list)
    else:
        test_list_method(*args)  # smoke test

    for c, expect in zip(
        test_list.events.call_args_list, expected, strict=False
    ):
        event = c.args[0]
        assert event.type == expect


def test_hash(test_list):
    assert id(test_list) == hash(test_list)


def test_list_interface_exceptions(test_list):
    bad_index = {'a': 'dict'}
    with pytest.raises(TypeError):
        test_list[bad_index]

    with pytest.raises(TypeError):
        test_list[bad_index] = 1

    with pytest.raises(TypeError):
        del test_list[bad_index]

    with pytest.raises(TypeError):
        test_list.insert([bad_index], 0)


def test_copy(test_list, regular_list):
    """Copying an evented list should return a same-class evented list."""
    new_test = test_list.copy()
    new_reg = regular_list.copy()
    assert id(new_test) != id(test_list)
    assert new_test == test_list
    assert tuple(new_test) == tuple(test_list) == tuple(new_reg)
    test_list.events.assert_not_called()


def test_move(test_list):
    """Test the that we can move objects with the move method"""
    test_list.events = Mock(wraps=test_list.events)

    def _fail():
        raise AssertionError('unexpected event called')

    test_list.events.removing.connect(_fail)
    test_list.events.removed.connect(_fail)
    test_list.events.inserting.connect(_fail)
    test_list.events.inserted.connect(_fail)

    before = list(test_list)
    assert before == [0, 1, 2, 3, 4]  # from fixture
    # pop the object at 0 and insert at current position 3
    test_list.move(0, 3)
    expectation = [1, 2, 0, 3, 4]
    assert test_list != before
    assert test_list == expectation
    test_list.events.moving.assert_called_once()
    test_list.events.moved.assert_called_once()
    test_list.events.reordered.assert_called_with(value=expectation)

    # move the other way
    # pop the object at 3 and insert at current position 0
    assert test_list == [1, 2, 0, 3, 4]
    test_list.move(3, 0)
    assert test_list == [3, 1, 2, 0, 4]

    # negative index destination
    test_list.move(1, -2)
    assert test_list == [3, 2, 0, 1, 4]


BASIC_INDICES = [
    ((2,), 0, [2, 0, 1, 3, 4, 5, 6, 7]),  # move single item
    ([0, 2, 3], 6, [1, 4, 5, 0, 2, 3, 6, 7]),  # move back
    ([4, 7], 1, [0, 4, 7, 1, 2, 3, 5, 6]),  # move forward
    ([0, 5, 6], 3, [1, 2, 0, 5, 6, 3, 4, 7]),  # move in between
    ([1, 3, 5, 7], 3, [0, 2, 1, 3, 5, 7, 4, 6]),  # same as above
    ([0, 2, 3, 2, 3], 6, [1, 4, 5, 0, 2, 3, 6, 7]),  # strip dupe indices
]
OTHER_INDICES = [
    ([7, 4], 1, [0, 7, 4, 1, 2, 3, 5, 6]),  # move forward reorder
    ([3, 0, 2], 6, [1, 4, 5, 3, 0, 2, 6, 7]),  # move back reorder
    ((2, 4), -2, [0, 1, 3, 5, 6, 2, 4, 7]),  # negative indexing
    ([slice(None, 3)], 6, [3, 4, 5, 0, 1, 2, 6, 7]),  # move slice back
    ([slice(5, 8)], 2, [0, 1, 5, 6, 7, 2, 3, 4]),  # move slice forward
    ([slice(1, 8, 2)], 3, [0, 2, 1, 3, 5, 7, 4, 6]),  # move slice between
    ([slice(None, 8, 3)], 4, [1, 2, 0, 3, 6, 4, 5, 7]),
    ([slice(None, 8, 3), 0, 3, 6], 4, [1, 2, 0, 3, 6, 4, 5, 7]),
]
MOVING_INDICES = BASIC_INDICES + OTHER_INDICES


@pytest.mark.parametrize(('sources', 'dest', 'expectation'), MOVING_INDICES)
def test_move_multiple(sources, dest, expectation):
    """Test the that we can move objects with the move method"""
    el = EventedList(range(8))
    el.events = Mock(wraps=el.events)
    assert el == [0, 1, 2, 3, 4, 5, 6, 7]

    def _fail():
        raise AssertionError('unexpected event called')

    el.events.removing.connect(_fail)
    el.events.removed.connect(_fail)
    el.events.inserting.connect(_fail)
    el.events.inserted.connect(_fail)

    el.move_multiple(sources, dest)
    assert el == expectation
    el.events.moving.assert_called()
    el.events.moved.assert_called()
    el.events.reordered.assert_called_with(value=expectation)


def test_move_multiple_mimics_slice_reorder():
    """Test the that move_multiple provides the same result as slice insertion."""
    data = list(range(8))
    el = EventedList(data)
    el.events = Mock(wraps=el.events)
    assert el == data
    new_order = [1, 5, 3, 4, 6, 7, 2, 0]
    # this syntax
    el.move_multiple(new_order, 0)
    # is the same as this syntax
    data[:] = [data[i] for i in new_order]
    assert el == new_order
    assert el == data
    assert el.events.moving.call_args_list == [
        call(index=1, new_index=0),
        call(index=5, new_index=1),
        call(index=4, new_index=2),
        call(index=5, new_index=3),
        call(index=6, new_index=4),
        call(index=7, new_index=5),
        call(index=7, new_index=6),
    ]
    assert el.events.moved.call_args_list == [
        call(index=1, new_index=0, value=1),
        call(index=5, new_index=1, value=5),
        call(index=4, new_index=2, value=3),
        call(index=5, new_index=3, value=4),
        call(index=6, new_index=4, value=6),
        call(index=7, new_index=5, value=7),
        call(index=7, new_index=6, value=2),
    ]
    el.events.reordered.assert_called_with(value=new_order)

    # move_multiple also works omitting the insertion index
    el[:] = list(range(8))
    el.move_multiple(new_order)
    assert el == new_order


def test_slice(test_list, regular_list):
    """Slicing an evented list should return a same-class evented list."""
    test_slice = test_list[1:3]
    regular_slice = regular_list[1:3]
    assert tuple(test_slice) == tuple(regular_slice)
    assert isinstance(test_slice, test_list.__class__)


NEST = [0, [10, [110, [1110, 1111, 1112], 112], 12], 2]


def flatten(container):
    """Flatten arbitrarily nested list.

    Examples
    --------
    >>> a = [1, [2, [3], 4], 5]
    >>> list(flatten(a))
    [1, 2, 3, 4, 5]
    """
    for i in container:
        if isinstance(i, MutableSequence):
            yield from flatten(i)
        else:
            yield i


def test_nested_indexing():
    """test that we can index a nested list with nl[1, 2, 3] syntax."""
    ne_list = NestableEventedList(NEST)
    # 110 -> '110' -> (1, 1, 0)
    indices = [tuple(int(x) for x in str(n)) for n in flatten(NEST)]
    for index in indices:
        assert ne_list[index] == int(''.join(map(str, index)))

    assert ne_list.has_index(1)
    assert ne_list.has_index((1,))
    assert ne_list.has_index((1, 2))
    assert ne_list.has_index((1, 1, 2))
    assert not ne_list.has_index((1, 1, 3))
    assert not ne_list.has_index((1, 1, 2, 3, 4))
    assert not ne_list.has_index(100)


# indices in NEST that are themselves lists
@pytest.mark.parametrize(
    'group_index', [(), (1,), (1, 1), (1, 1, 1)], ids=lambda x: str(x)
)
@pytest.mark.parametrize(
    'meth',
    [
        # METHOD, ARGS, EXPECTED EVENTS
        # primary interface
        ('insert', (0, 10), ('inserting', 'inserted')),
        ('__getitem__', (2,), ()),  # read
        ('__setitem__', (2, 3), ('changed',)),  # update
        ('__delitem__', ((),), ('removing', 'removed')),  # delete
        ('__delitem__', ((1,),), ('removing', 'removed')),  # delete
        ('__delitem__', (2,), ('removing', 'removed')),  # delete
        (
            '__delitem__',
            (slice(2),),
            ('removing', 'removed') * 2,
        ),
        (
            '__delitem__',
            (slice(-1),),
            ('removing', 'removed') * 2,
        ),
        (
            '__delitem__',
            (slice(-2, None),),
            ('removing', 'removed') * 2,
        ),
        # inherited interface
        ('append', (3,), ('inserting', 'inserted')),
        ('clear', (), ('removing', 'removed') * 3),
        ('count', (110,), ()),
        ('extend', ([7, 8, 9],), ('inserting', 'inserted') * 3),
        ('index', (110,), ()),
        ('pop', (-1,), ('removing', 'removed')),
        ('__add__', ([7, 8, 9],), ()),
        ('__iadd__', ([7, 9],), ('inserting', 'inserted') * 2),
    ],
    ids=lambda x: x[0],
)
def test_nested_events(meth, group_index):
    ne_list = NestableEventedList(NEST)
    ne_list.events = Mock(wraps=ne_list.events)

    method_name, args, expected_events = meth
    method = getattr(ne_list[group_index], method_name)
    if method_name == 'index' and group_index == (1, 1, 1):
        # the expected value of '110' (in the pytest parameters)
        # is not present in any child of ne_list[1, 1, 1]
        with pytest.raises(ValueError, match='is not in list'):
            method(*args)
    else:
        # make sure we can call the method without error
        method(*args)

    # make sure the correct event type and number was emitted
    for c, expected in zip(
        ne_list.events.call_args_list, expected_events, strict=False
    ):
        event = c.args[0]
        assert event.type == expected
        if group_index == ():
            # in the root group, the index will be an int relative to root
            assert isinstance(event.index, int)
        else:
            assert event.index[:-1] == group_index


def test_setting_nested_slice():
    ne_list = NestableEventedList(NEST)
    ne_list[(1, 1, 1, slice(2))] = [9, 10]
    assert tuple(ne_list[1, 1, 1]) == (9, 10, 1112)


NESTED_POS_INDICES = [
    # indices           2       (2, 1)
    # original = [0, 1, [(2,0), [(2,1,0), (2,1,1)], (2,2)], 3, 4]
    [(), (), [0, 1, [20, [210, 211], 22], 3, 4]],  # no-op
    [((2, 0), (2, 1, 1), (3,)), (1), [0, 20, 211, 3, 1, [[210], 22], 4]],
    [((2, 0), (2, 1, 1), (3,)), (2), [0, 1, 20, 211, 3, [[210], 22], 4]],
    [((2, 0), (2, 1, 1), (3,)), (3), [0, 1, [[210], 22], 20, 211, 3, 4]],
    [((2, 1, 1), (3,)), (2, 0), [0, 1, [211, 3, 20, [210], 22], 4]],
    [((2, 1, 1),), (2, 1, 0), [0, 1, [20, [211, 210], 22], 3, 4]],
    [((2, 1, 1), (3,)), (2, 1, 0), [0, 1, [20, [211, 3, 210], 22], 4]],
    [((2, 1, 1), (3,)), (2, 1, 1), [0, 1, [20, [210, 211, 3], 22], 4]],
    [((2, 1, 1),), (0,), [211, 0, 1, [20, [210], 22], 3, 4]],
    [((2, 1, 1),), (), [0, 1, [20, [210], 22], 3, 4, 211]],
]

NESTED_NEG_INDICES = [
    [((2, 0), (2, 1, 1), (3,)), (-1), [0, 1, [[210], 22], 4, 20, 211, 3]],
    [((2, 0), (2, 1, 1), (3,)), (-2), [0, 1, [[210], 22], 20, 211, 3, 4]],
    [((2, 0), (2, 1, 1), (3,)), (-4), [0, 1, 20, 211, 3, [[210], 22], 4]],
    [((2, 1, 1), (3,)), (2, -1), [0, 1, [20, [210], 22, 211, 3], 4]],
    [((2, 1, 1), (3,)), (2, -2), [0, 1, [20, [210], 211, 3, 22], 4]],
]

NESTED_INDICES = NESTED_POS_INDICES + NESTED_NEG_INDICES  # type: ignore


@pytest.mark.parametrize(('sources', 'dest', 'expectation'), NESTED_INDICES)
def test_nested_move_multiple(sources, dest, expectation):
    """Test that moving multiple indices works and emits right events."""
    ne_list = NestableEventedList([0, 1, [20, [210, 211], 22], 3, 4])
    ne_list.events = Mock(wraps=ne_list.events)
    ne_list.move_multiple(sources, dest)
    ne_list.events.reordered.assert_called_with(value=expectation)


class E:
    def __init__(self) -> None:
        self.events = EmitterGroup(test=None)


def test_child_events():
    """Test that evented lists bubble child events."""
    # create a random object that emits events
    e_obj = E()
    # and two nestable evented lists
    root = EventedList()
    observed = []
    root.events.connect(lambda e: observed.append(e))
    root.append(e_obj)
    e_obj.events.test(value='hi')
    obs = [(e.type, e.index, getattr(e, 'value', None)) for e in observed]
    expected = [
        ('inserting', 0, None),  # before we inserted b into root
        ('inserted', 0, e_obj),  # after b was inserted into root
        ('test', 0, 'hi'),  # when e_obj emitted an event called "test"
    ]
    for o, e in zip(obs, expected, strict=False):
        assert o == e


def test_nested_child_events():
    """Test that nested lists bubbles nested child events.

    If you add an object that implements the ``SupportsEvents`` Protocol
    (i.e. has an attribute ``events`` that is an ``EmitterGroup``), to a
    ``NestableEventedList``, then the parent container will re-emit those
    events (and this works recursively up to the root container).  The
    index/indices of each child(ren) that bubbled the event will be added
    to the event.

    See docstring of :ref:`NestableEventedList` for more info.
    """

    # create a random object that emits events
    e_obj = E()
    # and two nestable evented lists
    root = NestableEventedList()
    b = NestableEventedList()
    # collect all events emitted by the root list
    observed = []
    root.events.connect(lambda e: observed.append(e))

    # now append a list to root
    root.append(b)
    # and append the event-emitter object to the nested list
    b.append(e_obj)
    # then have the deeply nested event-emitter actually emit an event
    e_obj.events.test(value='hi')

    # look at the (type, index, and value) of all of the events emitted by root
    # and make sure they match expectations
    obs = [(e.type, e.index, getattr(e, 'value', None)) for e in observed]
    expected = [
        ('inserting', 0, None),  # before we inserted b into root
        ('inserted', 0, b),  # after b was inserted into root
        ('inserting', (0, 0), None),  # before we inserted e_obj into b
        ('inserted', (0, 0), e_obj),  # after e_obj was inserted into b
        ('test', (0, 0), 'hi'),  # when e_obj emitted an event called "test"
    ]
    for o, e in zip(obs, expected, strict=False):
        assert o == e


def test_evented_list_subclass():
    """Test that multiple inheritance maintains events from superclass."""

    class A:
        events = EmitterGroup(boom=None)

    class B(A, EventedList):
        pass

    lst = B([1, 2])
    assert hasattr(lst, 'events')
    assert 'boom' in lst.events.emitters
    assert lst == [1, 2]


def test_array_like_setitem():
    """Test that EventedList.__setitem__ works for array-like items"""
    array = np.array((10, 10))
    evented_list = EventedList([array])
    evented_list[0] = array
