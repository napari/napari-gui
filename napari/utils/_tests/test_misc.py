from enum import auto
from importlib.metadata import version as package_version
from os.path import abspath, expanduser, sep
from pathlib import Path

import numpy as np
import pytest
from packaging.version import parse as parse_version

from napari.utils.misc import (
    StringEnum,
    _is_array_type,
    _pandas_dataframe_equal,
    _quiet_array_equal,
    abspath_or_url,
    ensure_iterable,
    ensure_list_of_layer_data_tuple,
    ensure_sequence_of_iterables,
    is_iterable,
    pick_equality_operator,
)

ITERABLE = (0, 1, 2)
NESTED_ITERABLE = [ITERABLE, ITERABLE, ITERABLE]
DICT = {'a': 1, 'b': 3, 'c': 5}
LIST_OF_DICTS = [DICT, DICT, DICT]
PARTLY_NESTED_ITERABLE = [ITERABLE, None, None]
REPEATED_PARTLY_NESTED_ITERABLE = [PARTLY_NESTED_ITERABLE] * 3


@pytest.mark.parametrize(
    ('input_data', 'expected'),
    [
        (ITERABLE, NESTED_ITERABLE),
        (NESTED_ITERABLE, NESTED_ITERABLE),
        ((ITERABLE, (2,), (3, 1, 6)), (ITERABLE, (2,), (3, 1, 6))),
        (DICT, LIST_OF_DICTS),
        (LIST_OF_DICTS, LIST_OF_DICTS),
        (None, (None, None, None)),
        (PARTLY_NESTED_ITERABLE, REPEATED_PARTLY_NESTED_ITERABLE),
        ([], ([], [], [])),
    ],
)
def test_sequence_of_iterables(input_data, expected):
    """Test ensure_sequence_of_iterables returns a sequence of iterables."""
    zipped = zip(
        range(3),
        ensure_sequence_of_iterables(input_data, repeat_empty=True),
        expected,
        strict=False,
    )
    for _i, result, expectation in zipped:
        assert result == expectation


def test_sequence_of_iterables_allow_none():
    input_data = [(1, 2), None]
    assert (
        ensure_sequence_of_iterables(input_data, allow_none=True) == input_data
    )


def test_sequence_of_iterables_no_repeat_empty():
    assert ensure_sequence_of_iterables([], repeat_empty=False) == []
    with pytest.raises(ValueError, match='must equal'):
        ensure_sequence_of_iterables([], repeat_empty=False, length=3)


def test_sequence_of_iterables_raises():
    with pytest.raises(ValueError, match='must equal'):
        # the length argument asserts a specific length
        ensure_sequence_of_iterables(((0, 1),), length=4)

    # BEWARE: only the first element of a nested sequence is checked.
    iterable = (None, (0, 1), (0, 2))
    result = iter(ensure_sequence_of_iterables(iterable))
    with pytest.raises(AssertionError):
        assert next(result) is None


@pytest.mark.parametrize(
    ('input_data', 'expected'),
    [
        (ITERABLE, ITERABLE),
        (DICT, DICT),
        (1, [1, 1, 1]),
        ('foo', ['foo', 'foo', 'foo']),
        (None, [None, None, None]),
    ],
)
def test_ensure_iterable(input_data, expected):
    """Test test_ensure_iterable returns an iterable."""
    zipped = zip(range(3), ensure_iterable(input_data), expected, strict=False)
    for _i, result, expectation in zipped:
        assert result == expectation


def test_string_enum():
    # Make a test StringEnum
    class TestEnum(StringEnum):
        THING = auto()
        OTHERTHING = auto()

    # test setting by value, correct case
    assert TestEnum('thing') == TestEnum.THING

    # test setting by value mixed case
    assert TestEnum('thInG') == TestEnum.THING

    # test setting by instance of self
    assert TestEnum(TestEnum.THING) == TestEnum.THING

    # test setting by name correct case
    assert TestEnum['THING'] == TestEnum.THING

    # test setting by name mixed case
    assert TestEnum['tHiNg'] == TestEnum.THING

    # test setting by value with incorrect value
    with pytest.raises(ValueError, match='not a valid'):
        TestEnum('NotAThing')

    # test  setting by name with incorrect name
    with pytest.raises(KeyError):
        TestEnum['NotAThing']

    # test creating a StringEnum with the functional API
    animals = StringEnum('Animal', 'AARDVARK BUFFALO CAT DOG')
    assert str(animals.AARDVARK) == 'aardvark'
    assert animals('BUffALO') == animals.BUFFALO
    assert animals['BUffALO'] == animals.BUFFALO

    # test setting by instance of self
    class OtherEnum(StringEnum):
        SOMETHING = auto()

    #  test setting by instance of a different StringEnum is an error
    with pytest.raises(ValueError, match='may only be called with'):
        TestEnum(OtherEnum.SOMETHING)

    # test string conversion
    assert str(TestEnum.THING) == 'thing'

    # test direct comparison with a string
    assert TestEnum.THING == 'thing'
    assert TestEnum.THING == 'thing'
    assert TestEnum.THING != 'notathing'
    assert TestEnum.THING != 'notathing'

    # test comparison with another enum with same value names
    class AnotherTestEnum(StringEnum):
        THING = auto()
        ANOTHERTHING = auto()

    assert TestEnum.THING != AnotherTestEnum.THING

    # test lookup in a set
    assert TestEnum.THING in {TestEnum.THING, TestEnum.OTHERTHING}
    assert TestEnum.THING not in {TestEnum.OTHERTHING}
    assert TestEnum.THING in {'thing', TestEnum.OTHERTHING}
    assert TestEnum.THING not in {
        AnotherTestEnum.THING,
        AnotherTestEnum.ANOTHERTHING,
    }


def test_abspath_or_url():
    relpath = '~' + sep + 'something'
    assert abspath_or_url(relpath) == expanduser(relpath)
    assert abspath_or_url('something') == abspath('something')
    assert abspath_or_url(sep + 'something') == abspath(sep + 'something')
    assert abspath_or_url('https://something') == 'https://something'
    assert abspath_or_url('http://something') == 'http://something'
    assert abspath_or_url('ftp://something') == 'ftp://something'
    assert abspath_or_url('s3://something') == 's3://something'
    assert abspath_or_url('file://something') == 'file://something'

    with pytest.raises(TypeError):
        abspath_or_url({'a', '~'})


def test_type_stable():
    assert isinstance(abspath_or_url('~'), str)
    assert isinstance(abspath_or_url(Path('~')), Path)


def test_equality_operator():
    import operator

    import dask.array as da
    import numpy as np
    import pandas as pd
    import xarray as xr
    import zarr

    class MyNPArray(np.ndarray):
        pass

    assert pick_equality_operator(np.ones((1, 1))) == _quiet_array_equal
    assert pick_equality_operator(MyNPArray([1, 1])) == _quiet_array_equal
    assert pick_equality_operator(da.ones((1, 1))) == operator.is_
    assert pick_equality_operator(zarr.ones((1, 1))) == operator.is_
    assert (
        pick_equality_operator(xr.DataArray(np.ones((1, 1))))
        == _quiet_array_equal
    )
    assert pick_equality_operator(
        pd.DataFrame({'A': [1]}) == _pandas_dataframe_equal
    )


@pytest.mark.skipif(
    parse_version(package_version('numpy')) >= parse_version('1.25.0'),
    reason='Numpy 1.25.0 return true for below comparison',
)
def test_equality_operator_silence():
    import numpy as np

    eq = pick_equality_operator(np.asarray([]))
    # make sure this doesn't warn
    assert not eq(np.asarray([]), np.asarray([], '<U32'))


def test_is_array_type_with_xarray():
    import numpy as np
    import xarray as xr

    assert _is_array_type(xr.DataArray(), 'xarray.DataArray')
    assert not _is_array_type(xr.DataArray(), 'xr.DataArray')
    assert not _is_array_type(
        xr.DataArray(), 'xarray.core.dataarray.DataArray'
    )
    assert not _is_array_type([], 'xarray.DataArray')
    assert not _is_array_type(np.array([]), 'xarray.DataArray')


@pytest.mark.parametrize(
    ('input_data', 'expected'),
    [
        ([([1, 10],)], [([1, 10],)]),
        ([([1, 10], {'name': 'hi'})], [([1, 10], {'name': 'hi'})]),
        (
            [([1, 10], {'name': 'hi'}, 'image')],
            [([1, 10], {'name': 'hi'}, 'image')],
        ),
        ([], []),
    ],
)
def test_ensure_list_of_layer_data_tuple(input_data, expected):
    """Ensure that when given layer data that a tuple can be generated.

    When data with a name is supplied a layer should be created and named.
    When an empty dataset is supplied no layer is created and no errors are produced.
    """
    assert ensure_list_of_layer_data_tuple(input_data) == expected


@pytest.mark.parametrize(
    ('data', 'expected'),
    [
        (1, False),
        (1.0, False),
        ([1], True),
        ('aaa', False),
        (object(), False),
        (None, False),
        (np.arange(5), True),
        ({1, 2, 3}, True),
    ],
)
def test_is_iterable(data, expected):
    assert is_iterable(data) == expected
