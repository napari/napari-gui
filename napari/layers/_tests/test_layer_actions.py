import numpy as np
import pytest

from napari.components.layerlist import LayerList
from napari.layers import Image, Labels, Points, Shapes
from napari.layers._layer_actions import (
    _convert,
    _convert_dtype,
    _duplicate_layer,
    _hide_show_unselected,
    _project,
    _show_hide_selected,
    _toggle_visibility,
)


def test_duplicate_layers():
    def _dummy():
        pass

    layer_list = LayerList()
    layer_list.append(Points([[0, 0]], name="test"))
    layer_list.selection.active = layer_list[0]
    layer_list[0].events.data.connect(_dummy)
    assert len(layer_list[0].events.data.callbacks) == 2
    assert len(layer_list) == 1
    _duplicate_layer(layer_list)
    assert len(layer_list) == 2
    assert layer_list[0].name == "test"
    assert layer_list[1].name == "test copy"
    assert layer_list[1].events.source is layer_list[1]
    assert (
        len(layer_list[1].events.data.callbacks) == 1
    )  # `events` Event Emitter
    assert layer_list[1].source.parent() is layer_list[0]


def test_toggle_selected_layers():
    layer_list = make_three_layer_layerlist()
    layer_list[0].visible = False
    layer_list[1].visible = True
    layer_list[2].visible = True

    layer_list.selection.active = layer_list[0]
    layer_list.selection.add(layer_list[1])

    assert layer_list[0].visible is False
    assert layer_list[1].visible is True
    assert layer_list[2].visible is True

    _toggle_visibility(layer_list)

    assert layer_list[0].visible is True
    assert layer_list[1].visible is False
    assert layer_list[2].visible is True

    _toggle_visibility(layer_list)

    assert layer_list[0].visible is False
    assert layer_list[1].visible is True
    assert layer_list[2].visible is True


def test_hide_show_unselected_layers():
    layer_list = make_three_layer_layerlist()
    layer_list[0].visible = True
    layer_list[1].visible = True
    layer_list[2].visible = True

    layer_list.selection.active = layer_list[1]

    assert layer_list[0].visible is True
    assert layer_list[1].visible is True
    assert layer_list[2].visible is True

    _hide_show_unselected(layer_list)

    assert layer_list[0].visible is False
    assert layer_list[1].visible is True
    assert layer_list[2].visible is False

    _hide_show_unselected(layer_list)

    assert layer_list[0].visible is True
    assert layer_list[1].visible is True
    assert layer_list[2].visible is True


def test_show_hide_selected_layers():
    layer_list = make_three_layer_layerlist()
    layer_list[0].visible = False
    layer_list[1].visible = True
    layer_list[2].visible = True

    layer_list.selection.active = layer_list[0]
    layer_list.selection.add(layer_list[1])

    assert layer_list[0].visible is False
    assert layer_list[1].visible is True
    assert layer_list[2].visible is True

    _show_hide_selected(layer_list)

    assert layer_list[0].visible is True
    assert layer_list[1].visible is True
    assert layer_list[2].visible is True

    _show_hide_selected(layer_list)

    assert layer_list[0].visible is False
    assert layer_list[1].visible is False
    assert layer_list[2].visible is True


@pytest.mark.parametrize(
    'mode', ['max', 'min', 'std', 'sum', 'mean', 'median']
)
def test_projections(mode):
    ll = LayerList()
    ll.append(Image(np.random.rand(8, 8, 8)))
    assert len(ll) == 1
    assert ll[-1].data.ndim == 3
    _project(ll, mode=mode)
    assert len(ll) == 2
    # because keepdims = False
    assert ll[-1].data.shape == (8, 8)


@pytest.mark.parametrize(
    'mode',
    ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'],
)
def test_convert_dtype(mode):
    ll = LayerList()
    data = np.zeros((10, 10), dtype=np.int16)
    ll.append(Labels(data))
    assert ll[-1].data.dtype == np.int16

    data[5, 5] = 1000
    assert data[5, 5] == 1000
    if mode == 'int8' or mode == 'uint8':
        # label value 1000 is outside of the target data type range.
        with pytest.raises(AssertionError):
            _convert_dtype(ll, mode=mode)
        assert ll[-1].data.dtype == np.int16
    else:
        _convert_dtype(ll, mode=mode)
        assert ll[-1].data.dtype == np.dtype(mode)

    assert ll[-1].data[5, 5] == 1000
    assert ll[-1].data.flatten().sum() == 1000


@pytest.mark.parametrize(
    'layer, type_',
    [
        (Image(np.random.rand(10, 10)), 'labels'),
        (Labels(np.ones((10, 10), dtype=int)), 'image'),
        (Shapes([np.array([[0, 0], [0, 10], [10, 0], [10, 10]])]), 'labels'),
    ],
)
def test_convert_layer(layer, type_):
    ll = LayerList()
    layer.scale *= 1.5
    original_scale = layer.scale.copy()
    ll.append(layer)
    assert ll[0]._type_string != type_
    _convert(ll, type_)
    assert ll[0]._type_string == type_
    assert np.array_equal(ll[0].scale, original_scale)


def make_three_layer_layerlist():
    layer_list = LayerList()
    layer_list.append(Points([[0, 0]], name="test"))
    layer_list.append(Image(np.random.rand(8, 8, 8)))
    layer_list.append(Image(np.random.rand(8, 8, 8)))

    return layer_list
