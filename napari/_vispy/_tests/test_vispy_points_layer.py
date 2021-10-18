import numpy as np
import pytest

from napari._vispy.layers.points import VispyPointsLayer
from napari.layers import Points


@pytest.mark.parametrize("opacity", [0, 0.3, 0.7, 1])
def test_VispyPointsLayer(opacity):
    points = np.array([[100, 100], [200, 200], [300, 100]])
    layer = Points(points, size=30, opacity=opacity)
    visual = VispyPointsLayer(layer)
    assert visual.node.opacity == opacity


def test_no_points_no_error():
    layer = Points()
    VispyPointsLayer(layer)


def test_change_text_updates_node_string():
    points = np.random.rand(3, 2)
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'name': np.array(['D', 'E', 'F']),
    }
    layer = Points(points, text='class', properties=properties)
    vispy_layer = VispyPointsLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, properties['class'])

    layer.text = 'name'

    np.testing.assert_array_equal(text_node.text, properties['name'])


def test_change_text_color_updates_node_color():
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    text = {'string': 'class', 'color': [1, 0, 0]}
    layer = Points(points, text=text, properties=properties)
    vispy_layer = VispyPointsLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.color.rgb, [[1, 0, 0]] * 3)

    layer.text.color = [0, 0, 1]

    np.testing.assert_array_equal(text_node.color.rgb, [[0, 0, 1]] * 3)


def test_change_properties_updates_node_strings(make_napari_viewer):
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Points(points, properties=properties, text='class')
    vispy_layer = VispyPointsLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.properties = {'class': np.array(['D', 'E', 'F'])}

    np.testing.assert_array_equal(text_node.text, ['D', 'E', 'F'])


def test_update_property_value_then_refresh_text_updates_node_strings():
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Points(points, properties=properties, text='class')
    vispy_layer = VispyPointsLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.properties['class'][1] = 'D'
    layer.refresh_text()

    np.testing.assert_array_equal(text_node.text, ['A', 'D', 'C'])
