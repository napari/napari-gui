"""
Ensure that layers and their convenience methods on the viewer
have the same signatures and docstrings.
"""

import gc
import inspect
import re
from unittest.mock import MagicMock, call

import numpy as np
import pytest
from numpydoc.docscrape import ClassDoc, FunctionDoc

import napari
from napari import Viewer, layers as module
from napari._tests.utils import check_viewer_functioning, layer_test_data
from napari.utils.misc import camel_to_snake

layers = []

for name in dir(module):
    obj = getattr(module, name)

    if obj is module.Layer or not inspect.isclass(obj):
        continue

    if issubclass(obj, module.Layer):
        layers.append(obj)


@pytest.mark.parametrize('layer', layers, ids=lambda layer: layer.__name__)
def test_docstring(layer):
    name = layer.__name__

    method_name = f'add_{camel_to_snake(name)}'
    method = getattr(Viewer, method_name)

    method_doc = FunctionDoc(method)
    layer_doc = ClassDoc(layer)

    # check summary section
    method_summary = ' '.join(method_doc['Summary'])  # join multi-line summary

    if name == 'Image':
        summary_format = 'Add one or more Image layers to the layer list.'
    else:
        summary_format = 'Add an? .+? layers? to the layer list.'

    assert re.match(
        summary_format, method_summary
    ), f"improper 'Summary' section of '{method_name}'"

    # check parameters section
    method_params = method_doc['Parameters']
    layer_params = layer_doc['Parameters']

    # Remove path parameter from viewer method if it exists
    method_params = [m for m in method_params if m.name != 'path']

    if name == 'Image':
        # For Image just test arguments that are in layer are in method
        named_method_params = [m.name for m in method_params]
        for layer_param in layer_params:
            l_name, l_type, l_description = layer_param
            assert l_name in named_method_params
    else:
        try:
            assert len(method_params) == len(layer_params)
            for method_param, layer_param in zip(
                method_params, layer_params, strict=False
            ):
                m_name, m_type, m_description = method_param
                l_name, l_type, l_description = layer_param

                # descriptions are treated as lists where each line is an
                # element
                m_description = ' '.join(m_description)
                l_description = ' '.join(l_description)

                assert m_name == l_name, 'different parameter names or order'
                assert (
                    m_type == l_type
                ), f"type mismatch of parameter '{m_name}'"
                assert (
                    m_description == l_description
                ), f"description mismatch of parameter '{m_name}'"
        except AssertionError as e:
            raise AssertionError(
                f"docstrings don't match for class {name}"
            ) from e

    # check returns section
    (method_returns,) = method_doc[
        'Returns'
    ]  # only one thing should be returned
    description = ' '.join(method_returns[-1])  # join multi-line description
    method_returns = *method_returns[:-1], description

    if name == 'Image':
        assert method_returns == (
            'layer',
            f':class:`napari.layers.{name}` or list',
            f'The newly-created {name.lower()} layer or list of {name.lower()} layers.',
        ), f"improper 'Returns' section of '{method_name}'"
    else:
        assert method_returns == (
            'layer',
            f':class:`napari.layers.{name}`',
            f'The newly-created {name.lower()} layer.',
        ), f"improper 'Returns' section of '{method_name}'"


@pytest.mark.parametrize('layer', layers, ids=lambda layer: layer.__name__)
def test_signature(layer):
    name = layer.__name__

    method = getattr(Viewer, f'add_{camel_to_snake(name)}')

    class_parameters = dict(inspect.signature(layer.__init__).parameters)
    method_parameters = dict(inspect.signature(method).parameters)

    fail_msg = f"signatures don't match for class {name}"
    if name == 'Image':
        # If Image just test that class params appear in method
        for class_param in class_parameters:
            assert class_param in method_parameters, fail_msg
    else:
        assert class_parameters == method_parameters, fail_msg


# plugin_manager fixture is added to prevent errors due to installed plugins
@pytest.mark.parametrize(('layer_type', 'data', 'ndim'), layer_test_data)
def test_view(qtbot, napari_plugin_manager, layer_type, data, ndim):
    np.random.seed(0)
    viewer = getattr(napari, f'view_{layer_type.__name__.lower()}')(
        data, show=False
    )
    view = viewer.window._qt_viewer
    check_viewer_functioning(viewer, view, data, ndim)
    viewer.close()


# plugin_manager fixture is added to prevent errors due to installed plugins
def test_view_multichannel(qtbot, napari_plugin_manager):
    """Test adding image."""
    np.random.seed(0)
    data = np.random.random((15, 10, 5))
    viewer = napari.view_image(data, channel_axis=-1, show=False)
    assert len(viewer.layers) == data.shape[-1]
    for i in range(data.shape[-1]):
        np.testing.assert_array_equal(
            viewer.layers[i].data, data.take(i, axis=-1)
        )
    viewer.close()


def test_kwargs_passed(monkeypatch):
    import napari.view_layers

    viewer_mock = MagicMock(napari.Viewer)
    monkeypatch.setattr(napari.view_layers, 'Viewer', viewer_mock)
    napari.view_path(
        path='some/path',
        title='my viewer',
        ndisplay=3,
        name='img name',
        scale=(1, 2, 3),
    )
    assert viewer_mock.mock_calls == [
        call(title='my viewer'),
        call().open(path='some/path', name='img name', scale=(1, 2, 3)),
    ]


# plugin_manager fixture is added to prevent errors due to installed plugins
def test_imshow(qtbot, napari_plugin_manager):
    shape = (10, 15)
    ndim = len(shape)
    np.random.seed(0)
    data = np.random.random(shape)
    viewer, layer = napari.imshow(data, channel_axis=None, show=False)
    view = viewer.window._qt_viewer
    check_viewer_functioning(viewer, view, data, ndim)
    assert isinstance(layer, napari.layers.Image)
    viewer.close()


# plugin_manager fixture is added to prevent errors due to installed plugins
def test_imshow_multichannel(qtbot, napari_plugin_manager):
    """Test adding image."""
    np.random.seed(0)
    data = np.random.random((15, 10, 5))
    viewer, layers = napari.imshow(data, channel_axis=-1, show=False)
    assert len(layers) == data.shape[-1]
    assert isinstance(layers, tuple)
    for i in range(data.shape[-1]):
        np.testing.assert_array_equal(layers[i].data, data.take(i, axis=-1))
    viewer.close()
    # Run a full garbage collection here so that any remaining viewer
    # and related instances are removed for future tests that may use
    # make_napari_viewer.
    gc.collect()


# plugin_manager fixture is added to prevent errors due to installed plugins
def test_imshow_with_viewer(qtbot, napari_plugin_manager, make_napari_viewer):
    shape = (10, 15)
    ndim = len(shape)
    np.random.seed(0)
    data = np.random.random(shape).astype(np.float32)
    viewer = make_napari_viewer()
    viewer2, layer = napari.imshow(data, viewer=viewer, show=False)
    assert viewer is viewer2
    np.testing.assert_array_equal(data, layer.data)
    view = viewer.window._qt_viewer
    check_viewer_functioning(viewer, view, data, ndim)
    viewer.close()
