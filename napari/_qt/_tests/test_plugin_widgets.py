from unittest.mock import MagicMock, patch

import pytest
from magicgui import magic_factory, magicgui
from magicgui.widgets import Container
from napari_plugin_engine import napari_hook_implementation
from npe2 import DynamicPlugin
from qtpy.QtWidgets import QWidget

import napari
from napari._app_model import get_app
from napari._qt.qt_main_window import _instantiate_dock_widget
from napari.plugins._npe2 import _get_widget_viewer_param
from napari.utils._proxies import PublicOnlyProxy
from napari.viewer import Viewer


class ErrorWidget:
    pass


class QWidget_example(QWidget):
    def __init__(napari_viewer):
        pass


class QWidget_string_annnot(QWidget):
    def __init__(test: "napari.viewer.Viewer"):
        pass


class Container_example(Container):
    def __init__(test: Viewer):
        pass


@magic_factory
def magic_widget_example():
    pass


def callable_example():
    @magicgui
    def magic_widget_example():
        pass

    return magic_widget_example


class Widg2(QWidget):
    def __init__(self, napari_viewer) -> None:
        self.viewer = napari_viewer
        super().__init__()


class Widg3(QWidget):
    def __init__(self, v: Viewer) -> None:
        self.viewer = v
        super().__init__()

    def fail(self):
        """private attr not allowed"""
        self.viewer.window._qt_window


def magicfunc(viewer: 'napari.Viewer'):
    return viewer


dwidget_args = {
    'single_class': QWidget_example,
    'class_tuple': (QWidget_example, {'area': 'right'}),
    'tuple_list': [(QWidget_example, {'area': 'right'}), (Widg2, {})],
    'tuple_list2': [(QWidget_example, {'area': 'right'}), Widg2],
    'bad_class': 1,
    'bad_tuple1': (QWidget_example, 1),
    'bad_double_tuple': ((QWidget_example, {}), (Widg2, {})),
}


# napari_plugin_manager from _testsupport.py
# monkeypatch, request, recwarn fixtures are from pytest
@pytest.mark.parametrize('arg', dwidget_args.values(), ids=dwidget_args.keys())
def test_dock_widget_registration(
    arg, napari_plugin_manager, request, recwarn
):
    """Test that dock widgets get validated and registerd correctly."""

    class Plugin:
        @napari_hook_implementation
        def napari_experimental_provide_dock_widget():
            return arg

    napari_plugin_manager.register(Plugin, name='Plugin')
    napari_plugin_manager.discover_widgets()
    widgets = napari_plugin_manager._dock_widgets

    if '[bad_' in request.node.name:
        assert len(recwarn) == 1
        assert not widgets
    else:
        assert len(recwarn) == 0
        assert widgets['Plugin']['Q Widget_example'][0] == QWidget_example
        if 'tuple_list' in request.node.name:
            assert widgets['Plugin']['Widg2'][0] == Widg2


@pytest.fixture
def test_plugin_widgets(monkeypatch, napari_plugin_manager):
    """A smattering of example registered dock widgets and function widgets."""
    tnpm = napari_plugin_manager
    dock_widgets = {
        "TestP1": {
            "QWidget_example": (QWidget_example, {}),
            "Widg2": (Widg2, {}),
        },
        "TestP2": {"Widg3": (Widg3, {})},
    }
    monkeypatch.setattr(tnpm, "_dock_widgets", dock_widgets)

    function_widgets = {'TestP3': {'magic': magicfunc}}
    monkeypatch.setattr(tnpm, "_function_widgets", function_widgets)
    yield


def test_inject_viewer_proxy(make_napari_viewer):
    """Test that the injected viewer is a public-only proxy"""
    viewer = make_napari_viewer()
    wdg = _instantiate_dock_widget(Widg3, viewer)
    assert isinstance(wdg.viewer, PublicOnlyProxy)

    # simulate access from outside napari
    with patch('napari.utils.misc.ROOT_DIR', new='/some/other/package'):
        with pytest.warns(FutureWarning):
            wdg.fail()


@pytest.mark.parametrize(
    "widget_callable, param",
    [
        (QWidget_example, 'napari_viewer'),
        (QWidget_string_annnot, 'test'),
        (Container_example, 'test'),
    ],
)
def test_get_widget_viewer_param(widget_callable, param):
    """Test `_get_widget_viewer_param` returns correct parameter name."""
    out = _get_widget_viewer_param(widget_callable, 'widget_name')
    assert out == param


def test_get_widget_viewer_param_error():
    """Test incorrect subclass raises error in `_get_widget_viewer_param`."""
    with pytest.raises(TypeError) as e:
        _get_widget_viewer_param(ErrorWidget, 'widget_name')
    assert "'widget_name' must be `QtWidgets.QWidget`" in str(e)


def test_widget_hide_destroy(make_napari_viewer):
    """Test that widget hide and destroy works."""
    viewer = make_napari_viewer()
    viewer.window.add_dock_widget(QWidget_example(), name='test')
    widget = viewer.window._dock_widgets['test']

    # Check widget persists after hide
    widget.title.hide_button.click()
    assert widget
    # Check that widget removed from `_dock_widgets` dict when closed
    widget.destroyOnClose()
    assert 'test' not in viewer.window._dock_widgets


@pytest.mark.parametrize(
    "Widget",
    [
        QWidget_example,
        Container_example,
        magic_widget_example,
        callable_example,
    ],
)
def test_widget_types_supported(
    make_napari_viewer, tmp_plugin: DynamicPlugin, Widget
):
    """Test all supported widget types correctly instantiated and call processor."""

    @tmp_plugin.contribute.widget(display_name='Widget')
    def widget():
        return Widget()

    app = get_app()
    # `strict_qt` prevents checking for leaked widgets resulting from parametrize
    viewer = make_napari_viewer(strict_qt=False)

    viewer.window.add_dock_widget = MagicMock()
    app.commands.execute_command('tmp_plugin:Widget')
    viewer.window.add_dock_widget.assert_called_once()
