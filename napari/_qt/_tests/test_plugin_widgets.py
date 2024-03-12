from itertools import dropwhile
from unittest.mock import Mock, patch

import pytest
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget

import napari
from napari import Viewer
from napari._qt.menus import PluginsMenu
from napari._qt.qt_main_window import _instantiate_dock_widget
from napari.utils._proxies import PublicOnlyProxy


class Widg1(QWidget):
    pass


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
    'single_class': Widg1,
    'class_tuple': (Widg1, {'area': 'right'}),
    'tuple_list': [(Widg1, {'area': 'right'}), (Widg2, {})],
    'tuple_list2': [(Widg1, {'area': 'right'}), Widg2],
    'bad_class': 1,
    'bad_tuple1': (Widg1, 1),
    'bad_double_tuple': ((Widg1, {}), (Widg2, {})),
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
        assert widgets['Plugin']['Widg1'][0] == Widg1
        if 'tuple_list' in request.node.name:
            assert widgets['Plugin']['Widg2'][0] == Widg2


@pytest.fixture
def test_plugin_widgets(monkeypatch, napari_plugin_manager):
    """A smattering of example registered dock widgets and function widgets."""
    tnpm = napari_plugin_manager
    dock_widgets = {
        'TestP1': {'Widg1': (Widg1, {}), 'Widg2': (Widg2, {})},
        'TestP2': {'Widg3': (Widg3, {})},
    }
    monkeypatch.setattr(tnpm, '_dock_widgets', dock_widgets)

    function_widgets = {'TestP3': {'magic': magicfunc}}
    monkeypatch.setattr(tnpm, '_function_widgets', function_widgets)
    yield


def test_plugin_widgets_menus(test_plugin_widgets, qtbot):
    """Test the plugin widgets get added to the window menu correctly."""
    # only take the plugin actions
    window = Mock()
    qtwin = QWidget()
    qtbot.addWidget(qtwin)
    with patch.object(window, '_qt_window', qtwin):
        actions = PluginsMenu(window=window).actions()
    actions = list(dropwhile(lambda a: a.text() != '', actions))
    texts = [a.text() for a in actions][1:]
    for t in ['TestP1', 'Widg3 (TestP2)', 'magic (TestP3)']:
        assert t in texts

    # Expect a submenu ("Test plugin1") with particular entries.
    tp1 = next(m for m in actions if m.text() == 'TestP1')
    assert tp1.parent()
    assert [a.text() for a in tp1.parent().actions()] == ['Widg1', 'Widg2']


def test_making_plugin_dock_widgets(
    test_plugin_widgets, make_napari_viewer, qtbot
):
    """Test that we can create dock widgets, and they get the viewer."""
    viewer = make_napari_viewer()
    # only take the plugin actions
    actions = viewer.window.plugins_menu.actions()
    actions = list(dropwhile(lambda a: a.text() != '', actions))

    # trigger the 'TestP2: Widg3' action
    tp2 = next(m for m in actions if m.text().endswith('(TestP2)'))
    tp2.trigger()
    # make sure that a dock widget was created
    assert 'Widg3 (TestP2)' in viewer.window._dock_widgets
    dw = viewer.window._dock_widgets['Widg3 (TestP2)']
    assert isinstance(dw.widget(), Widg3)
    # This widget uses the parameter annotation method to receive a viewer
    assert isinstance(dw.widget().viewer, napari.Viewer)
    # Add twice is ok, only does a show
    tp2.trigger()

    # trigger the 'TestP1 > Widg2' action (it's in a submenu)
    tp2 = next(m for m in actions if m.text().endswith('TestP1'))
    action = tp2.parent().actions()[1]
    assert action.text() == 'Widg2'
    action.trigger()
    # make sure that a dock widget was created
    assert 'Widg2 (TestP1)' in viewer.window._dock_widgets
    dw = viewer.window._dock_widgets['Widg2 (TestP1)']
    assert isinstance(dw.widget(), Widg2)
    # This widget uses parameter *name* "napari_viewer" to get a viewer
    assert isinstance(dw.widget().viewer, napari.Viewer)
    # Add twice is ok, only does a show
    action.trigger()
    # Check that widget is still there when closed.
    widg = dw.widget()
    dw.title.hide_button.click()
    assert widg
    # Check that widget is destroyed when closed.
    dw.destroyOnClose()
    assert action not in viewer.window.plugins_menu.actions()
    assert widg.parent() is None
    widg.deleteLater()
    widg.close()
    qtbot.wait(50)


def test_making_function_dock_widgets(test_plugin_widgets, make_napari_viewer):
    """Test that we can create magicgui widgets, and they get the viewer."""
    import magicgui

    viewer = make_napari_viewer()
    # only take the plugin actions
    actions = viewer.window.plugins_menu.actions()
    actions = dropwhile(lambda a: a.text() != '', actions)

    # trigger the 'TestP3: magic' action
    tp3 = next(m for m in actions if m.text().endswith('(TestP3)'))
    tp3.trigger()
    # make sure that a dock widget was created
    assert 'magic (TestP3)' in viewer.window._dock_widgets
    dw = viewer.window._dock_widgets['magic (TestP3)']
    # make sure that it contains a magicgui widget
    magic_widget = dw.widget()._magic_widget
    assert isinstance(magic_widget, magicgui.widgets.FunctionGui)
    # This magicgui widget uses the parameter annotation to receive a viewer
    assert isinstance(magic_widget.viewer.value, napari.Viewer)
    # The function just returns the viewer... make sure we can call it
    assert isinstance(magic_widget(), napari.Viewer)
    # Add twice is ok, only does a show
    tp3.trigger()


def test_inject_viewer_proxy(make_napari_viewer):
    """Test that the injected viewer is a public-only proxy"""
    viewer = make_napari_viewer()
    wdg = _instantiate_dock_widget(Widg3, viewer)
    assert isinstance(wdg.viewer, PublicOnlyProxy)

    # simulate access from outside napari
    with patch('napari.utils.misc.ROOT_DIR', new='/some/other/package'):
        with pytest.warns(FutureWarning):
            wdg.fail()


def test_get_widgets(make_napari_viewer):
    viewer = make_napari_viewer()
    # TODO: open a widget and make sure it's in this list
    assert viewer.window.dock_widgets == {}
