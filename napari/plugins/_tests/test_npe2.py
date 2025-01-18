from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import npe2
import numpy as np
import pytest
from npe2 import DynamicPlugin, PluginManifest

if TYPE_CHECKING:
    from npe2._pytest_plugin import TestPluginManager

from napari.layers import Image, Points
from napari.plugins import _npe2

PLUGIN_NAME = 'my-plugin'  # this matches the sample_manifest
PLUGIN_DISPLAY_NAME = 'My Plugin'  # this matches the sample_manifest
MANIFEST_PATH = Path(__file__).parent / '_sample_manifest.yaml'


@pytest.fixture
def mock_pm(npe2pm: 'TestPluginManager'):
    from napari.plugins import _initialize_plugins

    _initialize_plugins.cache_clear()
    mock_reg = MagicMock()
    npe2pm._command_registry = mock_reg
    with npe2pm.tmp_plugin(manifest=MANIFEST_PATH):
        yield npe2pm


def test_read(mock_pm: 'TestPluginManager'):
    _, hookimpl = _npe2.read(['some.fzzy'], stack=False)
    mock_pm.commands.get.assert_called_once_with(f'{PLUGIN_NAME}.some_reader')
    assert hookimpl.plugin_name == PLUGIN_NAME

    mock_pm.commands.get.reset_mock()
    _, hookimpl = _npe2.read(['some.fzzy'], stack=True)
    mock_pm.commands.get.assert_called_once_with(f'{PLUGIN_NAME}.some_reader')
    mock_pm.commands.get.reset_mock()
    with pytest.raises(ValueError, match='No compatible readers'):
        _npe2.read(['some.randomext'], stack=False)
    mock_pm.commands.get.assert_not_called()

    mock_pm.commands.get.reset_mock()
    assert (
        _npe2.read(['some.randomext'], stack=True, plugin='not-npe2-plugin')
        is None
    )
    mock_pm.commands.get.assert_not_called()

    mock_pm.commands.get.reset_mock()
    _, hookimpl = _npe2.read(
        ['some.fzzy'], stack=False, plugin='my-plugin.some_reader'
    )
    mock_pm.commands.get.assert_called_once_with(f'{PLUGIN_NAME}.some_reader')
    assert hookimpl.plugin_name == PLUGIN_NAME


@pytest.mark.skipif(
    npe2.__version__ < '0.7.0',
    reason='Older versions of npe2 do not throw specific error.',
)
def test_read_with_plugin_failure(mock_pm: 'TestPluginManager'):
    with pytest.raises(ValueError, match='is not a compatible reader'):
        _npe2.read(['some.randomext'], stack=True, plugin=PLUGIN_NAME)


def test_write(mock_pm: 'TestPluginManager'):
    # saving an image without a writer goes straight to npe2.write
    # it will use our plugin writer
    image = Image(np.random.rand(20, 20), name='ex_img')
    _npe2.write_layers('some_file.tif', [image])
    mock_pm.commands.get.assert_called_once_with(f'{PLUGIN_NAME}.my_writer')

    # points won't trigger our sample writer
    mock_pm.commands.get.reset_mock()
    points = Points(np.random.rand(20, 2), name='ex_points')
    _npe2.write_layers('some_file.tif', [points])
    mock_pm.commands.get.assert_not_called()

    # calling _npe2.write_layers with a specific writer contribution should
    # directly execute the writer.exec with arguments appropriate for the
    # writer spec (single or multi-writer)
    mock_pm.commands.get.reset_mock()
    writer = mock_pm.get_manifest(PLUGIN_NAME).contributions.writers[0]
    writer = MagicMock(wraps=writer)
    writer.exec.return_value = ['']
    assert _npe2.write_layers('some_file.tif', [points], writer=writer)[0] == [
        ''
    ]
    mock_pm.commands.get.assert_not_called()
    writer.exec.assert_called_once()
    assert writer.exec.call_args_list[0].kwargs['args'][0] == 'some_file.tif'


def test_get_widget_contribution(mock_pm: 'TestPluginManager'):
    # calling with plugin alone
    (_, display_name) = _npe2.get_widget_contribution(PLUGIN_NAME)
    mock_pm.commands.get.assert_called_once_with('my-plugin.some_widget')
    assert display_name == 'My Widget'

    # calling with plugin but wrong widget name provides a useful error msg
    with pytest.raises(KeyError) as e:
        _npe2.get_widget_contribution(PLUGIN_NAME, 'Not a widget')
    assert (
        f"Plugin {PLUGIN_NAME!r} does not provide a widget named 'Not a widget'"
        in str(e.value)
    )

    # calling with a non-existent plugin just returns None
    mock_pm.commands.get.reset_mock()
    assert not _npe2.get_widget_contribution('not-a-thing')
    mock_pm.commands.get.assert_not_called()


def test_get_widget_contribution_no_widgets(tmp_plugin: DynamicPlugin):
    """Test error raised when `widget_name` provided but plugin provides no widgets."""
    with pytest.raises(KeyError) as e:
        _npe2.get_widget_contribution('tmp_plugin', 'No widgets')
    assert "Plugin 'tmp_plugin' does not provide any widgets" in str(e.value)


def test_populate_qmenu(mock_pm: 'TestPluginManager'):
    menu = MagicMock()
    _npe2.populate_qmenu(menu, 'napari/file/new_layer')
    menu.addMenu.assert_called_once_with('My SubMenu')
    menu.addAction.assert_called_once_with('Hello World')


def test_file_extensions_string_for_layers(mock_pm: 'TestPluginManager'):
    layers = [Image(np.random.rand(20, 20), name='ex_img')]
    label, writers = _npe2.file_extensions_string_for_layers(layers)
    assert label == 'My Plugin (*.tif *.tiff)'
    writer = mock_pm.get_manifest(PLUGIN_NAME).contributions.writers[0]
    assert writers == [writer]


def test_get_readers(mock_pm):
    assert _npe2.get_readers('some.fzzy') == {PLUGIN_NAME: 'My Plugin'}


def test_iter_manifest(mock_pm):
    for i in _npe2.iter_manifests():
        assert isinstance(i, PluginManifest)


def test_get_sample_data(mock_pm):
    samples = mock_pm.get_manifest(PLUGIN_NAME).contributions.sample_data

    opener, _ = _npe2.get_sample_data(PLUGIN_NAME, 'random_data')
    assert isinstance(opener, MethodType)
    assert opener.__self__ is samples[0]

    opener, _ = _npe2.get_sample_data(PLUGIN_NAME, 'internet_image')
    assert isinstance(opener, MethodType)
    assert opener.__self__ is samples[1]

    opener, avail = _npe2.get_sample_data('not-a-plugin', 'nor-a-sample')
    assert opener is None
    assert avail == [
        (PLUGIN_NAME, 'random_data'),
        (PLUGIN_NAME, 'internet_image'),
    ]


def test_sample_iterator(mock_pm):
    samples = list(_npe2.sample_iterator())
    assert samples
    for plugin, contribs in samples:
        assert isinstance(plugin, str)
        assert isinstance(contribs, dict)
        # check that the manifest display_name is used
        assert plugin == PLUGIN_NAME
        assert contribs
        for i in contribs.values():
            assert 'data' in i
            assert 'display_name' in i


def test_widget_iterator(mock_pm):
    wdgs = list(_npe2.widget_iterator())
    assert wdgs == [('dock', (PLUGIN_NAME, ['My Widget']))]


def test_plugin_actions(mock_pm: 'TestPluginManager', mock_app_model):
    from napari._app_model import get_app_model
    from napari.plugins import _initialize_plugins

    app = get_app_model()
    # nothing yet registered with this menu
    assert 'napari/file/new_layer' not in app.menus
    # menus_items1 = list(app.menus.get_menu('napari/file/new_layer'))
    # assert 'my-plugin.hello_world' not in app.commands

    _initialize_plugins()  # connect registration callbacks and populate registries
    # the _sample_manifest should have added two items to menus

    # now we have command registered
    menus_items2 = list(app.menus.get_menu('napari/file/new_layer'))
    assert 'my-plugin.hello_world' in app.commands

    assert len(menus_items2) == 2

    # then disable and re-enable the plugin

    mock_pm.disable(PLUGIN_NAME)

    assert 'napari/file/new_layer' not in app.menus

    mock_pm.enable(PLUGIN_NAME)

    menus_items4 = list(app.menus.get_menu('napari/file/new_layer'))
    assert len(menus_items4) == 2
    assert 'my-plugin.hello_world' in app.commands
