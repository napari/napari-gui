import gc
import sys
from unittest import mock

import pytest

import napari
from napari import __main__


@pytest.fixture
def mock_run():
    """mock to prevent starting the event loop."""
    with mock.patch('napari._qt.widgets.qt_splash_screen.NapariSplashScreen'):
        with mock.patch('napari.run'):
            yield napari.run


def test_cli_works(monkeypatch, capsys):
    """Test the cli runs and shows help"""
    monkeypatch.setattr(sys, 'argv', ['napari', '-h'])
    with pytest.raises(SystemExit):
        __main__._run()
    assert 'napari command line viewer.' in str(capsys.readouterr())


def test_cli_shows_plugins(monkeypatch, capsys, tmp_plugin):
    """Test the cli --info runs and shows plugins"""
    monkeypatch.setattr(sys, 'argv', ['napari', '--info'])
    with pytest.raises(SystemExit):
        __main__._run()
    assert tmp_plugin.name in str(capsys.readouterr())


def test_cli_shows_plugins_widgets(monkeypatch, capsys, tmp_plugin):
    """Test the cli --info runs and shows plugins"""
    monkeypatch.setattr(sys, 'argv', ['napari', '--plugin-info'])
    with pytest.raises(SystemExit):
        __main__._run()
    stdout = capsys.readouterr().out
    assert tmp_plugin.name in stdout
    assert "Display Name" in stdout


def test_cli_parses_unknowns(mock_run, monkeypatch, make_napari_viewer):
    """test that we can parse layer keyword arg variants"""
    v = make_napari_viewer()  # our mock view_path will return this object

    def assert_kwargs(*args, **kwargs):
        assert ["file"] in args
        assert kwargs['contrast_limits'] == (0, 1)

    # testing all the variants of literal_evals
    with mock.patch('napari.Viewer', return_value=v):
        monkeypatch.setattr(
            napari.components.viewer_model.ViewerModel, 'open', assert_kwargs
        )
        with monkeypatch.context() as m:
            m.setattr(
                sys, 'argv', ['n', 'file', '--contrast-limits', '(0, 1)']
            )
            __main__._run()
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits', '(0,1)'])
            __main__._run()
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits=(0, 1)'])
            __main__._run()
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits=(0,1)'])
            __main__._run()


def test_cli_raises(monkeypatch):
    """test that unknown kwargs raise the correct errors."""
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', 'path/to/file', '--nonsense'])
        with pytest.raises(SystemExit) as e:
            __main__._run()
        assert str(e.value) == 'error: unrecognized argument: --nonsense'

    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', 'path/to/file', '--gamma'])
        with pytest.raises(SystemExit) as e:
            __main__._run()
        assert str(e.value) == 'error: argument --gamma expected one argument'


@mock.patch('runpy.run_path')
def test_cli_runscript(run_path, monkeypatch, tmp_path):
    """Test that running napari script.py runs a script"""
    script = tmp_path / 'test.py'
    script.write_text('import napari; v = napari.Viewer(show=False)')

    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', str(script)])
        __main__._run()

    run_path.assert_called_once_with(str(script))


@mock.patch('napari._qt.qt_viewer.QtViewer._qt_open')
def test_cli_passes_kwargs(qt_open, mock_run, monkeypatch, make_napari_viewer):
    """test that we can parse layer keyword arg variants"""
    v = make_napari_viewer()

    with mock.patch('napari.Viewer', return_value=v):
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['n', 'file', '--name', 'some name'])
            __main__._run()

    qt_open.assert_called_once_with(
        ['file'],
        stack=[],
        plugin=None,
        layer_type=None,
        name='some name',
    )
    mock_run.assert_called_once_with(gui_exceptions=True)


@mock.patch('napari._qt.qt_viewer.QtViewer._qt_open')
def test_cli_passes_kwargs_stack(
    qt_open, mock_run, monkeypatch, make_napari_viewer
):
    """test that we can parse layer keyword arg variants"""
    v = make_napari_viewer()

    with mock.patch('napari.Viewer', return_value=v):
        with monkeypatch.context() as m:
            m.setattr(
                sys,
                'argv',
                [
                    'n',
                    'file',
                    '--stack',
                    'file1',
                    'file2',
                    '--stack',
                    'file3',
                    'file4',
                    '--name',
                    'some name',
                ],
            )
            __main__._run()

    qt_open.assert_called_once_with(
        ['file'],
        stack=[['file1', 'file2'], ['file3', 'file4']],
        plugin=None,
        layer_type=None,
        name='some name',
    )
    mock_run.assert_called_once_with(gui_exceptions=True)


def test_cli_retains_viewer_ref(mock_run, monkeypatch, make_napari_viewer):
    """Test that napari.__main__ is retaining a reference to the viewer."""
    v = make_napari_viewer()  # our mock view_path will return this object
    ref_count = None  # counter that will be updated before __main__._run()

    def _check_refs(**kwargs):
        # when run() is called in napari.__main__, we will call this function
        # it forces garbage collection, and then makes sure that at least one
        # additional reference to our viewer exists.
        gc.collect()
        if sys.getrefcount(v) <= ref_count:  # pragma: no cover
            raise AssertionError(
                "Reference to napari.viewer has been lost by "
                "the time the event loop started in napari.__main__"
            )

    mock_run.side_effect = _check_refs
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', 'path/to/file.tif'])
        # return our local v
        with mock.patch('napari.Viewer', return_value=v) as mock_viewer:
            ref_count = sys.getrefcount(v)  # count current references
            # mock gui open so we're not opening dialogs/throwing errors on fake path
            with mock.patch(
                'napari._qt.qt_viewer.QtViewer._qt_open', return_value=None
            ) as mock_viewer_open:
                __main__._run()
            mock_viewer.assert_called_once()
            mock_viewer_open.assert_called_once()
