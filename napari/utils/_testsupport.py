import gc
import os
import sys
import warnings
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch
from weakref import WeakSet

import pytest

if TYPE_CHECKING:
    from pytest import FixtureRequest  # noqa: PT013

_SAVE_GRAPH_OPNAME = '--save-leaked-object-graph'


def _empty(*_, **__):
    """Empty function for mocking"""


def pytest_addoption(parser):
    parser.addoption(
        '--show-napari-viewer',
        action='store_true',
        default=False,
        help="don't show viewer during tests",
    )

    parser.addoption(
        _SAVE_GRAPH_OPNAME,
        action='store_true',
        default=False,
        help="Try to save a graph of leaked object's reference (need objgraph"
        'and graphviz installed',
    )


COUNTER = 0


def fail_obj_graph(Klass):  # pragma: no cover
    """
    Fail is a given class _instances weakset is non empty and print the object graph.
    """

    try:
        import objgraph
    except ModuleNotFoundError:
        return

    if len(Klass._instances) != 0:
        global COUNTER
        COUNTER += 1
        import gc

        leaked_objects_count = len(Klass._instances)

        gc.collect()
        file_path = Path(
            f'{Klass.__name__}-leak-backref-graph-{COUNTER}.pdf'
        ).absolute()
        objgraph.show_backrefs(
            list(Klass._instances),
            max_depth=20,
            filename=str(file_path),
        )

        Klass._instances.clear()

        assert file_path.exists()

        # DO not remove len, this can break as C++ obj are gone, but python objects
        # still hang around and _repr_ would crash.
        pytest.fail(
            f'Test run fail with leaked {leaked_objects_count} instances of {Klass}.'
            f'The object graph is saved in {file_path}.'
            f'{len(Klass._instances)} objects left after cleanup'
        )


@pytest.fixture
def napari_plugin_manager(monkeypatch):
    """A napari plugin manager that blocks discovery by default.

    Note you can still use `napari_plugin_manager.register()` to directly
    register a plugin module, class or dict for testing.

    Or, to re-enable global discovery, use:
    `napari_plugin_manager.discovery_blocker.stop()`
    """
    import napari
    import napari.plugins.io
    from napari.plugins._plugin_manager import NapariPluginManager

    pm = NapariPluginManager()

    # make it so that internal requests for the plugin_manager
    # get this test version for the duration of the test.
    monkeypatch.setattr(napari.plugins, 'plugin_manager', pm)
    monkeypatch.setattr(napari.plugins.io, 'plugin_manager', pm)
    with suppress(AttributeError):
        monkeypatch.setattr(napari._qt.qt_main_window, 'plugin_manager', pm)
    # prevent discovery of plugins in the environment
    # you can still use `pm.register` to explicitly register something.
    pm.discovery_blocker = patch.object(pm, 'discover')
    pm.discovery_blocker.start()
    pm._initialize()  # register our builtins
    yield pm
    pm.discovery_blocker.stop()


GCPASS = 0


@pytest.fixture(autouse=True)
def _clean_themes():
    from napari.utils import theme

    themes = set(theme.available_themes())
    yield
    for name in theme.available_themes():
        if name not in themes:
            del theme._themes[name]


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # https://docs.pytest.org/en/latest/example/simple.html#making-test-result-information-available-in-fixtures
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # set a report attribute for each phase of a call, which can
    # be "setup", "call", "teardown"

    setattr(item, f'rep_{rep.when}', rep)


@pytest.fixture
def mock_app_model():
    """Mock clean 'test_app' `NapariApplication` instance.

    This fixture must be used whenever `napari._app_model.get_app_model()` is called to
    return a 'test_app' `NapariApplication` instead of the 'napari'
    `NapariApplication`. The `make_napari_viewer` fixture is already equipped with
    a `mock_app_model`.

    Note that `NapariApplication` registers app-model actions.
    If this is not desired, please create a clean
    `app_model.Application` in the test.

    It does not register Qt related actions, providers or processors, which is done
    via `init_qactions()`. Nor does it register plugins, done via `_initialize_plugins`.
    If these are required, the `make_napari_viewer` fixture can be used, which
    will register ALL actions, providers and processors and register plugins.
    It will also automatically clear the lru cache.

    Alternatively, you can specifically run `init_qactions()` or
    `_initialize_plugins` within the test, ensuring that you `cache_clear()`
    first.
    """
    from app_model import Application

    from napari._app_model._app import NapariApplication, _napari_names

    app = NapariApplication('test_app')
    app.injection_store.namespace = _napari_names
    with patch.object(NapariApplication, 'get_app_model', return_value=app):
        try:
            yield app
        finally:
            Application.destroy('test_app')


@pytest.fixture
def make_napari_viewer(
    qtbot,
    request: 'FixtureRequest',
    mock_app_model,
    napari_plugin_manager,
    monkeypatch,
):
    """A pytest fixture function that creates a napari viewer for use in testing.

    This fixture will take care of creating a viewer and cleaning up at the end of the
    test. When using this function, it is **not** necessary to use a `qtbot` fixture,
    nor should you do any additional cleanup (such as using `qtbot.addWidget` or
    calling `viewer.close()`) at the end of the test. Duplicate cleanup may cause
    an error.

    To use this fixture as a function in your tests:

        def test_something_with_a_viewer(make_napari_viewer):
            # `make_napari_viewer` takes any keyword arguments that napari.Viewer() takes
            viewer = make_napari_viewer()

    It accepts all the same arguments as `napari.Viewer`, notably `show`
    which should be set to `True` for tests that require the `Viewer` to be visible
    (e.g., tests that check aspects of the Qt window or layer rendering).
    It also accepts the following test-related parameters:

    ViewerClass : Type[napari.Viewer], optional
        Override the viewer class being used.  By default, will
        use napari.viewer.Viewer
    strict_qt : bool or str, optional
        If True, a check will be performed after test cleanup to make sure that
        no top level widgets were created and *not* cleaned up during the
        test.  If the string "raise" is provided, an AssertionError will be
        raised.  Otherwise a warning is emitted.
        By default, this is False unless the test is being performed within
        the napari package.
        This can be made globally true by setting the 'NAPARI_STRICT_QT'
        environment variable.
    block_plugin_discovery : bool, optional
        Block discovery of non-builtin plugins.  Note: plugins can still be
        manually registered by using the 'napari_plugin_manager' fixture and
        the `napari_plugin_manager.register()` method. By default, True.

    Examples
    --------
    >>> def test_adding_shapes(make_napari_viewer):
    ...     viewer = make_napari_viewer()
    ...     viewer.add_shapes()
    ...     assert len(viewer.layers) == 1

    >>> def test_something_with_plugins(make_napari_viewer):
    ...     viewer = make_napari_viewer(block_plugin_discovery=False)

    >>> def test_something_with_strict_qt_tests(make_napari_viewer):
    ...     viewer = make_napari_viewer(strict_qt=True)
    """
    from qtpy.QtWidgets import QApplication, QWidget

    from napari import Viewer
    from napari._qt._qapp_model.qactions import init_qactions
    from napari._qt.qt_viewer import QtViewer
    from napari.plugins import _initialize_plugins
    from napari.settings import get_settings

    global GCPASS
    GCPASS += 1

    if GCPASS % 50 == 0:
        gc.collect()
    else:
        gc.collect(1)

    _do_not_inline_below = len(QtViewer._instances)
    # # do not inline to avoid pytest trying to compute repr of expression.
    # # it fails if C++ object gone but not Python object.
    if request.config.getoption(_SAVE_GRAPH_OPNAME):
        fail_obj_graph(QtViewer)
    QtViewer._instances.clear()
    assert _do_not_inline_below == 0, (
        'Some instance of QtViewer is not properly cleaned in one of previous test. For easier debug one may '
        f'use {_SAVE_GRAPH_OPNAME} flag for pytest to get graph of leaked objects. If you use qtbot (from pytest-qt)'
        ' to clean Qt objects after test you may need to switch to manual clean using '
        '`deleteLater()` and `qtbot.wait(50)` later.'
    )

    settings = get_settings()
    settings.reset()

    _initialize_plugins.cache_clear()
    init_qactions.cache_clear()

    viewers: WeakSet[Viewer] = WeakSet()
    request.node._viewer_weak_set = viewers

    # may be overridden by using the parameter `strict_qt`
    _strict = False

    initial = QApplication.topLevelWidgets()
    prior_exception = getattr(sys, 'last_value', None)
    is_internal_test = request.module.__name__.startswith('napari.')

    # disable thread for status checker
    monkeypatch.setattr(
        'napari._qt.threads.status_checker.StatusChecker.start',
        _empty,
    )
    monkeypatch.setattr(
        'napari._qt.qt_main_window._QtMainWindow._save_current_window_settings',
        _empty,
    )

    if 'enable_console' not in request.keywords:

        def _dummy_widget(*_):
            w = QWidget()
            w._update_theme = _empty
            return w

        monkeypatch.setattr(
            'napari._qt.qt_viewer.QtViewer._get_console', _dummy_widget
        )

    def actual_factory(
        *model_args,
        ViewerClass=Viewer,
        strict_qt=None,
        block_plugin_discovery=True,
        **model_kwargs,
    ):
        if strict_qt is None:
            strict_qt = is_internal_test or os.getenv('NAPARI_STRICT_QT')
        nonlocal _strict
        _strict = strict_qt

        if not block_plugin_discovery:
            napari_plugin_manager.discovery_blocker.stop()

        should_show = request.config.getoption('--show-napari-viewer')
        model_kwargs['show'] = model_kwargs.pop('show', should_show)
        viewer = ViewerClass(*model_args, **model_kwargs)
        viewers.add(viewer)

        return viewer

    yield actual_factory

    # Some tests might have the viewer closed, so this call will not be able
    # to access the window.
    with suppress(AttributeError):
        get_settings().reset()

    # close viewers, but don't saving window settings while closing
    for viewer in viewers:
        viewer.close()

    if GCPASS % 50 == 0 or len(QtViewer._instances):
        gc.collect()
    else:
        gc.collect(1)

    if request.config.getoption(_SAVE_GRAPH_OPNAME):
        fail_obj_graph(QtViewer)

    if request.node.rep_call.failed:
        # IF test failed do not check for leaks
        QtViewer._instances.clear()

    _do_not_inline_below = len(QtViewer._instances)

    QtViewer._instances.clear()  # clear to prevent fail of next test

    # do not inline to avoid pytest trying to compute repr of expression.
    # it fails if C++ object gone but not Python object.
    assert (
        _do_not_inline_below == 0
    ), f'{request.config.getoption(_SAVE_GRAPH_OPNAME)}, {_SAVE_GRAPH_OPNAME}'

    # only check for leaked widgets if an exception was raised during the test,
    # and "strict" mode was used.
    if _strict and getattr(sys, 'last_value', None) is prior_exception:
        QApplication.processEvents()
        leak = set(QApplication.topLevelWidgets()).difference(initial)
        leak = (x for x in leak if x.objectName() != 'handled_widget')
        # still not sure how to clean up some of the remaining vispy
        # vispy.app.backends._qt.CanvasBackendDesktop widgets...
        # observed in `test_sys_info.py`
        if any(n.__class__.__name__ != 'CanvasBackendDesktop' for n in leak):
            # just a warning... but this can be converted to test errors
            # in pytest with `-W error`
            msg = f"""The following Widgets leaked!: {leak}.

            Note: If other tests are failing it is likely that widgets will leak
            as they will be (indirectly) attached to the tracebacks of previous failures.
            Please only consider this an error if all other tests are passing.
            """
            # Explanation notes on the above: While we are indeed looking at the
            # difference in sets of widgets between before and after, new object can
            # still not be garbage collected because of it.
            # in particular with VisPyCanvas, it looks like if a traceback keeps
            # contains the type, then instances are still attached to the type.
            # I'm not too sure why this is the case though.
            if _strict == 'raise':
                raise AssertionError(msg)
            else:
                warnings.warn(msg)


@pytest.fixture
def make_napari_viewer_proxy(make_napari_viewer, monkeypatch):
    """Fixture that returns a function for creating a napari viewer wrapped in proxy.
    Use in the same way like `make_napari_viewer` fixture.

    Parameters
    ----------
    make_napari_viewer : fixture
        The make_napari_viewer fixture.

    Returns
    -------
    function
        A function that creates a napari viewer.
    """
    from napari.utils._proxies import PublicOnlyProxy

    proxies = []

    def actual_factory(*model_args, ensure_main_thread=True, **model_kwargs):
        monkeypatch.setenv(
            'NAPARI_ENSURE_PLUGIN_MAIN_THREAD', str(ensure_main_thread)
        )
        viewer = make_napari_viewer(*model_args, **model_kwargs)
        proxies.append(PublicOnlyProxy(viewer))
        return proxies[-1]

    proxies.clear()

    return actual_factory


@pytest.fixture
def MouseEvent():
    """Create a subclass for simulating vispy mouse events.

    Returns
    -------
    Event : Type
        A new dataclass named Event that can be used to create an
        object with fields "type" and "is_dragging".
    """

    @dataclass
    class Event:
        type: str
        position: tuple[float]
        is_dragging: bool = False
        dims_displayed: tuple[int] = (0, 1)
        dims_point: list[float] = None
        view_direction: list[int] = None
        pos: list[int] = (0, 0)
        button: int = None
        handled: bool = False

    return Event
