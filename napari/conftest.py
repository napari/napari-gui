"""

Notes for using the plugin-related fixtures here:

1. The `npe2pm_` fixture is always used, and it mocks the global npe2 plugin
   manager instance with a discovery-deficient plugin manager.  No plugins should be
   discovered in tests without explicit registration.
2. wherever the builtins need to be tested, the `builtins` fixture should be explicitly
   added to the test.  (it's a DynamicPlugin that registers our builtins.yaml with the
   global mock npe2 plugin manager)
3. wherever *additional* plugins or contributions need to be added, use the `tmp_plugin`
   fixture, and add additional contributions _within_ the test (not in the fixture):
    ```python
    def test_something(tmp_plugin):
        @tmp_plugin.contribute.reader(filname_patterns=["*.ext"])
        def f(path): ...

        # the plugin name can be accessed at:
        tmp_plugin.name
    ```
4. If you need a _second_ mock plugin, use `tmp_plugin.spawn(register=True)` to create
   another one.
   ```python
   new_plugin = tmp_plugin.spawn(register=True)

   @new_plugin.contribute.reader(filename_patterns=["*.tiff"])
   def get_reader(path):
       ...
   ```
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
from itertools import chain
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from npe2 import PackageMetadata

with suppress(ModuleNotFoundError):
    __import__('dotenv').load_dotenv()

from datetime import timedelta
from time import perf_counter

import dask.threaded
import numpy as np
import pytest
from _pytest.pathlib import bestrelpath
from IPython.core.history import HistoryManager
from packaging.version import parse as parse_version
from pytest_pretty import CustomTerminalReporter

from napari.components import LayerList
from napari.layers import Image, Labels, Points, Shapes, Vectors
from napari.utils.misc import ROOT_DIR

if TYPE_CHECKING:
    from npe2._pytest_plugin import TestPluginManager

# touch ~/.Xauthority for Xlib support, must happen before importing pyautogui
if os.getenv('CI') and sys.platform.startswith('linux'):
    xauth = Path('~/.Xauthority').expanduser()
    if not xauth.exists():
        xauth.touch()


@pytest.fixture
def layer_data_and_types():
    """Fixture that provides some layers and filenames

    Returns
    -------
    tuple
        ``layers, layer_data, layer_types, filenames``

        - layers: some image and points layers
        - layer_data: same as above but in LayerData form
        - layer_types: list of strings with type of layer
        - filenames: the expected filenames with extensions for the layers.
    """
    layers = [
        Image(np.random.rand(20, 20), name='ex_img'),
        Image(np.random.rand(20, 20)),
        Points(np.random.rand(20, 2), name='ex_pts'),
        Points(
            np.random.rand(20, 2), properties={'values': np.random.rand(20)}
        ),
    ]
    extensions = ['.tif', '.tif', '.csv', '.csv']
    layer_data = [layer.as_layer_data_tuple() for layer in layers]
    layer_types = [layer._type_string for layer in layers]
    filenames = [
        layer.name + e for layer, e in zip(layers, extensions, strict=False)
    ]
    return layers, layer_data, layer_types, filenames


@pytest.fixture(
    params=[
        'image',
        'labels',
        'points',
        'shapes',
        'shapes-rectangles',
        'vectors',
    ]
)
def layer(request):
    """Parameterized fixture that supplies a layer for testing.

    Parameters
    ----------
    request : _pytest.fixtures.SubRequest
        The pytest request object

    Returns
    -------
    napari.layers.Layer
        The desired napari Layer.
    """
    np.random.seed(0)
    if request.param == 'image':
        data = np.random.rand(20, 20)
        return Image(data)
    if request.param == 'labels':
        data = np.random.randint(10, size=(20, 20))
        return Labels(data)
    if request.param == 'points':
        data = np.random.rand(20, 2)
        return Points(data)
    if request.param == 'shapes':
        data = [
            np.random.rand(2, 2),
            np.random.rand(2, 2),
            np.random.rand(6, 2),
            np.random.rand(6, 2),
            np.random.rand(2, 2),
        ]
        shape_type = ['ellipse', 'line', 'path', 'polygon', 'rectangle']
        return Shapes(data, shape_type=shape_type)
    if request.param == 'shapes-rectangles':
        data = np.random.rand(7, 4, 2)
        return Shapes(data)
    if request.param == 'vectors':
        data = np.random.rand(20, 2, 2)
        return Vectors(data)

    return None


@pytest.fixture
def layers():
    """Fixture that supplies a layers list for testing.

    Returns
    -------
    napari.components.LayerList
        The desired napari LayerList.
    """
    np.random.seed(0)
    list_of_layers = [
        Image(np.random.rand(20, 20)),
        Labels(np.random.randint(10, size=(20, 2))),
        Points(np.random.rand(20, 2)),
        Shapes(np.random.rand(10, 2, 2)),
        Vectors(np.random.rand(10, 2, 2)),
    ]
    return LayerList(list_of_layers)


@pytest.fixture(autouse=True)
def _skip_examples(request):
    """Skip examples test if ."""
    if request.node.get_closest_marker(
        'examples'
    ) and request.config.getoption('--skip_examples'):
        pytest.skip('running with --skip_examples')


# _PYTEST_RAISE=1 will prevent pytest from handling exceptions.
# Use with a debugger that's set to break on "unhandled exceptions".
# https://github.com/pytest-dev/pytest/issues/7409
if os.getenv('_PYTEST_RAISE', '0') != '0':

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    """This fixture ensures that default settings are used for every test.

    and ensures that changes to settings in a test are reverted, and never
    saved to disk.
    """
    from napari import settings
    from napari.settings import NapariSettings
    from napari.settings._experimental import ExperimentalSettings

    # prevent the developer's config file from being used if it exists
    cp = NapariSettings.__private_attributes__['_config_path']
    monkeypatch.setattr(cp, 'default', None)

    monkeypatch.setattr(
        ExperimentalSettings.__fields__['compiled_triangulation'],
        'default',
        True,
    )

    # calling save() with no config path is normally an error
    # here we just have save() return if called without a valid path
    NapariSettings.__original_save__ = NapariSettings.save

    def _mock_save(self, path=None, **dict_kwargs):
        if not (path or self.config_path):
            return
        NapariSettings.__original_save__(self, path, **dict_kwargs)

    monkeypatch.setattr(NapariSettings, 'save', _mock_save)

    settings._SETTINGS = None
    # this makes sure that we start with fresh settings for every test.
    return


@pytest.fixture(autouse=True)
def _auto_shutdown_dask_threadworkers():
    """
    This automatically shutdown dask thread workers.

    We don't assert the number of threads in unchanged as other things
    modify the number of threads.
    """
    assert dask.threaded.default_pool is None
    try:
        yield
    finally:
        if isinstance(dask.threaded.default_pool, ThreadPool):
            dask.threaded.default_pool.close()
            dask.threaded.default_pool.join()
        elif dask.threaded.default_pool:
            dask.threaded.default_pool.shutdown()
        dask.threaded.default_pool = None


# this is not the proper way to configure IPython, but it's an easy one.
# This will prevent IPython to try to write history on its sql file and do
# everything in memory.
# 1) it saves a thread and
# 2) it can prevent issues with slow or read-only file systems in CI.
HistoryManager.enabled = False


@pytest.fixture
def napari_svg_name():
    """the plugin name changes with npe2 to `napari-svg` from `svg`."""
    from importlib.metadata import version

    if parse_version(version('napari-svg')) < parse_version('0.1.6'):
        return 'svg'

    return 'napari-svg'


@pytest.fixture(autouse=True)
def npe2pm_(npe2pm, monkeypatch):
    """Autouse npe2 & npe1 mock plugin managers with no registered plugins."""
    from napari.plugins import NapariPluginManager

    monkeypatch.setattr(NapariPluginManager, 'discover', lambda *_, **__: None)
    return npe2pm


@pytest.fixture
def builtins(npe2pm_: TestPluginManager):
    with npe2pm_.tmp_plugin(package='napari') as plugin:
        yield plugin


@pytest.fixture
def tmp_plugin(npe2pm_: TestPluginManager):
    with npe2pm_.tmp_plugin() as plugin:
        plugin.manifest.package_metadata = PackageMetadata(  # type: ignore[call-arg]
            version='0.1.0', name='test'
        )
        plugin.manifest.display_name = 'Temp Plugin'
        yield plugin


@pytest.fixture
def viewer_model():
    from napari.components import ViewerModel

    return ViewerModel()


@pytest.fixture
def qt_viewer_(qtbot, viewer_model, monkeypatch):
    from napari._qt.qt_viewer import QtViewer

    viewer = QtViewer(viewer_model)

    original_controls = viewer.__class__.controls.fget
    original_layers = viewer.__class__.layers.fget
    original_layer_buttons = viewer.__class__.layerButtons.fget
    original_viewer_buttons = viewer.__class__.viewerButtons.fget
    original_dock_layer_list = viewer.__class__.dockLayerList.fget
    original_dock_layer_controls = viewer.__class__.dockLayerControls.fget
    original_dock_console = viewer.__class__.dockConsole.fget
    original_dock_performance = viewer.__class__.dockPerformance.fget

    def hide_widget(widget):
        widget.hide()

    def hide_and_clear_qt_viewer(viewer: QtViewer):
        viewer._instances.clear()
        viewer.hide()

    def patched_controls(self):
        if self._controls is None:
            self._controls = original_controls(self)
            qtbot.addWidget(self._controls, before_close_func=hide_widget)
        return self._controls

    def patched_layers(self):
        if self._layers is None:
            self._layers = original_layers(self)
            qtbot.addWidget(self._layers, before_close_func=hide_widget)
        return self._layers

    def patched_layer_buttons(self):
        if self._layersButtons is None:
            self._layersButtons = original_layer_buttons(self)
            qtbot.addWidget(self._layersButtons, before_close_func=hide_widget)
        return self._layersButtons

    def patched_viewer_buttons(self):
        if self._viewerButtons is None:
            self._viewerButtons = original_viewer_buttons(self)
            qtbot.addWidget(self._viewerButtons, before_close_func=hide_widget)
        return self._viewerButtons

    def patched_dock_layer_list(self):
        if self._dockLayerList is None:
            self._dockLayerList = original_dock_layer_list(self)
            qtbot.addWidget(self._dockLayerList, before_close_func=hide_widget)
        return self._dockLayerList

    def patched_dock_layer_controls(self):
        if self._dockLayerControls is None:
            self._dockLayerControls = original_dock_layer_controls(self)
            qtbot.addWidget(
                self._dockLayerControls, before_close_func=hide_widget
            )
        return self._dockLayerControls

    def patched_dock_console(self):
        if self._dockConsole is None:
            self._dockConsole = original_dock_console(self)
            qtbot.addWidget(self._dockConsole, before_close_func=hide_widget)
        return self._dockConsole

    def patched_dock_performance(self):
        if self._dockPerformance is None:
            self._dockPerformance = original_dock_performance(self)
            qtbot.addWidget(
                self._dockPerformance, before_close_func=hide_widget
            )
        return self._dockPerformance

    monkeypatch.setattr(
        viewer.__class__, 'controls', property(patched_controls)
    )
    monkeypatch.setattr(viewer.__class__, 'layers', property(patched_layers))
    monkeypatch.setattr(
        viewer.__class__, 'layerButtons', property(patched_layer_buttons)
    )
    monkeypatch.setattr(
        viewer.__class__, 'viewerButtons', property(patched_viewer_buttons)
    )
    monkeypatch.setattr(
        viewer.__class__, 'dockLayerList', property(patched_dock_layer_list)
    )
    monkeypatch.setattr(
        viewer.__class__,
        'dockLayerControls',
        property(patched_dock_layer_controls),
    )
    monkeypatch.setattr(
        viewer.__class__, 'dockConsole', property(patched_dock_console)
    )
    monkeypatch.setattr(
        viewer.__class__, 'dockPerformance', property(patched_dock_performance)
    )

    qtbot.addWidget(viewer, before_close_func=hide_and_clear_qt_viewer)
    return viewer


@pytest.fixture
def qt_viewer(qt_viewer_):
    """We created `qt_viewer_` fixture to allow modifying qt_viewer
    if module-level-specific modifications are necessary.
    For example, in `test_qt_viewer.py`.
    """
    return qt_viewer_


@pytest.fixture(autouse=True)
def _clear_cached_action_injection():
    """Automatically clear cached property `Action.injected`.

    Allows action manager actions to be injected using current provider/processors
    and dependencies. See #7219 for details.
    To be removed after ActionManager deprecation.
    """
    from napari.utils.action_manager import action_manager

    for action in action_manager._actions.values():
        if 'injected' in action.__dict__:
            del action.__dict__['injected']


def _event_check(instance):
    def _prepare_check(name, no_event_):
        def check(instance, no_event=no_event_):
            if name in no_event:
                assert not hasattr(
                    instance.events, name
                ), f'event {name} defined'
            else:
                assert hasattr(
                    instance.events, name
                ), f'event {name} not defined'

        return check

    no_event_set = set()
    if isinstance(instance, tuple):
        no_event_set = instance[1]
        instance = instance[0]

    for name, value in instance.__class__.__dict__.items():
        if isinstance(value, property) and name[0] != '_':
            yield _prepare_check(name, no_event_set), instance, name


def pytest_generate_tests(metafunc):
    """Generate separate test for each test toc check if all events are defined."""
    if 'event_define_check' in metafunc.fixturenames:
        res = []
        ids = []

        for obj in metafunc.cls.get_objects():
            for check, instance, name in _event_check(obj):
                res.append((check, instance))
                ids.append(f'{name}-{instance}')

        metafunc.parametrize('event_define_check,obj', res, ids=ids)


def pytest_collection_modifyitems(session, config, items):
    test_subset = os.environ.get('NAPARI_TEST_SUBSET')

    test_order_prefix = [
        os.path.join('napari', 'utils'),
        os.path.join('napari', 'layers'),
        os.path.join('napari', 'components'),
        os.path.join('napari', 'settings'),
        os.path.join('napari', 'plugins'),
        os.path.join('napari', '_vispy'),
        os.path.join('napari', '_qt'),
        os.path.join('napari', 'qt'),
        os.path.join('napari', '_tests'),
        os.path.join('napari', '_tests', 'test_examples.py'),
    ]
    test_order = [[] for _ in test_order_prefix]
    test_order.append([])  # for not matching tests
    for item in items:
        if test_subset:
            if test_subset.lower() == 'qt' and 'qapp' not in item.fixturenames:
                # Skip non Qt tests
                continue
            if (
                test_subset.lower() == 'headless'
                and 'qapp' in item.fixturenames
            ):
                # Skip Qt tests
                continue

        index = -1
        for i, prefix in enumerate(test_order_prefix):
            if prefix in str(item.fspath):
                index = i
        test_order[index].append(item)
    items[:] = list(chain(*test_order))


@pytest.fixture(autouse=True)
def _disable_notification_dismiss_timer(monkeypatch):
    """
    This fixture disables starting timer for closing notification
    by setting the value of `NapariQtNotification.DISMISS_AFTER` to 0.

    As Qt timer is realised by thread and keep reference to the object,
    without increase of reference counter object could be garbage collected and
    cause segmentation fault error when Qt (C++) code try to access it without
    checking if Python object exists.

    This fixture is used in all tests because it is possible to call Qt code
    from non Qt test by connection of `NapariQtNotification.show_notification` to
    `NotificationManager` global instance.
    """

    with suppress(ImportError):
        from napari._qt.dialogs.qt_notification import NapariQtNotification

        monkeypatch.setattr(NapariQtNotification, 'DISMISS_AFTER', 0)
        monkeypatch.setattr(NapariQtNotification, 'FADE_IN_RATE', 0)
        monkeypatch.setattr(NapariQtNotification, 'FADE_OUT_RATE', 0)


@pytest.fixture
def single_threaded_executor():
    executor = ThreadPoolExecutor(max_workers=1)
    yield executor
    executor.shutdown()


def _get_calling_stack():  # pragma: no cover
    stack = []
    for i in range(2, sys.getrecursionlimit()):
        try:
            frame = sys._getframe(i)
        except ValueError:
            break
        stack.append(f'{frame.f_code.co_filename}:{frame.f_lineno}')
    return '\n'.join(stack)


def _get_calling_place(depth=1):  # pragma: no cover
    if not hasattr(sys, '_getframe'):
        return ''
    frame = sys._getframe(1 + depth)
    result = f'{frame.f_code.co_filename}:{frame.f_lineno}'
    if not frame.f_code.co_filename.startswith(ROOT_DIR):
        with suppress(ValueError):
            while not frame.f_code.co_filename.startswith(ROOT_DIR):
                frame = frame.f_back
                if frame is None:
                    break
            else:
                result += f' called from\n{frame.f_code.co_filename}:{frame.f_lineno}'
    return result


@pytest.fixture
def _dangling_qthreads(monkeypatch, qtbot, request):
    from qtpy.QtCore import QThread

    base_start = QThread.start
    thread_dict = WeakKeyDictionary()
    base_constructor = QThread.__init__

    def run_with_trace(self):  # pragma: no cover
        """
        QThread.run but adding execution to sys.settrace when measuring coverage.

        See https://github.com/nedbat/coveragepy/issues/686#issuecomment-634932753
        and `init_with_trace`. When running QThreads during testing, we monkeypatch
        the QThread constructor and run methods with traceable equivalents.
        """
        if 'coverage' in sys.modules:
            # https://github.com/nedbat/coveragepy/issues/686#issuecomment-634932753
            sys.settrace(threading._trace_hook)
        self._base_run()

    def init_with_trace(self, *args, **kwargs):
        """Constructor for QThread adding tracing for coverage measurements.

        Functions running in QThreads don't get measured by coverage.py, see
        https://github.com/nedbat/coveragepy/issues/686. Therefore, we will
        monkeypatch the constructor to add to the thread to `sys.settrace` when
        we call `run` and `coverage` is in `sys.modules`.
        """
        base_constructor(self, *args, **kwargs)
        self._base_run = self.run
        self.run = partial(run_with_trace, self)

    # dict of threads that have been started but not yet terminated

    if 'disable_qthread_start' in request.keywords:

        def start_with_save_reference(self, priority=QThread.InheritPriority):
            """Dummy function to prevent thread starts."""

    else:

        def start_with_save_reference(self, priority=QThread.InheritPriority):
            """Thread start function with logs to detect hanging threads.

            Saves a weak reference to the thread and detects hanging threads,
            as well as where the threads were started.
            """
            thread_dict[self] = _get_calling_place()
            base_start(self, priority)

    monkeypatch.setattr(QThread, 'start', start_with_save_reference)
    monkeypatch.setattr(QThread, '__init__', init_with_trace)

    yield

    dangling_threads_li = []

    for thread, calling in thread_dict.items():
        try:
            if thread.isRunning():
                dangling_threads_li.append((thread, calling))
        except RuntimeError as e:
            if (
                'wrapped C/C++ object of type' not in e.args[0]
                and 'Internal C++ object' not in e.args[0]
            ):
                raise

    for thread, _ in dangling_threads_li:
        with suppress(RuntimeError):
            thread.quit()
            qtbot.waitUntil(thread.isFinished, timeout=2000)

    long_desc = (
        'If you see this error, it means that a QThread was started in a test '
        'but not terminated. This can cause segfaults in the test suite. '
        'Please use the `qtbot` fixture to wait for the thread to finish. '
        'If you think that the thread is obsolete for this test, you can '
        'use the `@pytest.mark.disable_qthread_start` mark or  `monkeypatch` '
        'fixture to patch the `start` method of the '
        'QThread class to do nothing.\n'
    )

    if len(dangling_threads_li) > 1:
        long_desc += ' The QThreads were started in:\n'
    else:
        long_desc += ' The QThread was started in:\n'

    assert not dangling_threads_li, long_desc + '\n'.join(
        x[1] for x in dangling_threads_li
    )


@pytest.fixture
def _dangling_qthread_pool(monkeypatch, request):
    from qtpy.QtCore import QThreadPool

    base_start = QThreadPool.start
    threadpool_dict = WeakKeyDictionary()
    # dict of threadpools that have been used to run QRunnables

    if 'disable_qthread_pool_start' in request.keywords:

        def my_start(self, runnable, priority=0):
            """dummy function to prevent thread start"""

    else:

        def my_start(self, runnable, priority=0):
            if self not in threadpool_dict:
                threadpool_dict[self] = []
            threadpool_dict[self].append(_get_calling_place())
            base_start(self, runnable, priority)

    monkeypatch.setattr(QThreadPool, 'start', my_start)
    yield

    dangling_threads_pools = []

    for thread_pool, calling in threadpool_dict.items():
        thread_pool.clear()
        thread_pool.waitForDone(20)
        if thread_pool.activeThreadCount():
            dangling_threads_pools.append((thread_pool, calling))

    for thread_pool, _ in dangling_threads_pools:
        with suppress(RuntimeError):
            thread_pool.clear()
            thread_pool.waitForDone(2000)

    long_desc = (
        'If you see this error, it means that a QThreadPool was used to run '
        'a QRunnable in a test but not terminated. This can cause segfaults '
        'in the test suite. Please use the `qtbot` fixture to wait for the '
        'thread to finish. If you think that the thread is obsolete for this '
        'use the `@pytest.mark.disable_qthread_pool_start` mark or  `monkeypatch` '
        'fixture to patch the `start` '
        'method of the QThreadPool class to do nothing.\n'
    )
    if len(dangling_threads_pools) > 1:
        long_desc += ' The QThreadPools were used in:\n'
    else:
        long_desc += ' The QThreadPool was used in:\n'

    assert not dangling_threads_pools, long_desc + '\n'.join(
        '; '.join(x[1]) for x in dangling_threads_pools
    )


@pytest.fixture
def _dangling_qtimers(monkeypatch, request):
    from qtpy.QtCore import QTimer

    base_start = QTimer.start
    timer_dkt = WeakKeyDictionary()
    single_shot_list = []

    if 'disable_qtimer_start' in request.keywords:
        from pytestqt.qt_compat import qt_api

        def my_start(self, msec=None):
            """dummy function to prevent timer start"""

        _single_shot = my_start

        class OldTimer(QTimer):
            def start(self, time=None):
                if time is not None:
                    base_start(self, time)
                else:
                    base_start(self)

        monkeypatch.setattr(qt_api.QtCore, 'QTimer', OldTimer)
        # This monkeypatch is require to keep `qtbot.waitUntil` working

    else:

        def my_start(self, msec=None):
            calling_place = _get_calling_place()
            if 'superqt' in calling_place and 'throttler' in calling_place:
                calling_place += f' - {_get_calling_place(2)}'
            timer_dkt[self] = calling_place
            if msec is not None:
                base_start(self, msec)
            else:
                base_start(self)

        def single_shot(msec, reciver, method=None):
            t = QTimer()
            t.setSingleShot(True)
            if method is None:
                t.timeout.connect(reciver)
            else:
                t.timeout.connect(getattr(reciver, method))
            calling_place = _get_calling_place(2)
            if 'superqt' in calling_place and 'throttler' in calling_place:
                calling_place += _get_calling_stack()
            single_shot_list.append((t, _get_calling_place(2)))
            base_start(t, msec)

        def _single_shot(self, *args):
            if isinstance(self, QTimer):
                single_shot(*args)
            else:
                single_shot(self, *args)

    monkeypatch.setattr(QTimer, 'start', my_start)
    monkeypatch.setattr(QTimer, 'singleShot', _single_shot)

    yield

    dangling_timers = []

    for timer, calling in chain(timer_dkt.items(), single_shot_list):
        if timer.isActive():
            dangling_timers.append((timer, calling))

    for timer, _ in dangling_timers:
        with suppress(RuntimeError):
            timer.stop()

    long_desc = (
        'If you see this error, it means that a QTimer was started but not stopped. '
        'This can cause tests to fail, and can also cause segfaults. '
        'If this test does not require a QTimer to pass you could monkeypatch it out. '
        'If it does require a QTimer, you should stop or wait for it to finish before test ends. '
    )
    if len(dangling_timers) > 1:
        long_desc += 'The QTimers were started in:\n'
    else:
        long_desc += 'The QTimer was started in:\n'

    def _check_throttle_info(path):
        if 'superqt' in path and 'throttler' in path:
            return (
                path
                + " it's possible that there was a problem with unfinished work by a "
                'qthrottler; to solve this, you can either try to wait (such as with '
                '`qtbot.wait`) or disable throttling with the disable_throttling fixture'
            )
        return path

    assert not dangling_timers, long_desc + '\n'.join(
        _check_throttle_info(x[1]) for x in dangling_timers
    )


def _throttle_mock(self):
    self.triggered.emit()


def _flush_mock(self):
    """There are no waiting events."""


@pytest.fixture
def _disable_throttling(monkeypatch):
    """Disable qthrottler from superqt.

    This is sometimes necessary to avoid flaky failures in tests
    due to dangling qt timers.
    """
    # if this monkeypath fails then you should update path to GenericSignalThrottler
    monkeypatch.setattr(
        'superqt.utils._throttler.GenericSignalThrottler.throttle',
        _throttle_mock,
    )
    monkeypatch.setattr(
        'superqt.utils._throttler.GenericSignalThrottler.flush', _flush_mock
    )


@pytest.fixture
def _dangling_qanimations(monkeypatch, request):
    from qtpy.QtCore import QPropertyAnimation

    base_start = QPropertyAnimation.start
    animation_dkt = WeakKeyDictionary()

    if 'disable_qanimation_start' in request.keywords:

        def my_start(self):
            """dummy function to prevent thread start"""

    else:

        def my_start(self):
            animation_dkt[self] = _get_calling_place()
            base_start(self)

    monkeypatch.setattr(QPropertyAnimation, 'start', my_start)
    yield

    dangling_animations = []

    for animation, calling in animation_dkt.items():
        if animation.state() == QPropertyAnimation.Running:
            dangling_animations.append((animation, calling))

    for animation, _ in dangling_animations:
        with suppress(RuntimeError):
            animation.stop()

    long_desc = (
        'If you see this error, it means that a QPropertyAnimation was started but not stopped. '
        'This can cause tests to fail, and can also cause segfaults. '
        'If this test does not require a QPropertyAnimation to pass you could monkeypatch it out. '
        'If it does require a QPropertyAnimation, you should stop or wait for it to finish before test ends. '
    )
    if len(dangling_animations) > 1:
        long_desc += ' The QPropertyAnimations were started in:\n'
    else:
        long_desc += ' The QPropertyAnimation was started in:\n'
    assert not dangling_animations, long_desc + '\n'.join(
        x[1] for x in dangling_animations
    )


with contextlib.suppress(ImportError):
    # in headless test suite we don't have Qt bindings
    # So we cannot inherit from QtBot and declare the fixture

    from pytestqt.qtbot import QtBot

    class QtBotWithOnCloseRenaming(QtBot):
        """Modified QtBot that renames widgets when closing them in tests.

        After a test ends that uses QtBot, all instantiated widgets added to
        the bot have their name changed to 'handled_widget'. This allows us to
        detect leaking widgets at the end of a test run, and avoid the
        segmentation faults that often result from such leaks. [1]_

        See Also
        --------
        `_find_dangling_widgets`: fixture that finds all widgets that have not
        been renamed to 'handled_widget'.

        References
        ----------
        .. [1] https://czaki.github.io/blog/2024/09/16/preventing-segfaults-in-test-suite-that-has-qt-tests/
        """

        def addWidget(self, widget, *, before_close_func=None):
            if widget.objectName() == '':
                # object does not have a name, so we can set it
                widget.setObjectName('handled_widget')
                before_close_func_ = before_close_func
            elif before_close_func is None:
                # there is no custom teardown function,
                # so we provide one that will set object name

                def before_close_func_(w):
                    w.setObjectName('handled_widget')
            else:
                # user provided custom teardown function,
                # so we need to wrap it to set object name

                def before_close_func_(w):
                    before_close_func(w)
                    w.setObjectName('handled_widget')

            super().addWidget(widget, before_close_func=before_close_func_)

    @pytest.fixture
    def qtbot(qapp, request):  # pragma: no cover
        """Fixture to create a QtBotWithOnCloseRenaming instance for testing.

        Make sure to call addWidget for each top-level widget you create to
        ensure that they are properly closed after the test ends.

        The `qapp` fixture is used to ensure that the QApplication is created
        before, so we need it, even without using it directly in this fixture.
        """
        return QtBotWithOnCloseRenaming(request)


@pytest.fixture
def _find_dangling_widgets(request, qtbot):
    yield

    from qtpy.QtWidgets import QApplication

    from napari._qt.qt_main_window import _QtMainWindow

    top_level_widgets = QApplication.topLevelWidgets()

    viewer_weak_set = getattr(request.node, '_viewer_weak_set', set())

    problematic_widgets = []

    for widget in top_level_widgets:
        if widget.parent() is not None:
            continue
        if (
            isinstance(widget, _QtMainWindow)
            and widget._qt_viewer.viewer in viewer_weak_set
        ):
            continue

        if widget.__class__.__module__.startswith('qtconsole'):
            continue

        if widget.objectName() == 'handled_widget':
            continue

        if widget.__class__.__name__ == 'CanvasBackendDesktop':
            # TODO: we don't understand why this class leaks in
            #  napari/_tests/test_sys_info.py, so we make an exception
            #  here and we don't raise when this class leaks.
            continue

        problematic_widgets.append(widget)

    if problematic_widgets:
        text = '\n'.join(
            f'Widget: {widget} of type {type(widget)} with name {widget.objectName()}'
            for widget in problematic_widgets
        )

        for widget in problematic_widgets:
            widget.setObjectName('handled_widget')

        raise RuntimeError(f'Found dangling widgets:\n{text}')


def pytest_runtest_setup(item):
    """Add Qt leak detection fixtures *only* in tests using the qapp fixture.

    Because we have headless test suite that does not include Qt, we cannot
    simply use `@pytest.fixture(autouse=True)` on all our fixtures for
    detecting leaking Qt objects.

    Instead, here we detect whether the `qapp` fixture is being used, detecting
    tests that use Qt and need to be checked for Qt objects leaks.

    A note to maintainers: tests *may* attempt to use Qt classes but not use
    the `qapp` fixture. This is BAD, and may cause Qt failures to be reported
    far away from the problematic code or test. If you find any tests
    instantiating Qt objects but not using qapp or qtbot, please submit a PR
    adding the qtbot fixture and adding any top-level Qt widgets with::

        qtbot.addWidget(widget_instance)

    """

    if 'qapp' in item.fixturenames:
        # here we do autouse for dangling fixtures only if qapp is used
        if 'qtbot' not in item.fixturenames:
            # for proper waiting for threads to finish
            item.fixturenames.append('qtbot')
        item.fixturenames.extend(
            [
                '_find_dangling_widgets',
                '_dangling_qthread_pool',
                '_dangling_qanimations',
                '_dangling_qthreads',
                '_dangling_qtimers',
            ]
        )


class NapariTerminalReporter(CustomTerminalReporter):
    """
    This ia s custom terminal reporter to how long it takes to finish given part of tests.
    It prints time each time when test from different file is started.

    It is created to be able to see if timeout is caused by long time execution, or it is just hanging.
    """

    currentfspath: Path | None

    def write_fspath_result(self, nodeid: str, res, **markup: bool) -> None:
        if getattr(self, '_start_time', None) is None:
            self._start_time = perf_counter()
        fspath = self.config.rootpath / nodeid.split('::')[0]
        if self.currentfspath is None or fspath != self.currentfspath:
            if self.currentfspath is not None and self._show_progress_info:
                self._write_progress_information_filling_space()
                if os.environ.get('CI', False):
                    self.write(
                        f' [{timedelta(seconds=int(perf_counter() - self._start_time))}]'
                    )
            self.currentfspath = fspath
            relfspath = bestrelpath(self.startpath, fspath)
            self._tw.line()
            self.write(relfspath + ' ')
        self.write(res, flush=True, **markup)


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # Get the standard terminal reporter plugin and replace it with our
    standard_reporter = config.pluginmanager.getplugin('terminalreporter')
    custom_reporter = NapariTerminalReporter(config, sys.stdout)
    if standard_reporter._session is not None:
        custom_reporter._session = standard_reporter._session
    config.pluginmanager.unregister(standard_reporter)
    config.pluginmanager.register(custom_reporter, 'terminalreporter')
