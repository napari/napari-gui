"""

Notes for using the plugin-related fixtures here:

1. The `_mock_npe2_pm` fixture is always used, and it mocks the global npe2 plugin
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

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from itertools import chain
from multiprocessing.pool import ThreadPool
from typing import TYPE_CHECKING
from unittest.mock import patch
from weakref import WeakKeyDictionary

with suppress(ModuleNotFoundError):
    __import__('dotenv').load_dotenv()

import dask.threaded
import numpy as np
import pytest
from IPython.core.history import HistoryManager

from napari.components import LayerList
from napari.layers import Image, Labels, Points, Shapes, Vectors
from napari.layers.layergroup import LayerGroup
from napari.utils.config import async_loading
from napari.utils.misc import ROOT_DIR

if TYPE_CHECKING:
    from npe2._pytest_plugin import TestPluginManager


def pytest_addoption(parser):
    """Add napari specific command line options.

    --aysnc_only
        Run only asynchronous tests, not sync ones.

    Notes
    -----
    Due to the placement of this conftest.py file, you must specifically name
    the napari folder such as "pytest napari --aysnc_only"
    """

    parser.addoption(
        "--async_only",
        action="store_true",
        default=False,
        help="run only asynchronous tests",
    )


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
    filenames = [layer.name + e for layer, e in zip(layers, extensions)]
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


@pytest.fixture(params=[LayerList, LayerGroup])
def layers(request):
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
    return request.param(list_of_layers)


# Currently we cannot run async and async in the invocation of pytest
# because we get a segfault for unknown reasons. So for now:
# "pytest" runs sync_only
# "pytest napari --async_only" runs async only
@pytest.fixture(scope="session", autouse=True)
def configure_loading(request):
    """Configure async/async loading."""
    if request.config.getoption("--async_only"):
        # Late import so we don't import experimental code unless using it.
        from napari.components.experimental.chunk import synchronous_loading

        with synchronous_loading(False):
            yield
    else:
        yield  # Sync so do nothing.


def _is_async_mode() -> bool:
    """Return True if we are currently loading chunks asynchronously

    Returns
    -------
    bool
        True if we are currently loading chunks asynchronously.
    """
    if not async_loading:
        return False  # Not enabled at all.

    # Late import so we don't import experimental code unless using it.
    from napari.components.experimental.chunk import chunk_loader

    return not chunk_loader.force_synchronous


@pytest.fixture(autouse=True)
def skip_sync_only(request):
    """Skip async_only tests if running async."""
    sync_only = request.node.get_closest_marker('sync_only')
    if _is_async_mode() and sync_only:
        pytest.skip("running with --async_only")


@pytest.fixture(autouse=True)
def skip_async_only(request):
    """Skip async_only tests if running sync."""
    async_only = request.node.get_closest_marker('async_only')
    if not _is_async_mode() and async_only:
        pytest.skip("not running with --async_only")


@pytest.fixture(autouse=True)
def skip_examples(request):
    """Skip examples test if ."""
    if request.node.get_closest_marker(
        'examples'
    ) and request.config.getoption("--skip_examples"):
        pytest.skip("running with --skip_examples")


# _PYTEST_RAISE=1 will prevent pytest from handling exceptions.
# Use with a debugger that's set to break on "unhandled exceptions".
# https://github.com/pytest-dev/pytest/issues/7409
if os.getenv('_PYTEST_RAISE', "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch):
    """This fixture ensures that default settings are used for every test.

    and ensures that changes to settings in a test are reverted, and never
    saved to disk.
    """
    from napari import settings
    from napari.settings import NapariSettings

    # prevent the developer's config file from being used if it exists
    cp = NapariSettings.__private_attributes__['_config_path']
    monkeypatch.setattr(cp, 'default', None)

    # calling save() with no config path is normally an error
    # here we just have save() return if called without a valid path
    NapariSettings.__original_save__ = NapariSettings.save

    def _mock_save(self, path=None, **dict_kwargs):
        if not (path or self.config_path):
            return
        NapariSettings.__original_save__(self, path, **dict_kwargs)

    monkeypatch.setattr(NapariSettings, 'save', _mock_save)

    # this makes sure that we start with fresh settings for every test.
    settings._SETTINGS = None
    yield


@pytest.fixture(autouse=True)
def auto_shutdown_dask_threadworkers():
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
    from importlib.metadata import metadata

    if tuple(metadata('napari-svg')['Version'].split('.')) < ('0', '1', '6'):
        return 'svg'

    return 'napari-svg'


@pytest.fixture(autouse=True, scope='session')
def _no_error_reports():
    """Turn off napari_error_reporter if it's installed."""
    try:
        p1 = patch('napari_error_reporter.capture_exception')
        p2 = patch('napari_error_reporter.install_error_reporter')
        with p1, p2:
            yield
    except (ModuleNotFoundError, AttributeError):
        yield


@pytest.fixture(autouse=True)
def _npe2pm(npe2pm, monkeypatch):
    """Autouse the npe2 mock plugin manager with no registered plugins."""
    from napari.plugins import NapariPluginManager

    monkeypatch.setattr(NapariPluginManager, 'discover', lambda *_, **__: None)
    return npe2pm


@pytest.fixture
def builtins(_npe2pm: TestPluginManager):
    with _npe2pm.tmp_plugin(package='napari') as plugin:
        yield plugin


@pytest.fixture
def tmp_plugin(_npe2pm: TestPluginManager):
    with _npe2pm.tmp_plugin() as plugin:
        plugin.manifest.package_metadata = {'version': '0.1.0', 'name': 'test'}
        plugin.manifest.display_name = 'Temp Plugin'
        yield plugin


def _event_check(instance):
    def _prepare_check(name, no_event_):
        def check(instance, no_event=no_event_):
            if name in no_event:
                assert not hasattr(
                    instance.events, name
                ), f"event {name} defined"
            else:
                assert hasattr(
                    instance.events, name
                ), f"event {name} not defined"

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
                ids.append(f"{name}-{instance}")

        metafunc.parametrize('event_define_check,obj', res, ids=ids)


def pytest_collection_modifyitems(session, config, items):
    test_order_prefix = [
        os.path.join("napari", "utils"),
        os.path.join("napari", "layers"),
        os.path.join("napari", "components"),
        os.path.join("napari", "settings"),
        os.path.join("napari", "plugins"),
        os.path.join("napari", "_vispy"),
        os.path.join("napari", "_qt"),
        os.path.join("napari", "qt"),
        os.path.join("napari", "_tests"),
        os.path.join("napari", "_tests", "test_examples.py"),
    ]
    test_order = [[] for _ in test_order_prefix]
    test_order.append([])  # for not matching tests
    for item in items:
        index = -1
        for i, prefix in enumerate(test_order_prefix):
            if prefix in str(item.fspath):
                index = i
        test_order[index].append(item)
    items[:] = list(chain(*test_order))


@pytest.fixture(autouse=True)
def disable_notification_dismiss_timer(monkeypatch):
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

        monkeypatch.setattr(NapariQtNotification, "DISMISS_AFTER", 0)
        monkeypatch.setattr(NapariQtNotification, "FADE_IN_RATE", 0)
        monkeypatch.setattr(NapariQtNotification, "FADE_OUT_RATE", 0)


@pytest.fixture()
def single_threaded_executor():
    executor = ThreadPoolExecutor(max_workers=1)
    yield executor
    executor.shutdown()


@pytest.fixture(autouse=True)
def _mock_app():
    """Mock clean 'test_app' `NapariApplication` instance.

    This is used whenever `napari._app_model.get_app()` is called to return
    a 'test_app' `NapariApplication` instead of the 'napari'
    `NapariApplication`.

    Note that `NapariApplication` registers app-model actions, providers and
    processors. If this is not desired, please create a clean
    `app_model.Application` in the test. It does not however, register Qt
    related actions or providers. If this is required for a unit test,
    `napari._qt._qapp_model.qactions.init_qactions()` can be used within
    the test.
    """
    from app_model import Application

    from napari._app_model._app import NapariApplication, _napari_names

    app = NapariApplication('test_app')
    app.injection_store.namespace = _napari_names
    with patch.object(NapariApplication, 'get_app', return_value=app):
        try:
            yield app
        finally:
            Application.destroy('test_app')


def _get_calling_place(depth=1):
    if not hasattr(sys, "_getframe"):
        return ""
    frame = sys._getframe(1 + depth)
    result = f"{frame.f_code.co_filename}:{frame.f_lineno}"
    if not frame.f_code.co_filename.startswith(ROOT_DIR):
        with suppress(ValueError):
            while not frame.f_code.co_filename.startswith(ROOT_DIR):
                frame = frame.f_back
                if frame is None:
                    break
            else:
                result += f" called from\n{frame.f_code.co_filename}:{frame.f_lineno}"
    return result


@pytest.fixture
def dangling_qthreads(monkeypatch, qtbot, request):
    from qtpy.QtCore import QThread

    base_start = QThread.start
    thread_dict = WeakKeyDictionary()
    # dict of threads that have been started but not yet terminated

    if "disable_qthread_start" in request.keywords:

        def my_start(*_, **__):
            """dummy function to prevent thread start"""

    else:

        def my_start(self, priority=QThread.InheritPriority):
            thread_dict[self] = _get_calling_place()
            base_start(self, priority)

    monkeypatch.setattr(QThread, 'start', my_start)
    yield

    dangling_threads_li = []

    for thread, calling in thread_dict.items():
        try:
            if thread.isRunning():
                dangling_threads_li.append((thread, calling))
        except RuntimeError as e:
            if (
                "wrapped C/C++ object of type" not in e.args[0]
                and "Internal C++ object" not in e.args[0]
            ):
                raise

    for thread, _ in dangling_threads_li:
        with suppress(RuntimeError):
            thread.quit()
            qtbot.waitUntil(thread.isFinished, timeout=2000)

    long_desc = (
        "If you see this error, it means that a QThread was started in a test "
        "but not terminated. This can cause segfaults in the test suite. "
        "Please use the `qtbot` fixture to wait for the thread to finish. "
        "If you think that the thread is obsolete for this test, you can "
        "use the `@pytest.mark.disable_qthread_start` mark or  `monkeypatch` "
        "fixture to patch the `start` method of the "
        "QThread class to do nothing.\n"
    )

    if len(dangling_threads_li) > 1:
        long_desc += " The QThreads were started in:\n"
    else:
        long_desc += " The QThread was started in:\n"

    assert not dangling_threads_li, long_desc + "\n".join(
        x[1] for x in dangling_threads_li
    )


@pytest.fixture
def dangling_qthread_pool(monkeypatch, request):
    from qtpy.QtCore import QThreadPool

    base_start = QThreadPool.start
    threadpool_dict = WeakKeyDictionary()
    # dict of threadpools that have been used to run QRunnables

    if "disable_qthread_pool_start" in request.keywords:

        def my_start(*_, **__):
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
        "If you see this error, it means that a QThreadPool was used to run "
        "a QRunnable in a test but not terminated. This can cause segfaults "
        "in the test suite. Please use the `qtbot` fixture to wait for the "
        "thread to finish. If you think that the thread is obsolete for this "
        "use the `@pytest.mark.disable_qthread_pool_start` mark or  `monkeypatch` "
        "fixture to patch the `start` "
        "method of the QThreadPool class to do nothing.\n"
    )
    if len(dangling_threads_pools) > 1:
        long_desc += " The QThreadPools were used in:\n"
    else:
        long_desc += " The QThreadPool was used in:\n"

    assert not dangling_threads_pools, long_desc + "\n".join(
        "; ".join(x[1]) for x in dangling_threads_pools
    )


@pytest.fixture
def dangling_qtimers(monkeypatch, request):
    from qtpy.QtCore import QTimer

    base_start = QTimer.start
    timer_dkt = WeakKeyDictionary()
    single_shot_list = []

    if "disable_qtimer_start" in request.keywords:
        from pytestqt.qt_compat import qt_api

        def my_start(*_, **__):
            """dummy function to prevent timer start"""

        _single_shot = my_start

        class OldTimer(QTimer):
            def start(self, time=None):
                if time is not None:
                    base_start(self, time)
                else:
                    base_start(self)

        monkeypatch.setattr(qt_api.QtCore, "QTimer", OldTimer)
        # This monkeypatch is require to keep `qtbot.waitUntil` working

    else:

        def my_start(self, msec=None):
            timer_dkt[self] = _get_calling_place()
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
        "If you see this error, it means that a QTimer was started but not stopped. "
        "This can cause tests to fail, and can also cause segfaults. "
        "If this test does not require a QTimer to pass you could monkeypatch it out. "
        "If it does require a QTimer, you should stop or wait for it to finish before test ends. "
    )
    if len(dangling_timers) > 1:
        long_desc += "The QTimers were started in:\n"
    else:
        long_desc += "The QTimer was started in:\n"
    assert not dangling_timers, long_desc + "\n".join(
        x[1] for x in dangling_timers
    )


@pytest.fixture
def dangling_qanimations(monkeypatch, request):
    from qtpy.QtCore import QPropertyAnimation

    base_start = QPropertyAnimation.start
    animation_dkt = WeakKeyDictionary()

    if "disable_qanimation_start" in request.keywords:

        def my_start(*_, **__):
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
        "If you see this error, it means that a QPropertyAnimation was started but not stopped. "
        "This can cause tests to fail, and can also cause segfaults. "
        "If this test does not require a QPropertyAnimation to pass you could monkeypatch it out. "
        "If it does require a QPropertyAnimation, you should stop or wait for it to finish before test ends. "
    )
    if len(dangling_animations) > 1:
        long_desc += " The QPropertyAnimations were started in:\n"
    else:
        long_desc += " The QPropertyAnimation was started in:\n"
    assert not dangling_animations, long_desc + "\n".join(
        x[1] for x in dangling_animations
    )


def pytest_runtest_setup(item):
    if "qapp" in item.fixturenames:
        # here we do autouse for dangling fixtures only if qapp is used
        if "qtbot" not in item.fixturenames:
            # for proper waiting for threads to finish
            item.fixturenames.append("qtbot")

        item.fixturenames.extend(
            [
                "dangling_qthread_pool",
                "dangling_qanimations",
                "dangling_qthreads",
                "dangling_qtimers",
            ]
        )
