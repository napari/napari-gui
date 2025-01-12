"""Patch callables (functions and methods).

Our perfmon system using this to patch in perf_timers, but this can be used
for any type of patching. See patch_callables() below as the main entrypoint.
"""

import logging
import types
from collections.abc import Callable
from importlib import import_module
from typing import Union

from napari.utils.translations import trans

# The parent of a callable is a module or a class, class is of type "type".
CallableParent = Union[types.ModuleType, type]

# An example PatchFunction is:
# def _patch_perf_timer(parent, callable: str, label: str) -> None
PatchFunction = Callable[[CallableParent, str, str], None]


class PatchError(Exception):
    """Failed to patch target, config file error?"""


def _patch_attribute(
    module: types.ModuleType, attribute_str: str, patch_func: PatchFunction
) -> None:
    """Patch the module's callable pointed to by the attribute string.

    Parameters
    ----------
    module : types.ModuleType
        The module to patch.
    attribute_str : str
        An attribute in the module like "function" or "class.method".
    patch_func : PatchFunction
        This function is called to perform the patch.

    """
    # We expect attribute_str is <function> or <class>.<method>. We could
    # allow nested classes and functions if we wanted to extend this some.
    if attribute_str.count('.') > 1:
        raise PatchError(
            trans._(
                'Nested attribute not found: {attribute_str}',
                deferred=True,
                attribute_str=attribute_str,
            )
        )

    if '.' in attribute_str:
        # Assume attribute_str is <class>.<method>
        class_str, callable_str = attribute_str.split('.')
        try:
            parent = getattr(module, class_str)
        except AttributeError as e:
            raise PatchError(
                trans._(
                    'Module {module_name} has no attribute {attribute_str}',
                    deferred=True,
                    module_name=module.__name__,
                    attribute_str=attribute_str,
                )
            ) from e
        parent_str = class_str
    else:
        # Assume attribute_str is <function>.
        class_str = None
        parent = module
        parent_str = module.__name__
        callable_str = attribute_str

    try:
        getattr(parent, callable_str)
    except AttributeError as e:
        raise PatchError(
            trans._(
                'Parent {parent_str} has no attribute {callable_str}',
                deferred=True,
                parent_str=parent_str,
                callable_str=callable_str,
            )
        ) from e

    label = (
        callable_str if class_str is None else f'{class_str}.{callable_str}'
    )

    # Patch it with the user-provided patch_func.
    logging.info('patching %s.%s', module.__name__, label)
    patch_func(parent, callable_str, label)


def _import_module(
    target_str: str,
) -> tuple[types.ModuleType, str] | tuple[None, None]:
    """Import the module portion of this target string.

    Try importing successively longer segments of the target_str. For example:
       napari.components.experimental.chunk._loader.ChunkLoader.load_chunk
    will import:
        napari (success)
        napari.components (success)
        napari.components.experimental (success)
        napari.components.experimental.chunk (success)
        napari.components.experimental.chunk._loader (success)
        napari.components.experimental.chunk._loader.ChunkLoader (failure, not a module)

    The last one fails because ChunkLoader is a class not a module.

    Parameters
    ----------
    target_str : str
        The fully qualified callable such as "module1.module2.function".

    Returns
    -------
    Tuple[types.ModuleType, str]
        Where the module is the inner most imported module, and the string
        is the rest of target_str that was not modules.
    """
    parts = target_str.split('.')
    module = None  # Inner-most module imported so far.

    # Progressively try to import longer and longer segments of the path.
    for i in range(1, len(target_str)):
        module_path = '.'.join(parts[:i])
        try:
            module = import_module(module_path)
        except ModuleNotFoundError as e:
            if module is None:
                # The very first top-level module import failed!
                raise PatchError(
                    trans._(
                        'Module not found: {module_path}',
                        deferred=True,
                        module_path=module_path,
                    )
                ) from e

            # We successfully imported part of the target_str but then
            # we got a failure. Usually this is because we tried
            # importing a class or function. Return the inner-most
            # module we did successfully import. And return the rest of
            # the module_path we didn't use.
            attribute_str = '.'.join(parts[i - 1 :])
            return module, attribute_str
    return None, None


def patch_callables(callables: list[str], patch_func: PatchFunction) -> None:
    """Patch the given list of callables.

    Parameters
    ----------
    callables : List[str]
        Patch all of these callables (functions or methods).
    patch_func : PatchFunction
        Called on every callable to patch it.

    Notes
    -----
    The callables list should look like:
    [
        "module.module.ClassName.method_name",
        "module.function_name"
        ...
    ]

    Nested classes and methods not allowed, but support could be added.

    An example patch_func is::

        import wrapt
        def _my_patcher(parent: CallableParent, callable: str, label: str):
            @wrapt.patch_function_wrapper(parent, callable)
            def my_announcer(wrapped, instance, args, kwargs):
                print(f"Announce {label}")
                return wrapped(*args, **kwargs)
    """
    patched: set[str] = set()

    for target_str in callables:
        if target_str in patched:
            # Ignore duplicated targets in the config file.
            logging.warning('skipping duplicate %s', target_str)
            continue

        # Patch the target and note that we did.
        try:
            module, attribute_str = _import_module(target_str)
            if module is not None and attribute_str is not None:
                _patch_attribute(module, attribute_str, patch_func)
        except PatchError:
            # We don't stop on error because if you switch around branches
            # but keep the same config file, it's easy for your config
            # file to contain targets that aren't in the code.

            # logging.exception magically logs the stack trace too!
            logging.exception('Something went wrong while patching')

        patched.add(target_str)
