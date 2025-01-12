"""Methods to create a new viewer instance then add a particular layer type.

All functions follow this pattern, (where <layer_type> is replaced with one
of the layer types, like "image", "points", etc...):

.. code-block:: python

    def view_<layer_type>(*args, **kwargs):
        # ... pop all of the viewer kwargs out of kwargs into viewer_kwargs
        viewer = Viewer(**viewer_kwargs)
        add_method = getattr(viewer, f"add_{<layer_type>}")
        add_method(*args, **kwargs)
        return viewer
"""

import inspect
from typing import Any

from numpydoc.docscrape import NumpyDocString as _NumpyDocString

from napari.components.dims import Dims
from napari.layers import Image
from napari.viewer import Viewer

__all__ = [
    'imshow',
    'view_image',
    'view_labels',
    'view_path',
    'view_points',
    'view_shapes',
    'view_surface',
    'view_tracks',
    'view_vectors',
]

_doc_template = """Create a viewer and add a{n} {layer_string} layer.

{params}

Returns
-------
viewer : :class:`napari.Viewer`
    The newly-created viewer.
"""

_VIEW_DOC = _NumpyDocString(Viewer.__doc__)
_VIEW_PARAMS = '    ' + '\n'.join(_VIEW_DOC._str_param_list('Parameters')[2:])


def _merge_docstrings(add_method, layer_string):
    # create combined docstring with parameters from add_* and Viewer methods
    import textwrap

    add_method_doc = _NumpyDocString(add_method.__doc__)

    # this ugliness is because the indentation of the parsed numpydocstring
    # is different for the first parameter :(
    lines = add_method_doc._str_param_list('Parameters')
    lines = lines[:3] + textwrap.dedent('\n'.join(lines[3:])).splitlines()
    params = '\n'.join(lines) + '\n' + textwrap.dedent(_VIEW_PARAMS)
    n = 'n' if layer_string.startswith(tuple('aeiou')) else ''
    return _doc_template.format(n=n, layer_string=layer_string, params=params)


def _merge_layer_viewer_sigs_docs(func):
    """Make combined signature, docstrings, and annotations for `func`.

    This is a decorator that combines information from `Viewer.__init__`,
    and one of the `viewer.add_*` methods.  It updates the docstring,
    signature, and type annotations of the decorated function with the merged
    versions.

    Parameters
    ----------
    func : callable
        `view_<layer_type>` function to modify

    Returns
    -------
    func : callable
        The same function, with merged metadata.
    """
    from napari.utils.misc import _combine_signatures

    # get the `Viewer.add_*` method
    layer_string = func.__name__.replace('view_', '')
    if layer_string == 'path':
        add_method = Viewer.open
    else:
        add_method = getattr(Viewer, f'add_{layer_string}')

    # merge the docstrings of Viewer and viewer.add_*
    func.__doc__ = _merge_docstrings(add_method, layer_string)

    # merge the signatures of Viewer and viewer.add_*
    func.__signature__ = _combine_signatures(
        add_method,
        Viewer,
        return_annotation=Viewer,
        exclude=('self', 'axis_labels'),
    )

    # merge the __annotations__
    func.__annotations__ = {
        **add_method.__annotations__,
        **Viewer.__init__.__annotations__,
        'return': Viewer,
    }

    # _forwardrefns_ is used by stubgen.py to populate the globals
    # when evaluate forward references with get_type_hints
    func._forwardrefns_ = {**add_method.__globals__}
    return func


_viewer_params = inspect.signature(Viewer).parameters
_dims_params = Dims.__fields__


def _make_viewer_then(
    add_method: str,
    /,
    *args,
    viewer: Viewer | None = None,
    **kwargs,
) -> tuple[Viewer, Any]:
    """Create a viewer, call given add_* method, then return viewer and layer.

    This function will be deprecated soon (See #4693)

    Parameters
    ----------
    add_method : str
        Which ``add_`` method to call on the viewer, e.g. `add_image`,
        or `add_labels`.
    *args : list
        Positional arguments for the ``add_`` method.
    viewer : Viewer, optional
        A pre-existing viewer, which will be used provided, rather than
        creating a new one.
    **kwargs : dict
        Keyword arguments for either the `Viewer` constructor or for the
        ``add_`` method.

    Returns
    -------
    viewer : napari.Viewer
        The created viewer, or the same one that was passed in, if given.
    layer(s): napari.layers.Layer or List[napari.layers.Layer]
        The value returned by the add_method. Can be a list of layers if
        ``add_image`` is called with a ``channel_axis=`` keyword
        argument.
    """
    vkwargs = {
        k: kwargs.pop(k)
        for k in list(kwargs)
        if k in _viewer_params
        if k != 'axis_labels'
    }
    if 'axis_labels' in kwargs:
        vkwargs['axis_labels'] = (
            kwargs['axis_labels'] if kwargs['axis_labels'] is not None else ()
        )
    # separate dims kwargs because we want to set those after adding data
    dims_kwargs = {
        k: vkwargs.pop(k) for k in list(vkwargs) if k in _dims_params
    }
    if viewer is None:
        viewer = Viewer(**vkwargs)
    kwargs.update(kwargs.pop('kwargs', {}))
    method = getattr(viewer, add_method)
    added = method(*args, **kwargs)
    if isinstance(added, list):
        added = tuple(added)
    for arg_name, arg_val in dims_kwargs.items():
        setattr(viewer.dims, arg_name, arg_val)
    return viewer, added


# Each of the following functions will have this pattern:
#
# def view_image(*args, **kwargs):
#     # ... pop all of the viewer kwargs out of kwargs into viewer_kwargs
#     viewer = Viewer(**viewer_kwargs)
#     viewer.add_image(*args, **kwargs)
#     return viewer


@_merge_layer_viewer_sigs_docs
def view_image(*args, **kwargs):
    return _make_viewer_then('add_image', *args, **kwargs)[0]


@_merge_layer_viewer_sigs_docs
def view_labels(*args, **kwargs):
    return _make_viewer_then('add_labels', *args, **kwargs)[0]


@_merge_layer_viewer_sigs_docs
def view_points(*args, **kwargs):
    return _make_viewer_then('add_points', *args, **kwargs)[0]


@_merge_layer_viewer_sigs_docs
def view_shapes(*args, **kwargs):
    return _make_viewer_then('add_shapes', *args, **kwargs)[0]


@_merge_layer_viewer_sigs_docs
def view_surface(*args, **kwargs):
    return _make_viewer_then('add_surface', *args, **kwargs)[0]


@_merge_layer_viewer_sigs_docs
def view_tracks(*args, **kwargs):
    return _make_viewer_then('add_tracks', *args, **kwargs)[0]


@_merge_layer_viewer_sigs_docs
def view_vectors(*args, **kwargs):
    return _make_viewer_then('add_vectors', *args, **kwargs)[0]


@_merge_layer_viewer_sigs_docs
def view_path(*args, **kwargs):
    return _make_viewer_then('open', *args, **kwargs)[0]


def imshow(
    data,
    *,
    channel_axis=None,
    affine=None,
    axis_labels=None,
    attenuation=0.05,
    blending=None,
    cache=True,
    colormap=None,
    contrast_limits=None,
    custom_interpolation_kernel_2d=None,
    depiction='volume',
    experimental_clipping_planes=None,
    gamma=1.0,
    interpolation2d='nearest',
    interpolation3d='linear',
    iso_threshold=None,
    metadata=None,
    multiscale=None,
    name=None,
    opacity=1.0,
    plane=None,
    projection_mode='none',
    rendering='mip',
    rgb=None,
    rotate=None,
    scale=None,
    shear=None,
    translate=None,
    units=None,
    visible=True,
    viewer=None,
    title='napari',
    ndisplay=2,
    order=(),
    show=True,
) -> tuple[Viewer, list['Image']]:
    """Load data into an Image layer and return the Viewer and Layer.

    Parameters
    ----------
    data : array or list of array
        Image data. Can be N >= 2 dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a
        list and arrays are decreasing in shape then the data is treated as
        a multiscale image. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    channel_axis : int, optional
        Axis to expand image along. If provided, each channel in the data
        will be added as an individual image layer. In channel_axis mode,
        other parameters MAY be provided as lists. The Nth value of the list
        will be applied to the Nth channel in the data. If a single value
        is provided, it will be broadcast to all Layers.
        All parameters except data, rgb, and multiscale can be provided as
        list of values. If a list is provided, it must be the same length as
        the axis that is being expanded as channels.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a
        napari `Affine` transform object. Applied as an extra transform on
        top of the provided scale, rotate, and shear values.
    attenuation : float or list of float
        Attenuation rate for attenuated maximum intensity projection.
    blending : str or list of str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'translucent', 'translucent_no_depth', 'additive', 'minimum', 'opaque'}.
    cache : bool or list of bool
        Whether slices of out-of-core datasets should be cached upon
        retrieval. Currently, this only applies to dask arrays.
    colormap : str, napari.utils.Colormap, tuple, dict, list or list of these types
        Colormaps to use for luminance images. If a string, it can be the name
        of a supported colormap from vispy or matplotlib or the name of
        a vispy color or a hexadecimal RGB color representation.
        If a tuple, the first value must be a string to assign as a name to a
        colormap and the second item must be a Colormap. If a dict, the key must
        be a string to assign as a name to a colormap and the value must be a
        Colormap.
    contrast_limits : list (2,)
        Intensity value limits to be used for determining the minimum and maximum colormap bounds for
        luminance images. If not passed, they will be calculated as the min and max intensity value of
        the image.
    custom_interpolation_kernel_2d : np.ndarray
        Convolution kernel used with the 'custom' interpolation mode in 2D rendering.
    depiction : str or list of str
        3D Depiction mode. Must be one of {'volume', 'plane'}.
        The default value is 'volume'.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.
    gamma : float or list of float
        Gamma correction for determining colormap linearity; defaults to 1.
    interpolation2d : str or list of str
        Interpolation mode used by vispy for rendering 2d data.
        Must be one of our supported modes.
        (for list of supported modes see Interpolation enum)
        'custom' is a special mode for 2D interpolation in which a regular grid
        of samples is taken from the texture around a position using 'linear'
        interpolation before being multiplied with a custom interpolation kernel
        (provided with 'custom_interpolation_kernel_2d').
    interpolation3d : str or list of str
        Same as 'interpolation2d' but for 3D rendering.
    iso_threshold : float or list of float
        Threshold for isosurface.
    metadata : dict or list of dict
        Layer metadata.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array-like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape,
        then it will be taken to be multiscale. The first image in the list
        should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    name : str or list of str
        Name of the layer.
    opacity : float or list
        Opacity of the layer visual, between 0.0 and 1.0.
    plane : dict or SlicingPlane
        Properties defining plane rendering in 3D. Properties are defined in
        data coordinates. Valid dictionary keys are
        {'position', 'normal', 'thickness', and 'enabled'}.
    projection_mode : str
        How data outside the viewed dimensions, but inside the thick Dims slice will
        be projected onto the viewed dimensions. Must fit to cls._projectionclass
    rendering : str or list of str
        Rendering mode used by vispy. Must be one of our supported
        modes. If a list then must be same length as the axis that is being
        expanded as channels.
    rgb : bool, optional
        Whether the image is RGB or RGBA if rgb. If not
        specified by user, but the last dimension of the data has length 3 or 4,
        it will be set as `True`. If `False`, the image is interpreted as a
        luminance image.
    rotate : float, 3-tuple of float, n-D array or list.
        If a float, convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple, convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise, assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        'np.degrees' if needed.
    scale : tuple of float or list of tuple of float
        Scale factors for the layer.
    shear : 1-D array or list.
        A vector of shear values for an upper triangular n-D shear matrix.
    translate : tuple of float or list of tuple of float
        Translation values for the layer.
    visible : bool or list of bool
        Whether the layer visual is currently being displayed.
    viewer : Viewer object, optional, by default None.
    title : string, optional
        The title of the viewer window. By default 'napari'.
    ndisplay : {2, 3}, optional
        Number of displayed dimensions. By default 2.
    order : tuple of int, optional
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3. By default None
    axis_labels : list of str, optional
        Dimension names. By default they are labeled with sequential numbers
    show : bool, optional
        Whether to show the viewer after instantiation. By default True.

    Returns
    -------
    viewer : napari.Viewer
        The created or passed viewer.
    layer(s) : napari.layers.Image or List[napari.layers.Image]
        The added layer(s). (May be more than one if the ``channel_axis`` keyword
        argument is given.
    """

    return _make_viewer_then(
        'add_image',
        data,
        viewer=viewer,
        channel_axis=channel_axis,
        axis_labels=axis_labels,
        rgb=rgb,
        colormap=colormap,
        contrast_limits=contrast_limits,
        gamma=gamma,
        interpolation2d=interpolation2d,
        interpolation3d=interpolation3d,
        rendering=rendering,
        depiction=depiction,
        iso_threshold=iso_threshold,
        attenuation=attenuation,
        name=name,
        metadata=metadata,
        scale=scale,
        translate=translate,
        rotate=rotate,
        shear=shear,
        affine=affine,
        opacity=opacity,
        blending=blending,
        visible=visible,
        multiscale=multiscale,
        cache=cache,
        plane=plane,
        units=units,
        experimental_clipping_planes=experimental_clipping_planes,
        custom_interpolation_kernel_2d=custom_interpolation_kernel_2d,
        projection_mode=projection_mode,
        title=title,
        ndisplay=ndisplay,
        order=order,
        show=show,
    )
