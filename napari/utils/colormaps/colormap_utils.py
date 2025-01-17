import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from functools import lru_cache
from threading import Lock
from typing import NamedTuple, Optional, Union

import numpy as np
import skimage.color as colorconv
from vispy.color import (
    BaseColormap as VispyColormap,
    Color,
    ColorArray,
    get_colormap,
    get_colormaps,
)
from vispy.color.colormap import LUT_len

from napari.utils.colormaps._accelerated_cmap import minimum_dtype_for_labels
from napari.utils.colormaps.bop_colors import bopd
from napari.utils.colormaps.colormap import (
    Colormap,
    ColormapInterpolationMode,
    CyclicLabelColormap,
    DirectLabelColormap,
)
from napari.utils.colormaps.inverse_colormaps import inverse_cmaps
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.colormaps.vendored import cm
from napari.utils.translations import trans

# All parsable input color types that a user can provide
ColorType = Union[list, tuple, np.ndarray, str, Color, ColorArray]


ValidColormapArg = Union[
    str,
    ColorType,
    VispyColormap,
    Colormap,
    tuple[str, VispyColormap],
    tuple[str, Colormap],
    dict[str, VispyColormap],
    dict[str, Colormap],
    dict,
]


matplotlib_colormaps = _MATPLOTLIB_COLORMAP_NAMES = OrderedDict(
    viridis=trans._p('colormap', 'viridis'),
    magma=trans._p('colormap', 'magma'),
    inferno=trans._p('colormap', 'inferno'),
    plasma=trans._p('colormap', 'plasma'),
    hsv=trans._p('colormap', 'hsv'),
    turbo=trans._p('colormap', 'turbo'),
    twilight=trans._p('colormap', 'twilight'),
    twilight_shifted=trans._p('colormap', 'twilight shifted'),
    gist_earth=trans._p('colormap', 'gist earth'),
    PiYG=trans._p('colormap', 'PiYG'),
)
_MATPLOTLIB_COLORMAP_NAMES_REVERSE = {
    v: k for k, v in matplotlib_colormaps.items()
}
_VISPY_COLORMAPS_ORIGINAL = _VCO = get_colormaps()
_VISPY_COLORMAPS_TRANSLATIONS = OrderedDict(
    autumn=(trans._p('colormap', 'autumn'), _VCO['autumn']),
    blues=(trans._p('colormap', 'blues'), _VCO['blues']),
    cool=(trans._p('colormap', 'cool'), _VCO['cool']),
    greens=(trans._p('colormap', 'greens'), _VCO['greens']),
    reds=(trans._p('colormap', 'reds'), _VCO['reds']),
    spring=(trans._p('colormap', 'spring'), _VCO['spring']),
    summer=(trans._p('colormap', 'summer'), _VCO['summer']),
    fire=(trans._p('colormap', 'fire'), _VCO['fire']),
    grays=(trans._p('colormap', 'grays'), _VCO['grays']),
    hot=(trans._p('colormap', 'hot'), _VCO['hot']),
    ice=(trans._p('colormap', 'ice'), _VCO['ice']),
    winter=(trans._p('colormap', 'winter'), _VCO['winter']),
    light_blues=(trans._p('colormap', 'light blues'), _VCO['light_blues']),
    orange=(trans._p('colormap', 'orange'), _VCO['orange']),
    viridis=(trans._p('colormap', 'viridis'), _VCO['viridis']),
    coolwarm=(trans._p('colormap', 'coolwarm'), _VCO['coolwarm']),
    PuGr=(trans._p('colormap', 'PuGr'), _VCO['PuGr']),
    GrBu=(trans._p('colormap', 'GrBu'), _VCO['GrBu']),
    GrBu_d=(trans._p('colormap', 'GrBu_d'), _VCO['GrBu_d']),
    RdBu=(trans._p('colormap', 'RdBu'), _VCO['RdBu']),
    cubehelix=(trans._p('colormap', 'cubehelix'), _VCO['cubehelix']),
    single_hue=(trans._p('colormap', 'single hue'), _VCO['single_hue']),
    hsl=(trans._p('colormap', 'hsl'), _VCO['hsl']),
    husl=(trans._p('colormap', 'husl'), _VCO['husl']),
    diverging=(trans._p('colormap', 'diverging'), _VCO['diverging']),
    RdYeBuCy=(trans._p('colormap', 'RdYeBuCy'), _VCO['RdYeBuCy']),
)
_VISPY_COLORMAPS_TRANSLATIONS_REVERSE = {
    v[0]: k for k, v in _VISPY_COLORMAPS_TRANSLATIONS.items()
}
_PRIMARY_COLORS = OrderedDict(
    red=(trans._p('colormap', 'red'), [1.0, 0.0, 0.0]),
    green=(trans._p('colormap', 'green'), [0.0, 1.0, 0.0]),
    blue=(trans._p('colormap', 'blue'), [0.0, 0.0, 1.0]),
    cyan=(trans._p('colormap', 'cyan'), [0.0, 1.0, 1.0]),
    magenta=(trans._p('colormap', 'magenta'), [1.0, 0.0, 1.0]),
    yellow=(trans._p('colormap', 'yellow'), [1.0, 1.0, 0.0]),
)

SIMPLE_COLORMAPS = {
    name: Colormap(
        name=name, display_name=display_name, colors=[[0.0, 0.0, 0.0], color]
    )
    for name, (display_name, color) in _PRIMARY_COLORS.items()
}

# add conventional grayscale colormap as a simple one
SIMPLE_COLORMAPS.update(
    {
        'gray': Colormap(
            name='gray',
            display_name='gray',
            colors=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        )
    }
)

# dictionary for bop colormap objects
BOP_COLORMAPS = {
    name: Colormap(value, name=name, display_name=display_name)
    for name, (display_name, value) in bopd.items()
}

INVERSE_COLORMAPS = {
    name: Colormap(value, name=name, display_name=display_name)
    for name, (display_name, value) in inverse_cmaps.items()
}

# Add the reversed grayscale colormap (white to black) to inverse colormaps
INVERSE_COLORMAPS.update(
    {
        'gray_r': Colormap(
            name='gray_r',
            display_name='gray r',
            colors=[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        ),
    }
)

_FLOAT32_MAX = float(np.finfo(np.float32).max)
_MAX_VISPY_SUPPORTED_VALUE = _FLOAT32_MAX / 8
# Using 8 as divisor comes from experiments.
# For some reason if use smaller number,
# the image is not displayed correctly.

_MINIMUM_SHADES_COUNT = 256


def _all_rgb():
    """Return all 256**3 valid rgb tuples."""
    base = np.arange(256, dtype=np.uint8)
    r, g, b = np.meshgrid(base, base, base, indexing='ij')
    return np.stack((r, g, b), axis=-1).reshape((-1, 3))


# The following values were precomputed and stored as constants
# here to avoid heavy computation when importing this module.
# The following code can be used to reproduce these values.
#
# rgb_colors = _all_rgb()
# luv_colors = colorconv.rgb2luv(rgb_colors)
# LUVMIN = np.amin(luv_colors, axis=(0,))
# LUVMAX = np.amax(luv_colors, axis=(0,))
# lab_colors = colorconv.rgb2lab(rgb_colors)
# LABMIN = np.amin(lab_colors, axis=(0,))
# LABMAX = np.amax(lab_colors, axis=(0,))

LUVMIN = np.array([0.0, -83.07790815, -134.09790293])
LUVMAX = np.array([100.0, 175.01447356, 107.39905336])
LUVRNG = LUVMAX - LUVMIN

LABMIN = np.array([0.0, -86.18302974, -107.85730021])
LABMAX = np.array([100.0, 98.23305386, 94.47812228])
LABRNG = LABMAX - LABMIN


def convert_vispy_colormap(colormap, name='vispy'):
    """Convert a vispy colormap object to a napari colormap.

    Parameters
    ----------
    colormap : vispy.color.Colormap
        Vispy colormap object that should be converted.
    name : str
        Name of colormap, optional.

    Returns
    -------
    napari.utils.Colormap
    """
    if not isinstance(colormap, VispyColormap):
        raise TypeError(
            trans._(
                'Colormap must be a vispy colormap if passed to from_vispy',
                deferred=True,
            )
        )

    # Not all vispy colormaps have an `_controls`
    # but if they do, we want to use it
    if hasattr(colormap, '_controls'):
        controls = colormap._controls
    else:
        controls = np.zeros((0,))

    # Not all vispy colormaps have an `interpolation`
    # but if they do, we want to use it
    if hasattr(colormap, 'interpolation'):
        interpolation = colormap.interpolation
    else:
        interpolation = 'linear'

    if name in _VISPY_COLORMAPS_TRANSLATIONS:
        display_name, _cmap = _VISPY_COLORMAPS_TRANSLATIONS[name]
    else:
        # Unnamed colormap
        display_name = trans._(name)

    return Colormap(
        name=name,
        display_name=display_name,
        colors=colormap.colors.rgba,
        controls=controls,
        interpolation=interpolation,
    )


def _validate_rgb(colors, *, tolerance=0.0):
    """Return the subset of colors that is in [0, 1] for all channels.

    Parameters
    ----------
    colors : array of float, shape (N, 3)
        Input colors in RGB space.

    Returns
    -------
    filtered_colors : array of float, shape (M, 3), M <= N
        The subset of colors that are in valid RGB space.

    Other Parameters
    ----------------
    tolerance : float, optional
        Values outside of the range by less than ``tolerance`` are allowed and
        clipped to be within the range.

    Examples
    --------
    >>> colors = np.array([[  0. , 1.,  1.  ],
    ...                    [  1.1, 0., -0.03],
    ...                    [  1.2, 1.,  0.5 ]])
    >>> _validate_rgb(colors)
    array([[0., 1., 1.]])
    >>> _validate_rgb(colors, tolerance=0.15)
    array([[0., 1., 1.],
           [1., 0., 0.]])
    """
    lo = 0 - tolerance
    hi = 1 + tolerance
    valid = np.all((colors >= lo) & (colors <= hi), axis=1)
    filtered_colors = np.clip(colors[valid], 0, 1)
    return filtered_colors


def low_discrepancy_image(image, seed=0.5, margin=1 / 256) -> np.ndarray:
    """Generate a 1d low discrepancy sequence of coordinates.

    Parameters
    ----------
    image : array of int
        A set of labels or label image.
    seed : float
        The seed from which to start the quasirandom sequence.
        Effective range is [0,1.0), as only the decimals are used.
    margin : float
        Values too close to 0 or 1 will get mapped to the edge of the colormap,
        so we need to offset to a margin slightly inside those values. Since
        the bin size is 1/256 by default, we offset by that amount.

    Returns
    -------
    image_out : array of float
        The set of ``labels`` remapped to [0, 1] quasirandomly.

    """
    phi_mod = 0.6180339887498948482
    image_float = np.float32(image)
    image_float = seed + image_float * phi_mod
    # We now map the floats to the range [0 + margin, 1 - margin]
    image_out = margin + (1 - 2 * margin) * (
        image_float - np.floor(image_float)
    )

    # Clear zero (background) values, matching the shader behavior in _glsl_label_step
    image_out[image == 0] = 0.0
    return image_out


def color_dict_to_colormap(colors):
    """
    Generate a color map based on the given color dictionary

    Parameters
    ----------
    colors : dict of int to array of float, shape (4)
        Mapping between labels and color

    Returns
    -------
    colormap : napari.utils.Colormap
        Colormap constructed with provided control colors
    label_color_index : dict of int
        Mapping of Label to color control point within colormap
    """

    MAX_DISTINCT_COLORS = LUT_len

    control_colors = np.unique(list(colors.values()), axis=0)

    if len(control_colors) >= MAX_DISTINCT_COLORS:
        warnings.warn(
            trans._(
                'Label layers with more than {max_distinct_colors} distinct colors will not render correctly. This layer has {distinct_colors}.',
                deferred=True,
                distinct_colors=str(len(control_colors)),
                max_distinct_colors=str(MAX_DISTINCT_COLORS),
            ),
            category=UserWarning,
        )

    colormap = Colormap(
        colors=control_colors, interpolation=ColormapInterpolationMode.ZERO
    )

    control2index = {
        tuple(color): control_point
        for color, control_point in zip(colormap.colors, colormap.controls)
    }

    control_small_delta = 0.5 / len(control_colors)
    label_color_index = {
        label: np.float32(control2index[tuple(color)] + control_small_delta)
        for label, color in colors.items()
    }

    return colormap, label_color_index


def _low_discrepancy(dim, n, seed=0.5):
    """Generate a 1d, 2d, or 3d low discrepancy sequence of coordinates.

    Parameters
    ----------
    dim : one of {1, 2, 3}
        The dimensionality of the sequence.
    n : int
        How many points to generate.
    seed : float or array of float, shape (dim,)
        The seed from which to start the quasirandom sequence.
        Effective range is [0,1.0), as only the decimals are used.

    Returns
    -------
    pts : array of float, shape (n, dim)
        The sampled points.

    References
    ----------
    ..[1]: http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/  # noqa: E501
    """
    phi1 = 1.6180339887498948482
    phi2 = 1.32471795724474602596
    phi3 = 1.22074408460575947536
    seed = np.broadcast_to(seed, (1, dim))
    phi = np.array([phi1, phi2, phi3])
    g = 1 / phi
    n = np.reshape(np.arange(n), (n, 1))
    pts = (seed + (n * g[:dim])) % 1
    return pts


def _color_random(n, *, colorspace='lab', tolerance=0.0, seed=0.5):
    """Generate n random RGB colors uniformly from LAB or LUV space.

    Parameters
    ----------
    n : int
        Number of colors to generate.
    colorspace : str, one of {'lab', 'luv', 'rgb'}
        The colorspace from which to get random colors.
    tolerance : float
        How much margin to allow for out-of-range RGB values (these are
        clipped to be in-range).
    seed : float or array of float, shape (3,)
        Value from which to start the quasirandom sequence.
        Effective range is [0,1.0), as only the decimals are used.

    Returns
    -------
    rgb : array of float, shape (n, 3)
        RGB colors chosen uniformly at random from given colorspace.
    """
    factor = 6  # about 1/5 of random LUV tuples are inside the space
    expand_factor = 2
    rgb = np.zeros((0, 3))
    while len(rgb) < n:
        random = _low_discrepancy(3, n * factor, seed=seed)
        if colorspace == 'luv':
            raw_rgb = colorconv.luv2rgb(random * LUVRNG + LUVMIN)
        elif colorspace == 'rgb':
            raw_rgb = random
        else:  # 'lab' by default
            # The values in random are in [0, 1], but since the LAB colorspace
            # is not exactly contained in the unit-box, some 3-tuples might not
            # be valid LAB color coordinates. scikit-image handles this by projecting
            # such coordinates into the colorspace, but will also warn when doing this.
            with warnings.catch_warnings():
                # skimage 0.20.0rc
                warnings.filterwarnings(
                    action='ignore',
                    message='Conversion from CIE-LAB, via XYZ to sRGB color space resulted in',
                    category=UserWarning,
                )
                # skimage <0.20
                warnings.filterwarnings(
                    action='ignore',
                    message='Color data out of range',
                    category=UserWarning,
                )
                raw_rgb = colorconv.lab2rgb(random * LABRNG + LABMIN)
        rgb = _validate_rgb(raw_rgb, tolerance=tolerance)
        factor *= expand_factor
    return rgb[:n]


def label_colormap(
    num_colors=256, seed=0.5, background_value=0
) -> CyclicLabelColormap:
    """Produce a colormap suitable for use with a given label set.

    Parameters
    ----------
    num_colors : int, optional
        Number of unique colors to use. Default used if not given.
        Colors are in addition to a transparent color 0.
    seed : float, optional
        The seed for the random color generator.
        Effective range is [0,1.0), as only the decimals are used.

    Returns
    -------
    colormap : napari.utils.CyclicLabelColormap
        A colormap for use with labels remapped to [0, 1].

    Notes
    -----
    0 always maps to fully transparent.
    """
    if num_colors < 1:
        raise ValueError('num_colors must be >= 1')

    # Starting the control points slightly above 0 and below 1 is necessary
    # to ensure that the background pixel 0 is transparent
    midpoints = np.linspace(0.00001, 1 - 0.00001, num_colors + 1)
    control_points = np.concatenate(
        (np.array([0]), midpoints, np.array([1.0]))
    )
    # make sure to add an alpha channel to the colors

    colors = np.concatenate(
        (
            _color_random(num_colors + 2, seed=seed),
            np.full((num_colors + 2, 1), 1),
        ),
        axis=1,
    )

    # from here
    values_ = np.arange(num_colors + 2)
    randomized_values = low_discrepancy_image(values_, seed=seed)

    indices = np.clip(
        np.searchsorted(control_points, randomized_values, side='right') - 1,
        0,
        len(control_points) - 1,
    )

    # here is an ugly hack to restore classical napari color order.
    colors = colors[indices][:-1]

    # ensure that we not need to deal with differences in float rounding for
    # CPU and GPU.
    uint8_max = np.iinfo(np.uint8).max
    rgb8_colors = (colors * uint8_max).astype(np.uint8)
    colors = rgb8_colors.astype(np.float32) / uint8_max

    return CyclicLabelColormap(
        name='label_colormap',
        display_name=trans._p('colormap', 'low discrepancy colors'),
        colors=colors,
        controls=np.linspace(0, 1, len(colors) + 1),
        interpolation='zero',
        background_value=background_value,
        seed=seed,
    )


@lru_cache
def _primes(upto=2**16):
    """Generate primes up to a given number.

    Parameters
    ----------
    upto : int
        The upper limit of the primes to generate.

    Returns
    -------
    primes : np.ndarray
        The primes up to the upper limit.
    """
    primes = np.arange(3, upto + 1, 2)
    isprime = np.ones((upto - 1) // 2, dtype=bool)
    max_factor = int(np.sqrt(upto))
    for factor in primes[: max_factor // 2]:
        if isprime[(factor - 2) // 2]:
            isprime[(factor * 3 - 2) // 2 : None : factor] = 0
    return np.concatenate(([2], primes[isprime]))


def shuffle_and_extend_colormap(
    colormap: CyclicLabelColormap, seed: int, min_random_choices: int = 5
) -> CyclicLabelColormap:
    """Shuffle the colormap colors and extend it to more colors.

    The new number of colors will be a prime number that fits into the same
    dtype as the current number of colors in the colormap.

    Parameters
    ----------
    colormap : napari.utils.CyclicLabelColormap
        Colormap to shuffle and extend.
    seed : int
        Seed for the random number generator.
    min_random_choices : int
        Minimum number of new table sizes to choose from. When choosing table
        sizes, every choice gives a 1/size chance of two labels being mapped
        to the same color. Since we try to stay within the same dtype as the
        original colormap, if the number of original colors is close to the
        maximum value for the dtype, there will not be enough prime numbers
        up to the dtype max to ensure that two labels can always be
        distinguished after one or few shuffles. In that case, we discard
        some colors and choose the `min_random_choices` largest primes to fit
        within the dtype.

    Returns
    -------
    colormap : napari.utils.CyclicLabelColormap
        Shuffled and extended colormap.
    """
    rng = np.random.default_rng(seed)
    n_colors_prev = len(colormap.colors)
    dtype = minimum_dtype_for_labels(n_colors_prev)
    indices = np.arange(n_colors_prev)
    rng.shuffle(indices)
    shuffled_colors = colormap.colors[indices]

    primes = _primes(
        np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 2**24
    )
    valid_primes = primes[primes > n_colors_prev]
    if len(valid_primes) < min_random_choices:
        valid_primes = primes[-min_random_choices:]
    n_colors = rng.choice(valid_primes)
    n_color_diff = n_colors - n_colors_prev

    extended_colors = np.concatenate(
        (
            shuffled_colors[: min(n_color_diff, 0) or None],
            shuffled_colors[rng.choice(indices, size=max(n_color_diff, 0))],
        ),
        axis=0,
    )

    new_colormap = CyclicLabelColormap(
        name=colormap.name,
        colors=extended_colors,
        controls=np.linspace(0, 1, len(extended_colors) + 1),
        interpolation='zero',
        background_value=colormap.background_value,
    )
    return new_colormap


def direct_colormap(color_dict=None):
    """Make a direct colormap from a dictionary mapping labels to colors.

    Parameters
    ----------
    color_dict : dict, optional
        A dictionary mapping labels to colors.

    Returns
    -------
    d : DirectLabelColormap
        A napari colormap whose map() function applies the color dictionary
        to an array.
    """
    # we don't actually use the color array, so pass dummy.
    return DirectLabelColormap(
        color_dict=color_dict or defaultdict(lambda: np.zeros(4)),
    )


def vispy_or_mpl_colormap(name) -> Colormap:
    """Try to get a colormap from vispy, or convert an mpl one to vispy format.

    Parameters
    ----------
    name : str
        The name of the colormap.

    Returns
    -------
    colormap : napari.utils.Colormap
        The found colormap.

    Raises
    ------
    KeyError
        If no colormap with that name is found within vispy or matplotlib.
    """
    if name in _VISPY_COLORMAPS_TRANSLATIONS:
        cmap = get_colormap(name)
        colormap = convert_vispy_colormap(cmap, name=name)
    else:
        try:
            mpl_cmap = getattr(cm, name)
            display_name = _MATPLOTLIB_COLORMAP_NAMES.get(name, name)
        except AttributeError as e:
            suggestion = _MATPLOTLIB_COLORMAP_NAMES_REVERSE.get(
                name
            ) or _VISPY_COLORMAPS_TRANSLATIONS_REVERSE.get(name)
            if suggestion:
                raise KeyError(
                    trans._(
                        'Colormap "{name}" not found in either vispy or matplotlib but you might want to use "{suggestion}".',
                        deferred=True,
                        name=name,
                        suggestion=suggestion,
                    )
                ) from e

            colormaps = set(_VISPY_COLORMAPS_ORIGINAL).union(
                set(_MATPLOTLIB_COLORMAP_NAMES)
            )
            raise KeyError(
                trans._(
                    'Colormap "{name}" not found in either vispy or matplotlib. Recognized colormaps are: {colormaps}',
                    deferred=True,
                    name=name,
                    colormaps=', '.join(sorted(f'"{cm}"' for cm in colormaps)),
                )
            ) from e
        mpl_colors = mpl_cmap(np.linspace(0, 1, 256))
        colormap = Colormap(
            name=name, display_name=display_name, colors=mpl_colors
        )

    return colormap


# A dictionary mapping names to VisPy colormap objects
ALL_COLORMAPS = {
    k: vispy_or_mpl_colormap(k) for k in _MATPLOTLIB_COLORMAP_NAMES
}
ALL_COLORMAPS.update(SIMPLE_COLORMAPS)
ALL_COLORMAPS.update(BOP_COLORMAPS)
ALL_COLORMAPS.update(INVERSE_COLORMAPS)

# ... sorted alphabetically by name
AVAILABLE_COLORMAPS = dict(
    sorted(ALL_COLORMAPS.items(), key=lambda cmap: cmap[0].lower())
)
# lock to allow update of AVAILABLE_COLORMAPS in threads
AVAILABLE_COLORMAPS_LOCK = Lock()

# curated colormap sets
# these are selected to look good or at least reasonable when using additive
# blending of multiple channels.
MAGENTA_GREEN = ['magenta', 'green']
RGB = ['red', 'green', 'blue']
CYMRGB = ['cyan', 'yellow', 'magenta', 'red', 'green', 'blue']


AVAILABLE_LABELS_COLORMAPS = {
    'lodisc-50': label_colormap(50),
}


def _increment_unnamed_colormap(
    existing: Iterable[str], name: str = '[unnamed colormap]'
) -> tuple[str, str]:
    """Increment name for unnamed colormap.

    NOTE: this assumes colormaps are *never* deleted, and does not check
          for name collision. If colormaps can ever be removed, please update.

    Parameters
    ----------
    existing : list of str
        Names of existing colormaps.
    name : str, optional
        Name of colormap to be incremented. by default '[unnamed colormap]'

    Returns
    -------
    name : str
        Name of colormap after incrementing.
    display_name : str
        Display name of colormap after incrementing.
    """
    display_name = trans._('[unnamed colormap]')
    if name == '[unnamed colormap]':
        past_names = [n for n in existing if n.startswith('[unnamed colormap')]
        name = f'[unnamed colormap {len(past_names)}]'
        display_name = trans._(
            '[unnamed colormap {number}]',
            number=len(past_names),
        )

    return name, display_name


def ensure_colormap(colormap: ValidColormapArg) -> Colormap:
    """Accept any valid colormap argument, and return Colormap, or raise.

    Adds any new colormaps to AVAILABLE_COLORMAPS in the process, except
    for custom unnamed colormaps created from color values.

    Parameters
    ----------
    colormap : ValidColormapArg
        See ValidColormapArg for supported input types.

    Returns
    -------
    Colormap

    Warns
    -----
    UserWarning
        If ``colormap`` is not a valid colormap argument type.

    Raises
    ------
    KeyError
        If a string is provided that is not in AVAILABLE_COLORMAPS
    TypeError
        If a tuple is provided and the first element is not a string or the
        second element is not a Colormap.
    TypeError
        If a dict is provided and any of the values are not Colormap instances
        or valid inputs to the Colormap constructor.
    """
    with AVAILABLE_COLORMAPS_LOCK:
        if isinstance(colormap, str):
            # when black given as end color, want reversed grayscale colormap
            # from white to black, named gray_r
            if colormap.startswith('#000000') or colormap.lower() == 'black':
                colormap = 'gray_r'

            # Is a colormap with this name already available?
            custom_cmap = AVAILABLE_COLORMAPS.get(colormap)
            if custom_cmap is None:
                name = (
                    colormap.lower() if colormap.startswith('#') else colormap
                )
                custom_cmap = _colormap_from_colors(colormap, name)

                if custom_cmap is None:
                    custom_cmap = vispy_or_mpl_colormap(colormap)

                for cmap_ in AVAILABLE_COLORMAPS.values():
                    if (
                        np.array_equal(cmap_.controls, custom_cmap.controls)
                        and np.array_equal(cmap_.colors, custom_cmap.colors)
                        and cmap_.interpolation == custom_cmap.interpolation
                    ):
                        custom_cmap = cmap_
                        break

            name = custom_cmap.name
            AVAILABLE_COLORMAPS[name] = custom_cmap
        elif isinstance(colormap, Colormap):
            AVAILABLE_COLORMAPS[colormap.name] = colormap
            name = colormap.name
        elif isinstance(colormap, VispyColormap):
            # if a vispy colormap instance is provided, make sure we don't already
            # know about it before adding a new unnamed colormap
            _name = None
            for key, val in AVAILABLE_COLORMAPS.items():
                if colormap == val:
                    _name = key
                    break

            if _name is None:
                name, _display_name = _increment_unnamed_colormap(
                    AVAILABLE_COLORMAPS
                )
            else:
                name = _name

            # Convert from vispy colormap
            cmap = convert_vispy_colormap(colormap, name=name)
            AVAILABLE_COLORMAPS[name] = cmap

        elif isinstance(colormap, tuple):
            if (
                len(colormap) == 2
                and isinstance(colormap[0], str)
                and isinstance(colormap[1], (VispyColormap, Colormap))
            ):
                name = colormap[0]
                cmap = colormap[1]
                # Convert from vispy colormap
                if isinstance(cmap, VispyColormap):
                    cmap = convert_vispy_colormap(cmap, name=name)
                else:
                    cmap.name = name
                AVAILABLE_COLORMAPS[name] = cmap
            else:
                colormap = _colormap_from_colors(colormap)
                if colormap is None:
                    raise TypeError(
                        trans._(
                            'When providing a tuple as a colormap argument, either 1) the first element must be a string and the second a Colormap instance 2) or the tuple should be convertible to one or more colors',
                            deferred=True,
                        )
                    )

                name, _display_name = _increment_unnamed_colormap(
                    AVAILABLE_COLORMAPS
                )
                colormap.update({'name': name, '_display_name': _display_name})
                AVAILABLE_COLORMAPS[name] = colormap

        elif isinstance(colormap, dict):
            if 'colors' in colormap and not (
                isinstance(colormap['colors'], (VispyColormap, Colormap))
            ):
                cmap = Colormap(**colormap)
                name = cmap.name
                AVAILABLE_COLORMAPS[name] = cmap
            elif not all(
                (isinstance(i, (VispyColormap, Colormap)))
                for i in colormap.values()
            ):
                raise TypeError(
                    trans._(
                        'When providing a dict as a colormap, all values must be Colormap instances',
                        deferred=True,
                    )
                )
            else:
                # Convert from vispy colormaps
                for key, cmap in colormap.items():
                    # Convert from vispy colormap
                    if isinstance(cmap, VispyColormap):
                        cmap = convert_vispy_colormap(cmap, name=key)
                    else:
                        cmap.name = key
                    name = key
                    colormap[name] = cmap
                AVAILABLE_COLORMAPS.update(colormap)
                if len(colormap) == 1:
                    name = next(iter(colormap))  # first key in dict
                elif len(colormap) > 1:
                    name = next(iter(colormap.keys()))

                    warnings.warn(
                        trans._(
                            'only the first item in a colormap dict is used as an argument',
                            deferred=True,
                        )
                    )
                else:
                    raise ValueError(
                        trans._(
                            'Received an empty dict as a colormap argument.',
                            deferred=True,
                        )
                    )
        else:
            colormap = _colormap_from_colors(colormap)
            if colormap is not None:
                name, _display_name = _increment_unnamed_colormap(
                    AVAILABLE_COLORMAPS
                )
                colormap.update({'name': name, '_display_name': _display_name})
                AVAILABLE_COLORMAPS[name] = colormap
            else:
                warnings.warn(
                    trans._(
                        'invalid type for colormap: {cm_type}. Must be a {{str, tuple, dict, napari.utils.Colormap, vispy.colors.Colormap}}. Reverting to default',
                        deferred=True,
                        cm_type=type(colormap),
                    )
                )
                # Use default colormap
                name = 'gray'

    return AVAILABLE_COLORMAPS[name]


def _colormap_from_colors(
    colors: ColorType,
    name: Optional[str] = 'custom',
    display_name: Optional[str] = None,
) -> Optional[Colormap]:
    try:
        color_array = transform_color(colors)
    except (ValueError, AttributeError, KeyError):
        return None
    if color_array.shape[0] == 1:
        color_array = np.array([[0, 0, 0, 1], color_array[0]])
    return Colormap(color_array, name=name, display_name=display_name)


def make_default_color_array():
    return np.array([0, 0, 0, 1])


def display_name_to_name(display_name):
    display_name_map = {
        v._display_name: k for k, v in AVAILABLE_COLORMAPS.items()
    }
    return display_name_map.get(
        display_name, next(iter(AVAILABLE_COLORMAPS.keys()))
    )


class CoercedContrastLimits(NamedTuple):
    contrast_limits: tuple[float, float]
    offset: float
    scale: float

    def coerce_data(self, data: np.ndarray) -> np.ndarray:
        if self.scale <= 1:
            return data * self.scale + self.offset

        return (data + self.offset / self.scale) * self.scale


def _coerce_contrast_limits(contrast_limits: tuple[float, float]):
    """Coerce contrast limits to be in the float32 range."""
    if np.abs(contrast_limits).max() > _MAX_VISPY_SUPPORTED_VALUE:
        return scale_down(contrast_limits)

    c_min = np.float32(contrast_limits[0])
    c_max = np.float32(contrast_limits[1])
    dist = c_max - c_min
    if (
        dist < np.abs(np.spacing(c_min)) * _MINIMUM_SHADES_COUNT
        or dist < np.abs(np.spacing(c_max)) * _MINIMUM_SHADES_COUNT
    ):
        return scale_up(contrast_limits)

    return CoercedContrastLimits(contrast_limits, 0, 1)


def scale_down(contrast_limits: tuple[float, float]):
    """Scale down contrast limits to be in the float32 range."""
    scale: float = min(
        1.0,
        (_MAX_VISPY_SUPPORTED_VALUE * 2)
        / (contrast_limits[1] - contrast_limits[0]),
    )
    ctrl_lim = contrast_limits[0] * scale, contrast_limits[1] * scale
    left_shift = max(0.0, -_MAX_VISPY_SUPPORTED_VALUE - ctrl_lim[0])
    right_shift = max(0.0, ctrl_lim[1] - _MAX_VISPY_SUPPORTED_VALUE)
    offset = left_shift - right_shift
    ctrl_lim = (ctrl_lim[0] + offset, ctrl_lim[1] + offset)
    return CoercedContrastLimits(ctrl_lim, offset, scale)


def scale_up(contrast_limits: tuple[float, float]):
    """Scale up contrast limits to be in the float32 precision."""
    scale = 1000 / (contrast_limits[1] - contrast_limits[0])
    shift = -contrast_limits[0] * scale

    return CoercedContrastLimits((0, 1000), shift, scale)
