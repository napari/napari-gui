from collections import OrderedDict
from enum import auto

from napari.utils.misc import StringEnum
from napari.utils.translations import trans


class ColorMode(StringEnum):
    """
    ColorMode: Color setting mode.

    DIRECT (default mode) allows each point to be set arbitrarily

    CYCLE allows the color to be set via a color cycle over an attribute

    COLORMAP allows color to be set via a color map over an attribute
    """

    DIRECT = auto()
    CYCLE = auto()
    COLORMAP = auto()


class Mode(StringEnum):
    """
    Mode: Interactive mode. The normal, default mode is PAN_ZOOM, which
    allows for normal interactivity with the canvas.

    ADD allows points to be added by clicking

    SELECT allows the user to select points by clicking on them
    """

    PAN_ZOOM = auto()
    TRANSFORM = auto()
    ADD = auto()
    SELECT = auto()


class Symbol(StringEnum):
    """Valid symbol/marker types for the Points layer."""

    ARROW = auto()
    CLOBBER = auto()
    CROSS = auto()
    DIAMOND = auto()
    DISC = auto()
    HBAR = auto()
    RING = auto()
    SQUARE = auto()
    STAR = auto()
    TAILED_ARROW = auto()
    TRIANGLE_DOWN = auto()
    TRIANGLE_UP = auto()
    VBAR = auto()
    X = auto()


# Mapping of symbol alias names to the deduplicated name
SYMBOL_ALIAS = {
    '>': Symbol.ARROW,
    '+': Symbol.CROSS,
    'o': Symbol.DISC,
    '-': Symbol.HBAR,
    's': Symbol.SQUARE,
    '*': Symbol.STAR,
    '->': Symbol.TAILED_ARROW,
    'v': Symbol.TRIANGLE_DOWN,
    '^': Symbol.TRIANGLE_UP,
    '|': Symbol.VBAR,
}

SYMBOL_TRANSLATION = OrderedDict(
    [
        (Symbol.ARROW, trans._('arrow')),
        (Symbol.CLOBBER, trans._('clobber')),
        (Symbol.CROSS, trans._('cross')),
        (Symbol.DIAMOND, trans._('diamond')),
        (Symbol.DISC, trans._('disc')),
        (Symbol.HBAR, trans._('hbar')),
        (Symbol.RING, trans._('ring')),
        (Symbol.SQUARE, trans._('square')),
        (Symbol.STAR, trans._('star')),
        (Symbol.TAILED_ARROW, trans._('tailed arrow')),
        (Symbol.TRIANGLE_DOWN, trans._('triangle down')),
        (Symbol.TRIANGLE_UP, trans._('triangle up')),
        (Symbol.VBAR, trans._('vbar')),
        (Symbol.X, trans._('x')),
    ]
)

SYMBOL_TRANSLATION_INVERTED = {v: k for k, v in SYMBOL_TRANSLATION.items()}


SYMBOL_DICT: dict[str | Symbol, Symbol] = {x: x for x in Symbol}
SYMBOL_DICT.update({str(x): x for x in Symbol})
SYMBOL_DICT.update(SYMBOL_TRANSLATION_INVERTED)  # type: ignore[arg-type]
SYMBOL_DICT.update(SYMBOL_ALIAS)  # type: ignore[arg-type]


class Shading(StringEnum):
    """Shading: Shading mode for the points.

    NONE no shading is applied.
    SPHERICAL shading and depth buffer are modified to mimic a 3D object with spherical shape
    """

    NONE = auto()
    SPHERICAL = auto()


SHADING_TRANSLATION = {
    trans._('none'): Shading.NONE,
    trans._('spherical'): Shading.SPHERICAL,
}


class PointsProjectionMode(StringEnum):
    """
    Projection mode for aggregating a thick nD slice onto displayed dimensions.

        * NONE: ignore slice thickness, only using the dims point
        * ALL: project all points in the slice onto displayed dimensions
    """

    NONE = auto()
    ALL = auto()
