from collections import OrderedDict
from enum import auto

from ...utils.misc import StringEnum
from ...utils.translations import trans


class Blending(StringEnum):
    """BLENDING: Blending mode for the layer.

    Selects a preset blending mode in vispy that determines how
            RGB and alpha values get mixed.
            Blending.OPAQUE
                Allows for only the top layer to be visible and corresponds to
                depth_test=True, cull_face=False, blend=False.
            Blending.TRANSLUCENT
                Allows for multiple layers to be blended with different opacity
                and corresponds to depth_test=True, cull_face=False,
                blend=True, blend_func=('src_alpha', 'one_minus_src_alpha').
            Blending.TRANSLUCENT_NO_DEPTH
                Allows for multiple layers to be blended with different opacity,
                but no depth testing is performed.
                and corresponds to depth_test=False, cull_face=False,
                blend=True, blend_func=('src_alpha', 'one_minus_src_alpha').
            Blending.ADDITIVE
                Allows for multiple layers to be blended together with
                different colors and opacity. Useful for creating overlays. It
                corresponds to depth_test=False, cull_face=False, blend=True,
                blend_func=('src_alpha', 'one').
            Blending.SUBTRACTIVE
                Allows for multiple layers to be blended together with
                different colors and opacity on a white background. This
                uses a subtraction blending function, so when used with a
                white background, colors will be "inverted" (i.e., 1-color
                where color is the RGB value). It corresponds to depth_test=False,
                cull_face=False, blend=True, blend_func=('src_alpha', 'one'), and
                blend_equation=('func_reverse_subtract').
    """

    TRANSLUCENT = auto()
    TRANSLUCENT_NO_DEPTH = auto()
    ADDITIVE = auto()
    SUBTRACTIVE = auto()
    OPAQUE = auto()


BLENDING_TRANSLATIONS = OrderedDict(
    [
        (Blending.TRANSLUCENT, trans._("translucent")),
        (Blending.TRANSLUCENT_NO_DEPTH, trans._("translucent_no_depth")),
        (Blending.ADDITIVE, trans._("additive")),
        (Blending.SUBTRACTIVE, trans._("subtractive")),
        (Blending.OPAQUE, trans._("opaque")),
    ]
)
