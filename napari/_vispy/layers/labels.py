from collections import defaultdict
from itertools import product
from math import ceil, isnan, log2, sqrt
from typing import Dict, Optional, Tuple

import numpy as np
from vispy.color import Colormap as VispyColormap
from vispy.gloo import Texture2D
from vispy.scene.node import Node
from vispy.scene.visuals import create_visual_node
from vispy.visuals.image import ImageVisual
from vispy.visuals.shaders import Function, FunctionChain

from napari._vispy.layers.image import ImageLayerNode, VispyImageLayer
from napari._vispy.utils.gl import get_max_texture_sizes
from napari._vispy.visuals.volume import Volume as VolumeNode

PRIME_NUM_TABLE = [
    [61, 59, 53],
    [127, 113, 109],
    [251, 241, 239],
    [509, 503, 499],
    [1021, 1019, 1013],
    [2039, 2029, 2027],
    [4093, 4091, 4079],
    [8191, 8179, 8171],
    [16381, 16369, 16363],
    [32749, 32719, 32717],
    [65521, 65519, 65497],
]

START_TWO_POWER = 6

MAX_LOAD_FACTOR = 0.25

MAX_TEXTURE_SIZE = None

low_disc_lookup_shader = """
uniform sampler2D texture2D_LUT;

vec4 sample_label_color(float t) {
    float phi_mod = 0.6180339887498948482;  // phi - 1
    float value = 0.0;
    float margin = 1.0 / 256;

    if (t == 0) {
        return vec4(0);
    }

    if (($use_selection) && ($selection != t)) {
        return vec4(0);
    }

    value = mod((t * phi_mod + $seed), 1.0) * (1 - 2*margin) + margin;

    return texture2D(
        texture2D_LUT,
        vec2(0.0, clamp(value, 0.0, 1.0))
    );
}
"""


direct_lookup_shader = """
uniform sampler2D texture2D_keys;
uniform sampler2D texture2D_values;
uniform sampler2D texture2D_keys_collision;
uniform sampler2D texture2D_values_collision;
uniform vec2 LUT_shape;
uniform vec2 collision_shape;

vec4 sample_label_color(float t) {
    if ($use_selection) {
        if ($selection == t) {
            return texture2D(
            texture2D_values,
            vec2(0.5, 0.5)
        );
        };
        return vec4(0);
    }

    float empty = 0.;
    // get position in the texture grid (same as hash2d_get)
    vec2 pos = vec2(
        mod(int(t / LUT_shape.y), LUT_shape.x),
        mod(t, LUT_shape.y)
    );

    // add .5 to move to the center of each texel and convert to texture coords
    vec2 pos_tex = (pos + vec2(.5)) / LUT_shape;

    // sample key texture
    float found = texture2D(
        texture2D_keys,
        pos_tex
    ).r;

    // return vec4(pos_tex, 0, 1); // debug if texel is calculated correctly (correct)
    // return vec4(found / 15, 0, 0, 1); // debug if key is calculated correctly (correct, should be a black-to-red gradient)

    // we get a different value:
    // - if it's the empty key, exit;
    // - otherwise, it's a hash collision: continue searching
    float initial_t = t;
    int count = 0;
    if ($collision && (abs(found - initial_t) > 1e-8)) {
        float column = mod(int(t), collision_shape.x);
        for (int i = 0; i < int(collision_shape.y); i++) {
            pos = vec2(column, i);
            pos_tex = (pos + vec2(.5)) / collision_shape;
            found = texture2D(
                texture2D_keys_collision,
                pos_tex
            ).r;
            if (abs(found - initial_t) < 1e-8) {
                return texture2D(
                    texture2D_values_collision,
                    pos_tex
                );
            }
        }
        // not found in collision table
        return vec4(0);
    }

    // return vec4(pos_tex, 0, 1); // debug if final texel is calculated correctly

    vec4 color = vec4(0);
    if (abs(found - empty) > 1e-8) {
        color = texture2D(
            texture2D_values,
            pos_tex
        );
    }
    return color;
}

"""


class LabelVispyColormap(VispyColormap):
    def __init__(
        self,
        colors,
        controls=None,
        seed=0.5,
        use_selection=False,
        selection=0.0,
    ):
        super().__init__(colors, controls, interpolation='zero')
        self.glsl_map = (
            low_disc_lookup_shader.replace('$seed', str(seed))
            .replace('$use_selection', str(use_selection).lower())
            .replace('$selection', str(selection))
        )


class DirectLabelVispyColormap(VispyColormap):
    def __init__(
        self,
        use_selection=False,
        selection=0.0,
        collision=True,
        color_count=1,
    ):
        colors = ['w', 'w']  # dummy values, since we use our own machinery
        super().__init__(colors, controls=None, interpolation='zero')
        self.glsl_map = (
            direct_lookup_shader.replace(
                "$use_selection", str(use_selection).lower()
            )
            .replace("$selection", str(selection))
            .replace("$collision", str(collision).lower())
            .replace("$color_count", str(color_count))
        )


def idx_to_2d(idx, shape):
    """
    From a 1D index generate a 2D index that fits the given shape.

    The 2D index will wrap around line by line and back to the beginning.
    """
    return int((idx // shape[1]) % shape[0]), int(idx % shape[1])


def hash2d_get(key, keys, col_key) -> Tuple[Optional[Tuple[int, int]], bool]:
    """
    Given a key, retrieve its location in the keys table and information if
    it is in the main table or collision table.
    If the key is not found, return None.
    """
    pos = idx_to_2d(key, keys.shape)
    if keys[pos] == key:
        return pos, True
    x = key % col_key.shape[0]
    for i in range(col_key.shape[1]):
        if col_key[x, i] == key:
            return (x, i), False
    return None, False


def hash2d_set(key: float, value, keys, values, empty_val=0) -> bool:
    """
    Set a value in the 2d hashmap, wrapping around to avoid collision.
    """
    if key is None or isnan(key):
        return False
    pos = idx_to_2d(key, keys.shape)
    if keys[pos] == key:
        # casting to float32 and collision with a close key for values above 2**23
        return False
    if keys[pos] == empty_val:
        keys[pos] = key
        values[pos] = value
        return False

    return True


def _get_shape_from_keys(
    keys: np.ndarray, fst_dim: int, snd_dim: int
) -> Optional[Tuple[int, int]]:
    """
    Get the smallest hashmap size without collisions, if any.
    """
    for fst_size, snd_size in product(
        PRIME_NUM_TABLE[fst_dim],
        PRIME_NUM_TABLE[snd_dim],
    ):
        fst_crd = (keys / snd_size) % fst_size
        snd_crd = keys % snd_size

        collision_set = set(zip(fst_crd, snd_crd))
        if len(collision_set) == len(set(keys)):
            return fst_size, snd_size
    return None


def _get_shape_from_dict(
    color_dict: Dict[float, Tuple[float, float, float, float]]
) -> Tuple[int, int]:
    """
    Get the shape of the 2D hashmap from the number of labels.
    The hardcoded shapes use prime numbers designed to avoid collisions.
    As the current collision resolution is non-linear, we decide to
    use hash table of size around four times bigger than the number
    of labels, instead of the classical 1.3 times bigger, to
    minimize the chance of collision in the shader at the cost
    of some video memory.

    We use PRIME_NUM_TABLE to get precomputed prime numbers.
    We decided to use primes close to powers of two, as they
    allowed to keeping fill of hash table between 0.125 to 0.25
    """
    size = len(color_dict) / MAX_LOAD_FACTOR
    size_sqrt = sqrt(size)
    size_log2 = log2(size_sqrt)
    fst_dim = max(int(ceil(size_log2)) - START_TWO_POWER, 0)
    snd_dim = max(int(round(size_log2, 0)) - START_TWO_POWER, 0)
    keys = np.array([x for x in color_dict if x is not None], dtype=np.int64)

    try:
        res = _get_shape_from_keys(keys, fst_dim, snd_dim)
        if res is None:
            res = _get_shape_from_keys(keys, fst_dim, snd_dim + 1)
        if res is None:
            return PRIME_NUM_TABLE[fst_dim][0], PRIME_NUM_TABLE[snd_dim][0]
    except IndexError:
        # Index error means that we have too many labels to fit in 2**16.
        if (max_size := PRIME_NUM_TABLE[-1][0] ** 2) < len(color_dict):
            raise OverflowError(
                f'Too many labels. We support maximally {max_size} labels'
            ) from None
        return PRIME_NUM_TABLE[-1][0], PRIME_NUM_TABLE[-1][0]
    return res


def get_shape_from_dict(color_dict):
    global MAX_TEXTURE_SIZE
    if MAX_TEXTURE_SIZE is None:
        MAX_TEXTURE_SIZE = get_max_texture_sizes()[0]

    shape = _get_shape_from_dict(color_dict)

    if MAX_TEXTURE_SIZE is not None and (
        shape[0] > MAX_TEXTURE_SIZE or shape[1] > MAX_TEXTURE_SIZE
    ):
        raise MemoryError(
            f'Too many labels. GPU does not support textures of this size.'
            f' Requested size is {shape[0]}x{shape[1]}, but maximum supported'
            f' size is {MAX_TEXTURE_SIZE}x{MAX_TEXTURE_SIZE}'
        )
    return shape


def _build_collision_table(
    dkt: Dict[float, Tuple[float, float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a secondary array for resolving hash collision.
    Using a separate array allow to reducing number of collisions in the main table as
    a collided key cannot generate new collision.
    """
    if not dkt:
        return np.zeros((1, 1), dtype=np.float32), np.zeros(
            (1, 1, 4), dtype=np.float32
        )

    table_size_index = max(int(ceil(log2(len(dkt)))) - START_TWO_POWER, 0)
    collision_count = len(dkt) + 10
    selected_table_size = PRIME_NUM_TABLE[table_size_index][0]
    keys = np.array(list(dkt.keys()), dtype=np.float32)
    for table_size in PRIME_NUM_TABLE[table_size_index]:
        collision_count_ = len(dkt) - len(set(keys % table_size))
        if collision_count_ < collision_count:
            collision_count = collision_count_
            selected_table_size = table_size
    mapping_dict = defaultdict(list)

    for key, value in dkt.items():
        mapping_dict[key % selected_table_size].append((key, value))

    second_dim = max(len(x) for x in mapping_dict.values())

    keys_array = np.zeros((selected_table_size, second_dim), dtype=np.float32)
    values_array = np.zeros(
        (selected_table_size, second_dim, 4), dtype=np.float32
    )
    for key, value in mapping_dict.items():
        if isnan(key):
            continue
        for i, (key_, value_) in enumerate(value):
            keys_array[int(key), i] = key_
            values_array[int(key), i] = value_

    return keys_array, values_array


def build_textures_from_dict(
    color_dict, empty_val=0, shape=None, use_selection=False, selection=0.0
):
    if use_selection:
        keys = np.full((1, 1), selection, dtype=np.float32)
        values = np.zeros((1, 1, 4), dtype=np.float32)
        values[0, 0] = color_dict[selection]
        return keys, values, keys, values, False

    if len(color_dict) > 2**31 - 2:
        raise OverflowError(
            f'Too many labels ({len(color_dict)}). Maximum supported number of labels is 2^31-2'
        )

    if shape is None:
        shape = get_shape_from_dict(color_dict)

    if len(color_dict) > shape[0] * shape[1]:
        raise OverflowError(
            f'Too many labels ({len(color_dict)}). Maximum supported number of labels for the given shape is {shape[0] * shape[1]}'
        )

    keys = np.full(shape, empty_val, dtype=np.float32)
    values = np.zeros(shape + (4,), dtype=np.float32)
    collided = set()
    collision_dict = {}
    for key, value in color_dict.items():
        key = np.float32(key)
        if key in collided:
            continue
        collided.add(key)
        if hash2d_set(key, value, keys, values):
            collision_dict[key] = value

    collision_keys, collision_values = _build_collision_table(collision_dict)

    return keys, values, collision_keys, collision_values, bool(collision_dict)


class VispyLabelsLayer(VispyImageLayer):
    def __init__(self, layer, node=None, texture_format='r32f') -> None:
        super().__init__(
            layer,
            node=node,
            texture_format=texture_format,
            layer_node_class=LabelLayerNode,
        )

        # self.layer.events.color_mode.connect(self._on_colormap_change)
        self.layer.events.labels_update.connect(self._on_partial_labels_update)
        self.layer.events.selected_label.connect(self._on_colormap_change)
        self.layer.events.show_selected_label.connect(self._on_colormap_change)

    def _on_rendering_change(self):
        # overriding the Image method so we can maintain the same old rendering name
        if isinstance(self.node, VolumeNode):
            rendering = self.layer.rendering
            self.node.method = (
                rendering
                if rendering != 'translucent'
                else 'translucent_categorical'
            )
            self._on_attenuation_change()
            self._on_iso_threshold_change()

    def _on_colormap_change(self, event=None):
        # self.layer.colormap is a labels_colormap, which is an evented model
        # from napari.utils.colormaps.Colormap (or similar). If we use it
        # in our constructor, we have access to the texture data we need
        if (
            event is not None
            and event.type == 'selected_label'
            and not self.layer.show_selected_label
        ):
            return
        colormap = self.layer.colormap
        mode = self.layer.color_mode

        if mode == 'auto':
            self.node.cmap = LabelVispyColormap(
                colors=colormap.colors,
                controls=colormap.controls,
                seed=colormap.seed,
                use_selection=colormap.use_selection,
                selection=colormap.selection,
            )
        elif mode == 'direct':
            color_dict = (
                self.layer.color
            )  # TODO: should probably account for non-given labels
            (
                key_texture,
                val_texture,
                key_collision,
                val_collision,
                collision,
            ) = build_textures_from_dict(
                color_dict,
                use_selection=colormap.use_selection,
                selection=colormap.selection,
            )

            self.node.cmap = DirectLabelVispyColormap(
                use_selection=colormap.use_selection,
                selection=colormap.selection,
                collision=collision,
                color_count=len(color_dict) + 1,
            )
            # note that textures have to be transposed here!
            self.node.shared_program['texture2D_keys'] = Texture2D(
                key_texture.T, internalformat='r32f', interpolation='nearest'
            )
            self.node.shared_program['texture2D_values'] = Texture2D(
                val_texture.swapaxes(0, 1),
                internalformat='rgba32f',
                interpolation='nearest',
            )
            self.node.shared_program['texture2D_keys_collision'] = Texture2D(
                key_collision.T, internalformat='r32f', interpolation='nearest'
            )
            self.node.shared_program['texture2D_values_collision'] = Texture2D(
                val_collision.swapaxes(0, 1),
                internalformat='rgba32f',
                interpolation='nearest',
            )
            self.node.shared_program['LUT_shape'] = key_texture.shape
            self.node.shared_program['collision_shape'] = key_collision.shape
        else:
            self.node.cmap = VispyColormap(*colormap)

    def _on_partial_labels_update(self, event):
        if not self.layer.loaded:
            return

        raw_displayed = self.layer._slice.image.raw
        ndims = len(event.offset)

        if self.node._texture.shape[:ndims] != raw_displayed.shape[:ndims]:
            self.layer.refresh()
            return

        self.node._texture.scale_and_set_data(
            event.data, copy=False, offset=event.offset
        )
        self.node.update()


class LabelVisual(ImageVisual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_color_transform(self):
        fun = FunctionChain(
            None,
            [
                Function(self._func_templates['red_to_luminance']),
                Function(self.cmap.glsl_map),
            ],
        )
        return fun


class LabelLayerNode(ImageLayerNode):
    def __init__(self, custom_node: Node = None, texture_format=None):
        self._custom_node = custom_node
        self._image_node = LabelNode(
            None
            if (texture_format is None or texture_format == 'auto')
            else np.array([[0.0]], dtype=np.float32),
            method='auto',
            texture_format=texture_format,
        )

        self._volume_node = VolumeNode(
            np.zeros((1, 1, 1), dtype=np.float32),
            clim=[0, 2**23 - 1],
            texture_format=texture_format,
        )


BaseLabel = create_visual_node(LabelVisual)


class LabelNode(BaseLabel):  # type: ignore [valid-type,misc]
    def _compute_bounds(self, axis, view):
        if self._data is None:
            return None
        elif axis > 1:  # noqa: RET505
            return 0, 0
        else:
            return 0, self.size[axis]
