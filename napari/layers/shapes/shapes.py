import warnings
from contextlib import contextmanager
from copy import copy, deepcopy
from itertools import cycle
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from vispy.color import get_color_names

from napari.layers.base import Layer, no_op
from napari.layers.base._base_constants import ActionType
from napari.layers.base._base_mouse_bindings import (
    highlight_box_handles,
    transform_with_box,
)
from napari.layers.shapes._shape_list import ShapeList
from napari.layers.shapes._shapes_constants import (
    Box,
    ColorMode,
    Mode,
    ShapeType,
    shape_classes,
)
from napari.layers.shapes._shapes_models import Ellipse, Polygon, Rectangle
from napari.layers.shapes._shapes_mouse_bindings import (
    add_ellipse,
    add_line,
    add_path_polygon,
    add_path_polygon_lasso,
    add_rectangle,
    finish_drawing_shape,
    highlight,
    polygon_creating,
    select,
    vertex_insert,
    vertex_remove,
)
from napari.layers.shapes._shapes_utils import (
    create_box,
    extract_shape_type,
    get_default_shape_type,
    get_shape_ndim,
    number_of_shapes,
    rdp,
    validate_num_vertices,
)
from napari.layers.utils.color_manager_utils import (
    guess_continuous,
    map_property,
)
from napari.layers.utils.color_transformations import (
    normalize_and_broadcast_colors,
    transform_color_cycle,
    transform_color_with_defaults,
)
from napari.layers.utils.interactivity_utils import (
    nd_line_segment_to_displayed_data_ray,
)
from napari.layers.utils.layer_utils import _FeatureTable, _unique_element
from napari.layers.utils.text_manager import TextManager
from napari.settings import get_settings
from napari.utils.colormaps import Colormap, ValidColormapArg, ensure_colormap
from napari.utils.colormaps.colormap_utils import ColorType
from napari.utils.colormaps.standardize_color import (
    hex_to_name,
    rgb_to_hex,
    transform_color,
)
from napari.utils.events import Event
from napari.utils.events.custom_types import Array
from napari.utils.misc import ensure_iterable
from napari.utils.translations import trans

DEFAULT_COLOR_CYCLE = np.array([[1, 0, 1, 1], [0, 1, 0, 1]])


class Shapes(Layer):
    """Shapes layer.

    Parameters
    ----------
    data : list or array
        List of shape data, where each element is an (N, D) array of the
        N vertices of a shape in D dimensions. Can be an 3-dimensional
        array if each shape has the same number of vertices.
    ndim : int
        Number of dimensions for shapes. When data is not None, ndim must be D.
        An empty shapes layer can be instantiated with arbitrary ndim.
    features : dict[str, array-like] or Dataframe-like
        Features table where each row corresponds to a shape and each column
        is a feature.
    feature_defaults : dict[str, Any] or Dataframe-like
        The default value of each feature in a table with one row.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each shape. Each property should be an array of length N,
        where N is the number of shapes.
    property_choices : dict {str: array (N,)}
        possible values for each property.
    text : str, dict
        Text to be displayed with the shapes. If text is set to a key in properties,
        the value of that property will be displayed. Multiple properties can be
        composed using f-string-like syntax (e.g., '{property_1}, {float_property:.2f}).
        A dictionary can be provided with keyword arguments to set the text values
        and display properties. See TextManager.__init__() for the valid keyword arguments.
        For example usage, see /napari/examples/add_shapes_with_text.py.
    shape_type : string or list
        String of shape shape_type, must be one of "{'line', 'rectangle',
        'ellipse', 'path', 'polygon'}". If a list is supplied it must be
        the same length as the length of `data` and each element will be
        applied to each shape otherwise the same value will be used for all
        shapes.
    edge_width : float or list
        Thickness of lines and edges. If a list is supplied it must be the
        same length as the length of `data` and each element will be
        applied to each shape otherwise the same value will be used for all
        shapes.
    edge_color : str, array-like
        If string can be any color name recognized by vispy or hex value if
        starting with `#`. If array-like must be 1-dimensional array with 3
        or 4 elements. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each shape
        otherwise the same value will be used for all shapes.
    edge_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to edge_color if a
        categorical attribute is used color the vectors.
    edge_colormap : str, napari.utils.Colormap
        Colormap to set edge_color if a continuous attribute is used to set face_color.
    edge_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    face_color : str, array-like
        If string can be any color name recognized by vispy or hex value if
        starting with `#`. If array-like must be 1-dimensional array with 3
        or 4 elements. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each shape
        otherwise the same value will be used for all shapes.
    face_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to face_color if a
        categorical attribute is used color the vectors.
    face_colormap : str, napari.utils.Colormap
        Colormap to set face_color if a continuous attribute is used to set face_color.
    face_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    z_index : int or list
        Specifier of z order priority. Shapes with higher z order are
        displayed ontop of others. If a list is supplied it must be the
        same length as the length of `data` and each element will be
        applied to each shape otherwise the same value will be used for all
        shapes.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.

    Attributes
    ----------
    data : (N, ) list of array
        List of shape data, where each element is an (N, D) array of the
        N vertices of a shape in D dimensions.
    features : Dataframe-like
        Features table where each row corresponds to a shape and each column
        is a feature.
    feature_defaults : DataFrame-like
        Stores the default value of each feature in a table with one row.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each shape. Each property should be an array of length N,
        where N is the number of shapes.
    text : str, dict
        Text to be displayed with the shapes. If text is set to a key in properties,
        the value of that property will be displayed. Multiple properties can be
        composed using f-string-like syntax (e.g., '{property_1}, {float_property:.2f}).
        For example usage, see /napari/examples/add_shapes_with_text.py.
    shape_type : (N, ) list of str
        Name of shape type for each shape.
    edge_color : str, array-like
        Color of the shape border. Numeric color values should be RGB(A).
    face_color : str, array-like
        Color of the shape face. Numeric color values should be RGB(A).
    edge_width : (N, ) list of float
        Edge width for each shape.
    z_index : (N, ) list of int
        z-index for each shape.
    current_edge_width : float
        Thickness of lines and edges of the next shape to be added or the
        currently selected shape.
    current_edge_color : str
        Color of the edge of the next shape to be added or the currently
        selected shape.
    current_face_color : str
        Color of the face of the next shape to be added or the currently
        selected shape.
    selected_data : set
        List of currently selected shapes.
    nshapes : int
        Total number of shapes.
    mode : Mode
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        The SELECT mode allows for entire shapes to be selected, moved and
        resized.

        The DIRECT mode allows for shapes to be selected and their individual
        vertices to be moved.

        The VERTEX_INSERT and VERTEX_REMOVE modes allow for individual
        vertices either to be added to or removed from shapes that are already
        selected. Note that shapes cannot be selected in this mode.

        The ADD_RECTANGLE, ADD_ELLIPSE, ADD_LINE, ADD_PATH, and ADD_POLYGON
        modes all allow for their corresponding shape type to be added.

    Notes
    -----
    _data_dict : Dict of ShapeList
        Dictionary containing all the shape data indexed by slice tuple
    _data_view : ShapeList
        Object containing the currently viewed shape data.
    _selected_data_history : set
        Set of currently selected captured on press of <space>.
    _selected_data_stored : set
        Set of selected previously displayed. Used to prevent rerendering the
        same highlighted shapes when no data has changed.
    _selected_box : None | np.ndarray
        `None` if no shapes are selected, otherwise a 10x2 array of vertices of
        the interaction box. The first 8 points are the corners and midpoints
        of the box. The 9th point is the center of the box, and the last point
        is the location of the rotation handle that can be used to rotate the
        box.
    _drag_start : None | np.ndarray
        If a drag has been started and is in progress then a length 2 array of
        the initial coordinates of the drag. `None` otherwise.
    _drag_box : None | np.ndarray
        If a drag box is being created to select shapes then this is a 2x2
        array of the two extreme corners of the drag. `None` otherwise.
    _drag_box_stored : None | np.ndarray
        If a drag box is being created to select shapes then this is a 2x2
        array of the two extreme corners of the drag that have previously been
        rendered. `None` otherwise. Used to prevent rerendering the same
        drag box when no data has changed.
    _is_moving : bool
        Bool indicating if any shapes are currently being moved.
    _is_selecting : bool
        Bool indicating if a drag box is currently being created in order to
        select shapes.
    _is_creating : bool
        Bool indicating if any shapes are currently being created.
    _fixed_aspect : bool
        Bool indicating if aspect ratio of shapes should be preserved on
        resizing.
    _aspect_ratio : float
        Value of aspect ratio to be preserved if `_fixed_aspect` is `True`.
    _fixed_vertex : None | np.ndarray
        If a scaling or rotation is in progress then a length 2 array of the
        coordinates that are remaining fixed during the move. `None` otherwise.
    _fixed_index : int
        If a scaling or rotation is in progress then the index of the vertex of
        the bounding box that is remaining fixed during the move. `None`
        otherwise.
    _update_properties : bool
        Bool indicating if properties are to allowed to update the selected
        shapes when they are changed. Blocking this prevents circular loops
        when shapes are selected and the properties are changed based on that
        selection
    _allow_thumbnail_update : bool
        Flag set to true to allow the thumbnail to be updated. Blocking the thumbnail
        can be advantageous where responsiveness is critical.
    _clipboard : dict
        Dict of shape objects that are to be used during a copy and paste.
    _colors : list
        List of supported vispy color names.
    _vertex_size : float
        Size of the vertices of the shapes and bounding box in Canvas
        coordinates.
    _rotation_handle_length : float
        Length of the rotation handle of the bounding box in Canvas
        coordinates.
    _input_ndim : int
        Dimensions of shape data.
    _thumbnail_update_thresh : int
        If there are more than this number of shapes, the thumbnail
        won't update during interactive events
    """

    _modeclass = Mode
    _colors = get_color_names()
    _vertex_size = 10
    _rotation_handle_length = 20
    _highlight_color = (0, 0.6, 1)
    _highlight_width = 1.5

    _face_color_property: str
    _edge_color_property: str
    _face_color_cycle: npt.NDArray
    _edge_color_cycle: npt.NDArray
    _face_color_cycle_values: npt.NDArray
    _edge_color_cycle_values: npt.NDArray
    _face_color_mode: str
    _edge_color_mode: str

    # If more shapes are present then they are randomly subsampled
    # in the thumbnail
    _max_shapes_thumbnail = 100

    _drag_modes: ClassVar[Dict[Mode, Callable[["Shapes", Event], Any]]] = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: transform_with_box,
        Mode.SELECT: select,
        Mode.DIRECT: select,
        Mode.VERTEX_INSERT: vertex_insert,
        Mode.VERTEX_REMOVE: vertex_remove,
        Mode.ADD_RECTANGLE: add_rectangle,
        Mode.ADD_ELLIPSE: add_ellipse,
        Mode.ADD_LINE: add_line,
        Mode.ADD_PATH: add_path_polygon,
        Mode.ADD_POLYGON: add_path_polygon,
        Mode.ADD_POLYGON_LASSO: add_path_polygon_lasso,
    }

    _move_modes: ClassVar[Dict[Mode, Callable[["Shapes", Event], Any]]] = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: highlight_box_handles,
        Mode.SELECT: highlight,
        Mode.DIRECT: highlight,
        Mode.VERTEX_INSERT: highlight,
        Mode.VERTEX_REMOVE: highlight,
        Mode.ADD_RECTANGLE: no_op,
        Mode.ADD_ELLIPSE: no_op,
        Mode.ADD_LINE: no_op,
        Mode.ADD_PATH: polygon_creating,
        Mode.ADD_POLYGON: polygon_creating,
        Mode.ADD_POLYGON_LASSO: polygon_creating,
    }

    _double_click_modes: ClassVar[
        Dict[Mode, Callable[["Shapes", Event], Any]]
    ] = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: no_op,
        Mode.SELECT: no_op,
        Mode.DIRECT: no_op,
        Mode.VERTEX_INSERT: no_op,
        Mode.VERTEX_REMOVE: no_op,
        Mode.ADD_RECTANGLE: no_op,
        Mode.ADD_ELLIPSE: no_op,
        Mode.ADD_LINE: no_op,
        Mode.ADD_PATH: finish_drawing_shape,
        Mode.ADD_POLYGON: finish_drawing_shape,
        Mode.ADD_POLYGON_LASSO: no_op,
    }

    _cursor_modes: ClassVar[Dict[Mode, str]] = {
        Mode.PAN_ZOOM: 'standard',
        Mode.TRANSFORM: 'standard',
        Mode.SELECT: 'pointing',
        Mode.DIRECT: 'pointing',
        Mode.VERTEX_INSERT: 'cross',
        Mode.VERTEX_REMOVE: 'cross',
        Mode.ADD_RECTANGLE: 'cross',
        Mode.ADD_ELLIPSE: 'cross',
        Mode.ADD_LINE: 'cross',
        Mode.ADD_PATH: 'cross',
        Mode.ADD_POLYGON: 'cross',
        Mode.ADD_POLYGON_LASSO: 'cross',
    }

    _interactive_modes: ClassVar[Set[Mode]] = {
        Mode.PAN_ZOOM,
    }

    def __init__(
        self,
        data=None,
        *,
        ndim=None,
        features=None,
        feature_defaults=None,
        properties=None,
        property_choices=None,
        text=None,
        shape_type='rectangle',
        edge_width=1,
        edge_color='#777777',
        edge_color_cycle=None,
        edge_colormap='viridis',
        edge_contrast_limits=None,
        face_color='white',
        face_color_cycle=None,
        face_colormap='viridis',
        face_contrast_limits=None,
        z_index=0,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=0.7,
        blending='translucent',
        visible=True,
        cache=True,
        experimental_clipping_planes=None,
        projection_mode='none',
    ) -> None:
        if data is None or len(data) == 0:
            if ndim is None:
                ndim = 2
            data = np.empty((0, 0, ndim))
        else:
            data, shape_type = extract_shape_type(data, shape_type)
            data_ndim = get_shape_ndim(data)
            if ndim is not None and ndim != data_ndim:
                raise ValueError(
                    trans._(
                        "Shape dimensions must be equal to ndim",
                        deferred=True,
                    )
                )
            ndim = data_ndim

        super().__init__(
            data,
            ndim=ndim,
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
            cache=cache,
            experimental_clipping_planes=experimental_clipping_planes,
            projection_mode=projection_mode,
        )

        self.events.add(
            edge_width=Event,
            edge_color=Event,
            face_color=Event,
            properties=Event,
            current_edge_color=Event,
            current_face_color=Event,
            current_properties=Event,
            highlight=Event,
            features=Event,
            feature_defaults=Event,
        )

        # Flag set to false to block thumbnail refresh
        self._allow_thumbnail_update = True

        self._display_order_stored = []
        self._ndisplay_stored = self._slice_input.ndisplay

        self._feature_table = _FeatureTable.from_layer(
            features=features,
            feature_defaults=feature_defaults,
            properties=properties,
            property_choices=property_choices,
            num_data=number_of_shapes(data),
        )

        # The following shape properties are for the new shapes that will
        # be drawn. Each shape has a corresponding property with the
        # value for itself
        if np.isscalar(edge_width):
            self._current_edge_width = edge_width
        else:
            self._current_edge_width = 1

        self._data_view = ShapeList(ndisplay=self._slice_input.ndisplay)
        self._data_view.slice_key = np.array(self._data_slice.point)[
            self._slice_input.not_displayed
        ]

        self._value = (None, None)
        self._value_stored = (None, None)
        self._moving_value: Tuple[Optional[int], Optional[int]] = (None, None)
        self._selected_data = set()
        self._selected_data_stored = set()
        self._selected_data_history = set()
        self._selected_box = None
        self._last_cursor_position = None

        self._drag_start = None
        self._fixed_vertex = None
        self._fixed_aspect = False
        self._aspect_ratio = 1
        self._is_moving = False

        # _moving_coordinates are needed for fixing aspect ratio during
        # a resize, it stores the last pointer coordinate value that happened
        # during a mouse move to that pressing/releasing shift
        # can trigger a redraw of the shape with a fixed aspect ratio.
        self._moving_coordinates = None

        self._fixed_index = 0
        self._is_selecting = False
        self._drag_box = None
        self._drag_box_stored = None
        self._is_creating = False
        self._clipboard: Dict[str, Shapes] = {}

        self._status = self.mode

        self._init_shapes(
            data,
            shape_type=shape_type,
            edge_width=edge_width,
            edge_color=edge_color,
            edge_color_cycle=edge_color_cycle,
            edge_colormap=edge_colormap,
            edge_contrast_limits=edge_contrast_limits,
            face_color=face_color,
            face_color_cycle=face_color_cycle,
            face_colormap=face_colormap,
            face_contrast_limits=face_contrast_limits,
            z_index=z_index,
        )

        # set the current_* properties
        if len(data) > 0:
            self._current_edge_color = self.edge_color[-1]
            self._current_face_color = self.face_color[-1]
        elif len(data) == 0 and len(self.properties) > 0:
            self._initialize_current_color_for_empty_layer(edge_color, 'edge')
            self._initialize_current_color_for_empty_layer(face_color, 'face')
        elif len(data) == 0 and len(self.properties) == 0:
            self._current_edge_color = transform_color_with_defaults(
                num_entries=1,
                colors=edge_color,
                elem_name="edge_color",
                default="black",
            )
            self._current_face_color = transform_color_with_defaults(
                num_entries=1,
                colors=face_color,
                elem_name="face_color",
                default="black",
            )

        self._text = TextManager._from_layer(
            text=text,
            features=self.features,
        )

        # Trigger generation of view slice and thumbnail
        self.refresh()

    def _initialize_current_color_for_empty_layer(
        self, color: ColorType, attribute: str
    ):
        """Initialize current_{edge,face}_color when starting with empty layer.

        Parameters
        ----------
        color : (N, 4) array or str
            The value for setting edge or face_color
        attribute : str in {'edge', 'face'}
            The name of the attribute to set the color of.
            Should be 'edge' for edge_color or 'face' for face_color.
        """
        color_mode = getattr(self, f'_{attribute}_color_mode')
        if color_mode == ColorMode.DIRECT:
            curr_color = transform_color_with_defaults(
                num_entries=1,
                colors=color,
                elem_name=f'{attribute}_color',
                default="white",
            )

        elif color_mode == ColorMode.CYCLE:
            color_cycle = getattr(self, f'_{attribute}_color_cycle')
            curr_color = transform_color(next(color_cycle))

            # add the new color cycle mapping
            color_property = getattr(self, f'_{attribute}_color_property')
            prop_value = self.feature_defaults[color_property][0]
            color_cycle_map = getattr(self, f'{attribute}_color_cycle_map')
            color_cycle_map[prop_value] = np.squeeze(curr_color)
            setattr(self, f'{attribute}_color_cycle_map', color_cycle_map)

        elif color_mode == ColorMode.COLORMAP:
            color_property = getattr(self, f'_{attribute}_color_property')
            prop_value = self.feature_defaults[color_property][0]
            colormap = getattr(self, f'{attribute}_colormap')
            contrast_limits = getattr(self, f'_{attribute}_contrast_limits')
            curr_color, _ = map_property(
                prop=prop_value,
                colormap=colormap,
                contrast_limits=contrast_limits,
            )
        setattr(self, f'_current_{attribute}_color', curr_color)

    @property
    def data(self):
        """list: Each element is an (N, D) array of the vertices of a shape."""
        return self._data_view.data

    @data.setter
    def data(self, data):
        self._finish_drawing()
        prior_data = len(self.data) > 0
        data, shape_type = extract_shape_type(data)
        n_new_shapes = number_of_shapes(data)
        # not given a shape_type through data
        if shape_type is None:
            shape_type = self.shape_type

        edge_widths = self._data_view.edge_widths
        edge_color = self._data_view.edge_color
        face_color = self._data_view.face_color
        z_indices = self._data_view.z_indices

        # fewer shapes, trim attributes
        if self.nshapes > n_new_shapes:
            shape_type = shape_type[:n_new_shapes]
            edge_widths = edge_widths[:n_new_shapes]
            z_indices = z_indices[:n_new_shapes]
            edge_color = edge_color[:n_new_shapes]
            face_color = face_color[:n_new_shapes]
        # more shapes, add attributes
        elif self.nshapes < n_new_shapes:
            n_shapes_difference = n_new_shapes - self.nshapes
            shape_type = (
                shape_type
                + [get_default_shape_type(shape_type)] * n_shapes_difference
            )
            edge_widths = edge_widths + [1] * n_shapes_difference
            z_indices = z_indices + [0] * n_shapes_difference
            edge_color = np.concatenate(
                (
                    edge_color,
                    self._get_new_shape_color(n_shapes_difference, 'edge'),
                )
            )
            face_color = np.concatenate(
                (
                    face_color,
                    self._get_new_shape_color(n_shapes_difference, 'face'),
                )
            )
        data_not_empty = (
            data is not None
            and (isinstance(data, np.ndarray) and data.size > 0)
            or (isinstance(data, list) and len(data) > 0)
        )
        kwargs = {
            "value": self.data,
            "vertex_indices": ((),),
            "data_indices": tuple(i for i in range(len(self.data))),
        }
        if prior_data and data_not_empty:
            kwargs["action"] = ActionType.CHANGING
        elif data_not_empty:
            kwargs["action"] = ActionType.ADDING
            kwargs["data_indices"] = tuple(i for i in range(len(data)))
        else:
            kwargs["action"] = ActionType.REMOVING

        self.events.data(**kwargs)
        self._data_view = ShapeList(ndisplay=self._slice_input.ndisplay)
        self._data_view.slice_key = np.array(self._data_slice.point)[
            self._slice_input.not_displayed
        ]
        self._add_shapes(
            data,
            shape_type=shape_type,
            edge_width=edge_widths,
            edge_color=edge_color,
            face_color=face_color,
            z_index=z_indices,
            n_new_shapes=n_new_shapes,
        )
        self._update_dims()

        kwargs["data_indices"] = tuple(i for i in range(len(data)))
        kwargs["value"] = self.data
        if prior_data and data_not_empty:
            kwargs["action"] = ActionType.CHANGED
        elif data_not_empty:
            kwargs["action"] = ActionType.ADDED
        else:
            kwargs["action"] = ActionType.REMOVED
        self.events.data(**kwargs)

        self._reset_editable()

    def _on_selection(self, selected: bool):
        # this method is slated for removal.  don't add anything new.
        if not selected:
            self._finish_drawing()

    @property
    def features(self):
        """Dataframe-like features table.

        It is an implementation detail that this is a `pandas.DataFrame`. In the future,
        we will target the currently-in-development Data API dataframe protocol [1].
        This will enable us to use alternate libraries such as xarray or cuDF for
        additional features without breaking existing usage of this.

        If you need to specifically rely on the pandas API, please coerce this to a
        `pandas.DataFrame` using `features_to_pandas_dataframe`.

        References
        ----------
        .. [1]: https://data-apis.org/dataframe-protocol/latest/API.html
        """
        return self._feature_table.values

    @features.setter
    def features(
        self,
        features: Union[Dict[str, np.ndarray], pd.DataFrame],
    ) -> None:
        self._feature_table.set_values(features, num_data=self.nshapes)
        if self._face_color_property and (
            self._face_color_property not in self.features
        ):
            self._face_color_property = ''
            warnings.warn(
                trans._(
                    'property used for face_color dropped',
                    deferred=True,
                ),
                RuntimeWarning,
            )

        if self._edge_color_property and (
            self._edge_color_property not in self.features
        ):
            self._edge_color_property = ''
            warnings.warn(
                trans._(
                    'property used for edge_color dropped',
                    deferred=True,
                ),
                RuntimeWarning,
            )

        self.text.refresh(self.features)

        self.events.properties()
        self.events.features()

    @property
    def feature_defaults(self):
        """Dataframe-like with one row of feature default values.

        See `features` for more details on the type of this property.
        """
        return self._feature_table.defaults

    @feature_defaults.setter
    def feature_defaults(
        self, defaults: Union[Dict[str, Any], pd.DataFrame]
    ) -> None:
        self._feature_table.set_defaults(defaults)
        self.events.current_properties()
        self.events.feature_defaults()

    @property
    def properties(self) -> Dict[str, np.ndarray]:
        """dict {str: np.ndarray (N,)}, DataFrame: Annotations for each shape"""
        return self._feature_table.properties()

    @properties.setter
    def properties(self, properties: Dict[str, Array]):
        self.features = properties

    @property
    def property_choices(self) -> Dict[str, np.ndarray]:
        return self._feature_table.choices()

    def _get_ndim(self):
        """Determine number of dimensions of the layer."""
        ndim = self.ndim if self.nshapes == 0 else self.data[0].shape[1]
        return ndim

    @property
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        if len(self.data) == 0:
            extrema = np.full((2, self.ndim), np.nan)
        else:
            maxs = np.max([np.max(d, axis=0) for d in self.data], axis=0)
            mins = np.min([np.min(d, axis=0) for d in self.data], axis=0)
            extrema = np.vstack([mins, maxs])
        return extrema

    @property
    def nshapes(self):
        """int: Total number of shapes."""
        return len(self._data_view.shapes)

    @property
    def current_edge_width(self):
        """float: Width of shape edges including lines and paths."""
        return self._current_edge_width

    @current_edge_width.setter
    def current_edge_width(self, edge_width):
        self._current_edge_width = edge_width
        if self._update_properties:
            for i in self.selected_data:
                self._data_view.update_edge_width(i, edge_width)
        self.events.edge_width()

    @property
    def current_edge_color(self):
        """str: color of shape edges including lines and paths."""
        hex_ = rgb_to_hex(self._current_edge_color)[0]
        return hex_to_name.get(hex_, hex_)

    @current_edge_color.setter
    def current_edge_color(self, edge_color):
        self._current_edge_color = transform_color(edge_color)
        if self._update_properties:
            for i in self.selected_data:
                self._data_view.update_edge_color(i, self._current_edge_color)
            self.events.edge_color()
            self._update_thumbnail()
        self.events.current_edge_color()

    @property
    def current_face_color(self):
        """str: color of shape faces."""
        hex_ = rgb_to_hex(self._current_face_color)[0]
        return hex_to_name.get(hex_, hex_)

    @current_face_color.setter
    def current_face_color(self, face_color):
        self._current_face_color = transform_color(face_color)
        if self._update_properties:
            for i in self.selected_data:
                self._data_view.update_face_color(i, self._current_face_color)
            self.events.face_color()
            self._update_thumbnail()
        self.events.current_face_color()

    @property
    def current_properties(self) -> Dict[str, np.ndarray]:
        """dict{str: np.ndarray(1,)}: properties for the next added shape."""
        return self._feature_table.currents()

    @current_properties.setter
    def current_properties(self, current_properties):
        update_indices = None
        if (
            self._update_properties
            and len(self.selected_data) > 0
            and self._mode in [Mode.SELECT, Mode.PAN_ZOOM]
        ):
            update_indices = list(self.selected_data)
        self._feature_table.set_currents(
            current_properties, update_indices=update_indices
        )
        if update_indices is not None:
            self.refresh_colors()
            self.events.properties()
            self.events.features()
        self.events.current_properties()
        self.events.feature_defaults()

    @property
    def shape_type(self):
        """list of str: name of shape type for each shape."""
        return self._data_view.shape_types

    @shape_type.setter
    def shape_type(self, shape_type):
        self._finish_drawing()

        new_data_view = ShapeList()
        shape_inputs = zip(
            self._data_view.data,
            ensure_iterable(shape_type),
            self._data_view.edge_widths,
            self._data_view.edge_color,
            self._data_view.face_color,
            self._data_view.z_indices,
        )

        self._add_shapes_to_view(shape_inputs, new_data_view)

        self._data_view = new_data_view
        self._update_dims()

    @property
    def edge_color(self):
        """(N x 4) np.ndarray: Array of RGBA face colors for each shape"""
        return self._data_view.edge_color

    @edge_color.setter
    def edge_color(self, edge_color):
        self._set_color(edge_color, 'edge')
        self.events.edge_color()
        self._update_thumbnail()

    @property
    def edge_color_cycle(self) -> np.ndarray:
        """Union[list, np.ndarray] :  Color cycle for edge_color.

        Can be a list of colors defined by name, RGB or RGBA
        """
        return self._edge_color_cycle_values

    @edge_color_cycle.setter
    def edge_color_cycle(self, edge_color_cycle: Union[list, np.ndarray]):
        self._set_color_cycle(np.asarray(edge_color_cycle), 'edge')

    @property
    def edge_colormap(self) -> Colormap:
        """Return the colormap to be applied to a property to get the edge color.

        Returns
        -------
        colormap : napari.utils.Colormap
            The Colormap object.
        """
        return self._edge_colormap

    @edge_colormap.setter
    def edge_colormap(self, colormap: ValidColormapArg):
        self._edge_colormap = ensure_colormap(colormap)

    @property
    def edge_contrast_limits(self) -> Union[Tuple[float, float], None]:
        """None, (float, float): contrast limits for mapping
        the edge_color colormap property to 0 and 1
        """
        return self._edge_contrast_limits

    @edge_contrast_limits.setter
    def edge_contrast_limits(
        self, contrast_limits: Union[None, Tuple[float, float]]
    ):
        self._edge_contrast_limits = contrast_limits

    @property
    def edge_color_mode(self) -> str:
        """str: Edge color setting mode

        DIRECT (default mode) allows each shape color to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
        """
        return str(self._edge_color_mode)

    @edge_color_mode.setter
    def edge_color_mode(self, edge_color_mode: Union[str, ColorMode]):
        self._set_color_mode(edge_color_mode, 'edge')

    @property
    def face_color(self):
        """(N x 4) np.ndarray: Array of RGBA face colors for each shape"""
        return self._data_view.face_color

    @face_color.setter
    def face_color(self, face_color):
        self._set_color(face_color, 'face')
        self.events.face_color()
        self._update_thumbnail()

    @property
    def face_color_cycle(self) -> np.ndarray:
        """Union[np.ndarray, cycle]:  Color cycle for face_color
        Can be a list of colors defined by name, RGB or RGBA
        """
        return self._face_color_cycle_values

    @face_color_cycle.setter
    def face_color_cycle(self, face_color_cycle: Union[np.ndarray, cycle]):
        self._set_color_cycle(face_color_cycle, 'face')

    @property
    def face_colormap(self) -> Colormap:
        """Return the colormap to be applied to a property to get the face color.

        Returns
        -------
        colormap : napari.utils.Colormap
            The Colormap object.
        """
        return self._face_colormap

    @face_colormap.setter
    def face_colormap(self, colormap: ValidColormapArg):
        self._face_colormap = ensure_colormap(colormap)

    @property
    def face_contrast_limits(self) -> Union[None, Tuple[float, float]]:
        """None, (float, float) : clims for mapping the face_color
        colormap property to 0 and 1
        """
        return self._face_contrast_limits

    @face_contrast_limits.setter
    def face_contrast_limits(
        self, contrast_limits: Union[None, Tuple[float, float]]
    ):
        self._face_contrast_limits = contrast_limits

    @property
    def face_color_mode(self) -> str:
        """str: Face color setting mode

        DIRECT (default mode) allows each shape color to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
        """
        return str(self._face_color_mode)

    @face_color_mode.setter
    def face_color_mode(self, face_color_mode):
        self._set_color_mode(face_color_mode, 'face')

    def _set_color_mode(
        self, color_mode: Union[ColorMode, str], attribute: str
    ):
        """Set the face_color_mode or edge_color_mode property

        Parameters
        ----------
        color_mode : str, ColorMode
            The value for setting edge or face_color_mode. If color_mode is a string,
            it should be one of: 'direct', 'cycle', or 'colormap'
        attribute : str in {'edge', 'face'}
            The name of the attribute to set the color of.
            Should be 'edge' for edge_colo_moder or 'face' for face_color_mode.
        """
        color_mode = ColorMode(color_mode)

        if color_mode == ColorMode.DIRECT:
            setattr(self, f'_{attribute}_color_mode', color_mode)
        elif color_mode in (ColorMode.CYCLE, ColorMode.COLORMAP):
            color_property = getattr(self, f'_{attribute}_color_property')
            if color_property == '':
                if self.properties:
                    new_color_property = next(iter(self.properties))
                    setattr(
                        self,
                        f'_{attribute}_color_property',
                        new_color_property,
                    )
                    warnings.warn(
                        trans._(
                            '_{attribute}_color_property was not set, setting to: {new_color_property}',
                            deferred=True,
                            attribute=attribute,
                            new_color_property=new_color_property,
                        )
                    )
                else:
                    raise ValueError(
                        trans._(
                            'There must be a valid Shapes.properties to use {color_mode}',
                            deferred=True,
                            color_mode=color_mode,
                        )
                    )

            # ColorMode.COLORMAP can only be applied to numeric properties
            color_property = getattr(self, f'_{attribute}_color_property')
            if (color_mode == ColorMode.COLORMAP) and not issubclass(
                self.properties[color_property].dtype.type, np.number
            ):
                raise TypeError(
                    trans._(
                        'selected property must be numeric to use ColorMode.COLORMAP',
                        deferred=True,
                    )
                )
            setattr(self, f'_{attribute}_color_mode', color_mode)
            self.refresh_colors()

    def _set_color_cycle(
        self, color_cycle: Union[np.ndarray, cycle], attribute: str
    ):
        """Set the face_color_cycle or edge_color_cycle property

        Parameters
        ----------
        color_cycle : (N, 4) or (N, 1) array
            The value for setting edge or face_color_cycle
        attribute : str in {'edge', 'face'}
            The name of the attribute to set the color of.
            Should be 'edge' for edge_color or 'face' for face_color.
        """
        transformed_color_cycle, transformed_colors = transform_color_cycle(
            color_cycle=color_cycle,
            elem_name=f'{attribute}_color_cycle',
            default="white",
        )
        setattr(self, f'_{attribute}_color_cycle_values', transformed_colors)
        setattr(self, f'_{attribute}_color_cycle', transformed_color_cycle)

        if self._update_properties is True:
            color_mode = getattr(self, f'_{attribute}_color_mode')
            if color_mode == ColorMode.CYCLE:
                self.refresh_colors(update_color_mapping=True)

    @property
    def edge_width(self):
        """list of float: edge width for each shape."""
        return self._data_view.edge_widths

    @edge_width.setter
    def edge_width(self, width):
        """Set edge width of shapes using float or list of float.

        If list of float, must be of equal length to n shapes

        Parameters
        ----------
        width : float or list of float
            width of all shapes, or each shape if list
        """
        if isinstance(width, list):
            if not len(width) == self.nshapes:
                raise ValueError(
                    trans._('Length of list does not match number of shapes')
                )

            widths = width
        else:
            widths = [width for _ in range(self.nshapes)]

        for i, width in enumerate(widths):
            self._data_view.update_edge_width(i, width)

    @property
    def z_index(self):
        """list of int: z_index for each shape."""
        return self._data_view.z_indices

    @z_index.setter
    def z_index(self, z_index):
        """Set z_index of shape using either int or list of int.

        When list of int is provided, must be of equal length to n shapes.

        Parameters
        ----------
        z_index : int or list of int
            z-index of shapes
        """
        if isinstance(z_index, list):
            if not len(z_index) == self.nshapes:
                raise ValueError(
                    trans._('Length of list does not match number of shapes')
                )

            z_indices = z_index
        else:
            z_indices = [z_index for _ in range(self.nshapes)]

        for i, z_idx in enumerate(z_indices):
            self._data_view.update_z_index(i, z_idx)

    @property
    def selected_data(self):
        """set: set of currently selected shapes."""
        return self._selected_data

    @selected_data.setter
    def selected_data(self, selected_data):
        self._selected_data = set(selected_data)
        self._selected_box = self.interaction_box(self._selected_data)

        # Update properties based on selected shapes
        if len(selected_data) > 0:
            selected_data_indices = list(selected_data)
            selected_face_colors = self._data_view._face_color[
                selected_data_indices
            ]
            if (
                unique_face_color := _unique_element(selected_face_colors)
            ) is not None:
                with self.block_update_properties():
                    self.current_face_color = unique_face_color

            selected_edge_colors = self._data_view._edge_color[
                selected_data_indices
            ]
            if (
                unique_edge_color := _unique_element(selected_edge_colors)
            ) is not None:
                with self.block_update_properties():
                    self.current_edge_color = unique_edge_color

            unique_edge_width = _unique_element(
                np.array(
                    [
                        self._data_view.shapes[i].edge_width
                        for i in selected_data
                    ]
                )
            )
            if unique_edge_width is not None:
                with self.block_update_properties():
                    self.current_edge_width = unique_edge_width

            unique_properties = {}
            for k, v in self.properties.items():
                unique_properties[k] = _unique_element(
                    v[selected_data_indices]
                )

            if all(p is not None for p in unique_properties.values()):
                with self.block_update_properties():
                    self.current_properties = unique_properties

    def _set_color(self, color, attribute: str):
        """Set the face_color or edge_color property

        Parameters
        ----------
        color : (N, 4) array or str
            The value for setting edge or face_color
        attribute : str in {'edge', 'face'}
            The name of the attribute to set the color of.
            Should be 'edge' for edge_color or 'face' for face_color.
        """
        if self._is_color_mapped(color):
            if guess_continuous(self.properties[color]):
                setattr(self, f'_{attribute}_color_mode', ColorMode.COLORMAP)
            else:
                setattr(self, f'_{attribute}_color_mode', ColorMode.CYCLE)
            setattr(self, f'_{attribute}_color_property', color)
            self.refresh_colors(update_color_mapping=True)

        else:
            if len(self.data) > 0:
                transformed_color = transform_color_with_defaults(
                    num_entries=len(self.data),
                    colors=color,
                    elem_name="face_color",
                    default="white",
                )
                colors = normalize_and_broadcast_colors(
                    len(self.data), transformed_color
                )
            else:
                colors = np.empty((0, 4))

            setattr(self._data_view, f'{attribute}_color', colors)
            setattr(self, f'_{attribute}_color_mode', ColorMode.DIRECT)

            color_event = getattr(self.events, f'{attribute}_color')
            color_event()

    def refresh_colors(self, update_color_mapping: bool = False):
        """Calculate and update face and edge colors if using a cycle or color map

        Parameters
        ----------
        update_color_mapping : bool
            If set to True, the function will recalculate the color cycle map
            or colormap (whichever is being used). If set to False, the function
            will use the current color cycle map or color map. For example, if you
            are adding/modifying shapes and want them to be colored with the same
            mapping as the other shapes (i.e., the new shapes shouldn't affect
            the color cycle map or colormap), set update_color_mapping=False.
            Default value is False.
        """
        self._refresh_color('face', update_color_mapping)
        self._refresh_color('edge', update_color_mapping)

    def _refresh_color(
        self, attribute: str, update_color_mapping: bool = False
    ):
        """Calculate and update face or edge colors if using a cycle or color map

        Parameters
        ----------
        attribute : str  in {'edge', 'face'}
            The name of the attribute to set the color of.
            Should be 'edge' for edge_color or 'face' for face_color.
        update_color_mapping : bool
            If set to True, the function will recalculate the color cycle map
            or colormap (whichever is being used). If set to False, the function
            will use the current color cycle map or color map. For example, if you
            are adding/modifying shapes and want them to be colored with the same
            mapping as the other shapes (i.e., the new shapes shouldn't affect
            the color cycle map or colormap), set update_color_mapping=False.
            Default value is False.
        """
        if self._update_properties:
            color_mode = getattr(self, f'_{attribute}_color_mode')
            if color_mode in [ColorMode.CYCLE, ColorMode.COLORMAP]:
                colors = self._map_color(attribute, update_color_mapping)
                setattr(self._data_view, f'{attribute}_color', colors)
                color_event = getattr(self.events, f'{attribute}_color')
                color_event()

    def _initialize_color(self, color, attribute: str, n_shapes: int):
        """Get the face/edge colors the Shapes layer will be initialized with

        Parameters
        ----------
        color : (N, 4) array or str
            The value for setting edge or face_color
        attribute : str in {'edge', 'face'}
            The name of the attribute to set the color of.
            Should be 'edge' for edge_color or 'face' for face_color.

        Returns
        -------
        init_colors : (N, 4) array or str
            The calculated values for setting edge or face_color
        """
        if self._is_color_mapped(color):
            if guess_continuous(self.properties[color]):
                setattr(self, f'_{attribute}_color_mode', ColorMode.COLORMAP)
            else:
                setattr(self, f'_{attribute}_color_mode', ColorMode.CYCLE)
            setattr(self, f'_{attribute}_color_property', color)
            init_colors = self._map_color(
                attribute, update_color_mapping=False
            )

        else:
            if n_shapes > 0:
                transformed_color = transform_color_with_defaults(
                    num_entries=n_shapes,
                    colors=color,
                    elem_name="face_color",
                    default="white",
                )
                init_colors = normalize_and_broadcast_colors(
                    n_shapes, transformed_color
                )
            else:
                init_colors = np.empty((0, 4))

            setattr(self, f'_{attribute}_color_mode', ColorMode.DIRECT)

        return init_colors

    def _map_color(self, attribute: str, update_color_mapping: bool = False):
        """Calculate the mapping for face or edge colors if using a cycle or color map

        Parameters
        ----------
        attribute : str  in {'edge', 'face'}
            The name of the attribute to set the color of.
            Should be 'edge' for edge_color or 'face' for face_color.
        update_color_mapping : bool
            If set to True, the function will recalculate the color cycle map
            or colormap (whichever is being used). If set to False, the function
            will use the current color cycle map or color map. For example, if you
            are adding/modifying shapes and want them to be colored with the same
            mapping as the other shapes (i.e., the new shapes shouldn't affect
            the color cycle map or colormap), set update_color_mapping=False.
            Default value is False.

        Returns
        -------
        colors : (N, 4) array or str
            The calculated values for setting edge or face_color
        """
        color_mode = getattr(self, f'_{attribute}_color_mode')
        if color_mode == ColorMode.CYCLE:
            color_property = getattr(self, f'_{attribute}_color_property')
            color_properties = self.properties[color_property]
            if update_color_mapping:
                color_cycle = getattr(self, f'_{attribute}_color_cycle')
                color_cycle_map = {
                    k: np.squeeze(transform_color(c))
                    for k, c in zip(np.unique(color_properties), color_cycle)
                }
                setattr(self, f'{attribute}_color_cycle_map', color_cycle_map)

            else:
                # add properties if they are not in the colormap
                # and update_color_mapping==False
                color_cycle_map = getattr(self, f'{attribute}_color_cycle_map')
                color_cycle_keys = [*color_cycle_map]
                props_in_map = np.in1d(color_properties, color_cycle_keys)
                if not np.all(props_in_map):
                    props_to_add = np.unique(
                        color_properties[np.logical_not(props_in_map)]
                    )
                    color_cycle = getattr(self, f'_{attribute}_color_cycle')
                    for prop in props_to_add:
                        color_cycle_map[prop] = np.squeeze(
                            transform_color(next(color_cycle))
                        )
                    setattr(
                        self,
                        f'{attribute}_color_cycle_map',
                        color_cycle_map,
                    )
            colors = np.array([color_cycle_map[x] for x in color_properties])
            if len(colors) == 0:
                colors = np.empty((0, 4))

        elif color_mode == ColorMode.COLORMAP:
            color_property = getattr(self, f'_{attribute}_color_property')
            color_properties = self.properties[color_property]
            if len(color_properties) > 0:
                contrast_limits = getattr(self, f'{attribute}_contrast_limits')
                colormap = getattr(self, f'{attribute}_colormap')
                if update_color_mapping or contrast_limits is None:
                    colors, contrast_limits = map_property(
                        prop=color_properties, colormap=colormap
                    )
                    setattr(
                        self,
                        f'{attribute}_contrast_limits',
                        contrast_limits,
                    )
                else:
                    colors, _ = map_property(
                        prop=color_properties,
                        colormap=colormap,
                        contrast_limits=contrast_limits,
                    )
            else:
                colors = np.empty((0, 4))

        return colors

    def _get_new_shape_color(self, adding: int, attribute: str):
        """Get the color for the shape(s) to be added.

        Parameters
        ----------
        adding : int
            the number of shapes that were added
            (and thus the number of color entries to add)
        attribute : str in {'edge', 'face'}
            The name of the attribute to set the color of.
            Should be 'edge' for edge_color_mode or 'face' for face_color_mode.

        Returns
        -------
        new_colors : (N, 4) array
            (Nx4) RGBA array of colors for the N new shapes
        """
        color_mode = getattr(self, f'_{attribute}_color_mode')
        if color_mode == ColorMode.DIRECT:
            current_face_color = getattr(self, f'_current_{attribute}_color')
            new_colors = np.tile(current_face_color, (adding, 1))
        elif color_mode == ColorMode.CYCLE:
            property_name = getattr(self, f'_{attribute}_color_property')
            color_property_value = self.current_properties[property_name][0]

            # check if the new color property is in the cycle map
            # and add it if it is not
            color_cycle_map = getattr(self, f'{attribute}_color_cycle_map')
            color_cycle_keys = [*color_cycle_map]
            if color_property_value not in color_cycle_keys:
                color_cycle = getattr(self, f'_{attribute}_color_cycle')
                color_cycle_map[color_property_value] = np.squeeze(
                    transform_color(next(color_cycle))
                )

                setattr(self, f'{attribute}_color_cycle_map', color_cycle_map)

            new_colors = np.tile(
                color_cycle_map[color_property_value], (adding, 1)
            )
        elif color_mode == ColorMode.COLORMAP:
            property_name = getattr(self, f'_{attribute}_color_property')
            color_property_value = self.current_properties[property_name][0]
            colormap = getattr(self, f'{attribute}_colormap')
            contrast_limits = getattr(self, f'_{attribute}_contrast_limits')

            fc, _ = map_property(
                prop=color_property_value,
                colormap=colormap,
                contrast_limits=contrast_limits,
            )
            new_colors = np.tile(fc, (adding, 1))

        return new_colors

    def _is_color_mapped(self, color):
        """determines if the new color argument is for directly setting or cycle/colormap"""
        if isinstance(color, str):
            return color in self.properties
        if isinstance(color, (list, np.ndarray)):
            return False

        raise ValueError(
            trans._(
                'face_color should be the name of a color, an array of colors, or the name of an property',
                deferred=True,
            )
        )

    def _get_state(self):
        """Get dictionary of layer state.

        Returns
        -------
        state : dict
            Dictionary of layer state.
        """
        state = self._get_base_state()
        face_color = self.face_color
        edge_color = self.edge_color
        if not face_color.size:
            face_color = self._current_face_color
        if not edge_color.size:
            edge_color = self._current_edge_color
        state.update(
            {
                'ndim': self.ndim,
                'properties': self.properties,
                'property_choices': self.property_choices,
                'text': self.text.dict(),
                'shape_type': self.shape_type,
                'opacity': self.opacity,
                'z_index': self.z_index,
                'edge_width': self.edge_width,
                'face_color': face_color,
                'face_color_cycle': self.face_color_cycle,
                'face_colormap': self.face_colormap.dict(),
                'face_contrast_limits': self.face_contrast_limits,
                'edge_color': edge_color,
                'edge_color_cycle': self.edge_color_cycle,
                'edge_colormap': self.edge_colormap.dict(),
                'edge_contrast_limits': self.edge_contrast_limits,
                'data': self.data,
                'features': self.features,
                'feature_defaults': self.feature_defaults,
            }
        )
        return state

    @property
    def _indices_view(self):
        return np.where(self._data_view._displayed)[0]

    @property
    def _view_text(self) -> np.ndarray:
        """Get the values of the text elements in view

        Returns
        -------
        text : (N x 1) np.ndarray
            Array of text strings for the N text elements in view
        """
        # This may be triggered when the string encoding instance changed,
        # in which case it has no cached values, so generate them here.
        self.text.string._apply(self.features)
        return self.text.view_text(self._indices_view)

    @property
    def _view_text_coords(self) -> Tuple[np.ndarray, str, str]:
        """Get the coordinates of the text elements in view

        Returns
        -------
        text_coords : (N x D) np.ndarray
            Array of coordinates for the N text elements in view
        anchor_x : str
            The vispy text anchor for the x axis
        anchor_y : str
            The vispy text anchor for the y axis
        """
        ndisplay = self._slice_input.ndisplay
        order = self._slice_input.order

        # get the coordinates of the vertices for the shapes in view
        in_view_shapes_coords = [
            self._data_view.data[i] for i in self._indices_view
        ]

        # get the coordinates for the dimensions being displayed
        sliced_in_view_coords = [
            position[:, self._slice_input.displayed]
            for position in in_view_shapes_coords
        ]
        # TODO: fix types here with np.asarray(sliced_in_view_coords)
        # but blocked by https://github.com/napari/napari/issues/6294
        return self.text.compute_text_coords(
            sliced_in_view_coords, ndisplay, order
        )

    @property
    def _view_text_color(self) -> np.ndarray:
        """Get the colors of the text elements at the given indices."""
        self.text.color._apply(self.features)
        return self.text._view_color(self._indices_view)

    @property
    def mode(self):
        """MODE: Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        The SELECT mode allows for entire shapes to be selected, moved and
        resized.

        The DIRECT mode allows for shapes to be selected and their individual
        vertices to be moved.

        The VERTEX_INSERT and VERTEX_REMOVE modes allow for individual
        vertices either to be added to or removed from shapes that are already
        selected. Note that shapes cannot be selected in this mode.

        The ADD_RECTANGLE, ADD_ELLIPSE, ADD_LINE, ADD_PATH, and ADD_POLYGON
        modes all allow for their corresponding shape type to be added.
        """
        return str(self._mode)

    @mode.setter
    def mode(self, val: Union[str, Mode]):
        mode = self._mode_setter_helper(val)
        if mode == self._mode:
            return

        self._mode = mode
        self.events.mode(mode=mode)

        draw_modes = {
            Mode.SELECT,
            Mode.DIRECT,
            Mode.VERTEX_INSERT,
            Mode.VERTEX_REMOVE,
        }

        # don't update thumbnail on mode changes
        with self.block_thumbnail_update():
            if not (mode in draw_modes and self._mode in draw_modes):
                # Shapes._finish_drawing() calls Shapes.refresh()
                self._finish_drawing()
            else:
                self.refresh()

    def _reset_editable(self) -> None:
        self.editable = self._slice_input.ndisplay == 2

    def _on_editable_changed(self) -> None:
        if not self.editable:
            self.mode = Mode.PAN_ZOOM

    def add_rectangles(
        self,
        data,
        *,
        edge_width=None,
        edge_color=None,
        face_color=None,
        z_index=None,
    ):
        """Add rectangles to the current layer.

        Parameters
        ----------
        data : Array | List[Array]
            List of rectangle data where each element is a (4, D) array of 4 vertices
            in D dimensions, or a (2, D) array of 2 vertices in D dimensions, where
            the vertices are top-left and bottom-right corners.
            Can be a 3-dimensional array for multiple shapes, or list of 2 or 4
            vertices for a single shape.
        edge_width : float | list
            thickness of lines and edges. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        edge_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        face_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        z_index : int | list
            Specifier of z order priority. Shapes with higher z order are
            displayed ontop of others. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        """
        # rectangles can have either 4 vertices or (top left, bottom right)
        valid_vertices_per_shape = (2, 4)
        validate_num_vertices(
            data, 'rectangle', valid_vertices=valid_vertices_per_shape
        )

        self.add(
            data,
            shape_type='rectangle',
            edge_width=edge_width,
            edge_color=edge_color,
            face_color=face_color,
            z_index=z_index,
        )

    def add_ellipses(
        self,
        data,
        *,
        edge_width=None,
        edge_color=None,
        face_color=None,
        z_index=None,
    ):
        """Add ellipses to the current layer.

        Parameters
        ----------
        data : Array | List[Array]
            List of ellipse data where each element is a (4, D) array of 4 vertices
            in D dimensions representing a bounding box, or a (2, D) array of
            center position and radii magnitudes in D dimensions.
            Can be a 3-dimensional array for multiple shapes, or list of 2 or 4
            vertices for a single shape.
        edge_width : float | list
            thickness of lines and edges. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        edge_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        face_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        z_index : int | list
            Specifier of z order priority. Shapes with higher z order are
            displayed ontop of others. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        """

        valid_elem_per_shape = (2, 4)
        validate_num_vertices(
            data, 'ellipse', valid_vertices=valid_elem_per_shape
        )

        self.add(
            data,
            shape_type='ellipse',
            edge_width=edge_width,
            edge_color=edge_color,
            face_color=face_color,
            z_index=z_index,
        )

    def add_polygons(
        self,
        data,
        *,
        edge_width=None,
        edge_color=None,
        face_color=None,
        z_index=None,
    ):
        """Add polygons to the current layer.

        Parameters
        ----------
        data : Array | List[Array]
            List of polygon data where each element is a (V, D) array of V vertices
            in D dimensions representing a polygon. Can be a 3-dimensional array if
            polygons have same number of vertices, or a list of V vertices for a
            single polygon.
        edge_width : float | list
            thickness of lines and edges. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        edge_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        face_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        z_index : int | list
            Specifier of z order priority. Shapes with higher z order are
            displayed ontop of others. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        """

        min_vertices = 3
        validate_num_vertices(data, 'polygon', min_vertices=min_vertices)

        self.add(
            data,
            shape_type='polygon',
            edge_width=edge_width,
            edge_color=edge_color,
            face_color=face_color,
            z_index=z_index,
        )

    def add_lines(
        self,
        data,
        *,
        edge_width=None,
        edge_color=None,
        face_color=None,
        z_index=None,
    ):
        """Add lines to the current layer.

        Parameters
        ----------
        data : Array | List[Array]
            List of line data where each element is a (2, D) array of 2 vertices
            in D dimensions representing a line. Can be a 3-dimensional array for
            multiple shapes, or list of 2 vertices for a single shape.
        edge_width : float | list
            thickness of lines and edges. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        edge_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        face_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        z_index : int | list
            Specifier of z order priority. Shapes with higher z order are
            displayed ontop of others. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        """

        valid_vertices_per_line = (2,)
        validate_num_vertices(
            data, 'line', valid_vertices=valid_vertices_per_line
        )

        self.add(
            data,
            shape_type='line',
            edge_width=edge_width,
            edge_color=edge_color,
            face_color=face_color,
            z_index=z_index,
        )

    def add_paths(
        self,
        data,
        *,
        edge_width=None,
        edge_color=None,
        face_color=None,
        z_index=None,
    ):
        """Add paths to the current layer.

        Parameters
        ----------
        data : Array | List[Array]
            List of path data where each element is a (V, D) array of V vertices
            in D dimensions representing a path. Can be a 3-dimensional array
            if all paths have same number of vertices, or a list of V vertices
            for a single path.
        edge_width : float | list
            thickness of lines and edges. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        edge_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        face_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        z_index : int | list
            Specifier of z order priority. Shapes with higher z order are
            displayed ontop of others. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        """

        min_vertices_per_path = 2
        validate_num_vertices(data, 'path', min_vertices=min_vertices_per_path)

        self.add(
            data,
            shape_type='path',
            edge_width=edge_width,
            edge_color=edge_color,
            face_color=face_color,
            z_index=z_index,
        )

    def add(
        self,
        data,
        *,
        shape_type='rectangle',
        edge_width=None,
        edge_color=None,
        face_color=None,
        z_index=None,
        gui=False,
    ):
        """Add shapes to the current layer.

        Parameters
        ----------
        data : Array | Tuple(Array,str) | List[Array | Tuple(Array, str)] | Tuple(List[Array], str)
            List of shape data, where each element is either an (N, D) array of the
            N vertices of a shape in D dimensions or a tuple containing an array of
            the N vertices and the shape_type string. When a shape_type is present,
            it overrides keyword arg shape_type. Can be an 3-dimensional array
            if each shape has the same number of vertices.
        shape_type : string | list
            String of shape shape_type, must be one of "{'line', 'rectangle',
            'ellipse', 'path', 'polygon'}". If a list is supplied it must be
            the same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes. Overridden by data shape_type, if present.
        edge_width : float | list
            thickness of lines and edges. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        edge_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        face_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        z_index : int | list
            Specifier of z order priority. Shapes with higher z order are
            displayed ontop of others. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        gui : bool
            Whether the shape is drawn by drawing in the gui.
        """
        data, shape_type = extract_shape_type(data, shape_type)

        n_new_shapes = number_of_shapes(data)

        if n_new_shapes > 0:
            self.events.data(
                value=self.data,
                action=ActionType.ADDING,
                data_indices=(-1,),
                vertex_indices=((),),
            )
            self._add_shapes(
                data,
                shape_type=shape_type,
                edge_width=edge_width,
                edge_color=edge_color,
                face_color=face_color,
                z_index=z_index,
                n_new_shapes=n_new_shapes,
            )
            # This should only emit when programmatically adding as with drawing this leads to premature emit.
            if not gui:
                self.events.data(
                    value=self.data,
                    action=ActionType.ADDED,
                    data_indices=(-1,),
                    vertex_indices=((),),
                )

    def _init_shapes(
        self,
        data,
        *,
        shape_type='rectangle',
        edge_width=None,
        edge_color=None,
        edge_color_cycle,
        edge_colormap,
        edge_contrast_limits,
        face_color=None,
        face_color_cycle,
        face_colormap,
        face_contrast_limits,
        z_index=None,
    ):
        """Add shapes to the data view.

        Parameters
        ----------
        data : Array | Tuple(Array,str) | List[Array | Tuple(Array, str)] | Tuple(List[Array], str)
            List of shape data, where each element is either an (N, D) array of the
            N vertices of a shape in D dimensions or a tuple containing an array of
            the N vertices and the shape_type string. When a shape_type is present,
            it overrides keyword arg shape_type. Can be an 3-dimensional array
            if each shape has the same number of vertices.
        shape_type : string | list
            String of shape shape_type, must be one of "{'line', 'rectangle',
            'ellipse', 'path', 'polygon'}". If a list is supplied it must be
            the same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes. Overriden by data shape_type, if present.
        edge_width : float | list
            thickness of lines and edges. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        edge_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        face_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        z_index : int | list
            Specifier of z order priority. Shapes with higher z order are
            displayed ontop of others. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        """

        n_shapes = number_of_shapes(data)
        with self.block_update_properties():
            self._edge_color_property = ''
            self.edge_color_cycle_map = {}
            self.edge_colormap = edge_colormap
            self._edge_contrast_limits = edge_contrast_limits
            if edge_color_cycle is None:
                edge_color_cycle = deepcopy(DEFAULT_COLOR_CYCLE)
            self.edge_color_cycle = edge_color_cycle
            edge_color = self._initialize_color(
                edge_color, attribute='edge', n_shapes=n_shapes
            )

            self._face_color_property = ''
            self.face_color_cycle_map = {}
            self.face_colormap = face_colormap
            self._face_contrast_limits = face_contrast_limits
            if face_color_cycle is None:
                face_color_cycle = deepcopy(DEFAULT_COLOR_CYCLE)
            self.face_color_cycle = face_color_cycle
            face_color = self._initialize_color(
                face_color, attribute='face', n_shapes=n_shapes
            )

        with self.block_thumbnail_update():
            self._add_shapes(
                data,
                shape_type=shape_type,
                edge_width=edge_width,
                edge_color=edge_color,
                face_color=face_color,
                z_index=z_index,
                n_new_shapes=n_shapes,
            )
            self._data_view._update_z_order()
            self.refresh_colors()

    def _add_shapes(
        self,
        data,
        *,
        shape_type='rectangle',
        edge_width=None,
        edge_color=None,
        face_color=None,
        z_index=None,
        n_new_shapes=0,
    ):
        """Add shapes to the data view.

        Parameters
        ----------
        data : Array | Tuple(Array,str) | List[Array | Tuple(Array, str)] | Tuple(List[Array], str)
            List of shape data, where each element is either an (N, D) array of the
            N vertices of a shape in D dimensions or a tuple containing an array of
            the N vertices and the shape_type string. When a shape_type is present,
            it overrides keyword arg shape_type. Can be an 3-dimensional array
            if each shape has the same number of vertices.
        shape_type : string | list
            String of shape shape_type, must be one of "{'line', 'rectangle',
            'ellipse', 'path', 'polygon'}". If a list is supplied it must be
            the same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes. Overridden by data shape_type, if present.
        edge_width : float | list
            thickness of lines and edges. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        edge_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        face_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        z_index : int | list
            Specifier of z order priority. Shapes with higher z order are
            displayed on top of others. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        n_new_shapes : int
            The number of new shapes to be added to the Shapes layer.
        """
        if n_new_shapes > 0:
            total_shapes = n_new_shapes + self.nshapes
            self._feature_table.resize(total_shapes)
            if hasattr(self, "text"):
                self.text.apply(self.features)

        if edge_color is None:
            edge_color = self._get_new_shape_color(
                n_new_shapes, attribute='edge'
            )
        if face_color is None:
            face_color = self._get_new_shape_color(
                n_new_shapes, attribute='face'
            )

        if edge_width is None:
            edge_width = self.current_edge_width
        if edge_color is None:
            edge_color = self._current_edge_color
        if face_color is None:
            face_color = self._current_face_color
        if self._data_view is not None:
            z_index = z_index or max(self._data_view._z_index, default=-1) + 1
        else:
            z_index = z_index or 0

        if len(data) > 0:
            if np.array(data[0]).ndim == 1:
                # If a single array for a shape has been passed turn into list
                data = [data]

            # transform the colors
            transformed_ec = transform_color_with_defaults(
                num_entries=len(data),
                colors=edge_color,
                elem_name="edge_color",
                default="white",
            )
            transformed_edge_color = normalize_and_broadcast_colors(
                len(data), transformed_ec
            )
            transformed_fc = transform_color_with_defaults(
                num_entries=len(data),
                colors=face_color,
                elem_name="face_color",
                default="white",
            )
            transformed_face_color = normalize_and_broadcast_colors(
                len(data), transformed_fc
            )

            # Turn input arguments into iterables
            shape_inputs = zip(
                data,
                ensure_iterable(shape_type),
                ensure_iterable(edge_width),
                transformed_edge_color,
                transformed_face_color,
                ensure_iterable(z_index),
            )

            self._add_shapes_to_view(shape_inputs, self._data_view)

        self._display_order_stored = copy(self._slice_input.order)
        self._ndisplay_stored = copy(self._slice_input.ndisplay)
        self._update_dims()

    def _add_shapes_to_view(self, shape_inputs, data_view):
        """Build new shapes and add them to the _data_view"""

        shape_inputs = tuple(shape_inputs)

        # build all shapes
        sh_inp = tuple(
            (
                shape_classes[ShapeType(st)](
                    d,
                    edge_width=ew,
                    z_index=z,
                    dims_order=self._slice_input.order,
                    ndisplay=self._slice_input.ndisplay,
                ),
                ec,
                fc,
            )
            for d, st, ew, ec, fc, z in shape_inputs
        )

        shapes, edge_colors, face_colors = tuple(zip(*sh_inp))

        # Add all shapes at once (faster than adding them one by one)
        data_view.add(
            shape=shapes,
            edge_color=edge_colors,
            face_color=face_colors,
            z_refresh=False,
        )

        data_view._update_z_order()

    @property
    def text(self) -> TextManager:
        """TextManager: The TextManager object containing the text properties"""
        return self._text

    @text.setter
    def text(self, text):
        self._text._update_from_layer(
            text=text,
            features=self.features,
        )

    def refresh_text(self):
        """Refresh the text values.

        This is generally used if the properties were updated without changing the data
        """
        self.text.refresh(self.features)

    @property
    def _normalized_scale_factor(self):
        """Scale factor accounting for layer scale.

        This is often needed when calculating screen-space sizes and distances
        of vertices for interactivity (rescaling, adding vertices, etc).
        """
        return self.scale_factor / self.scale[-1]

    @property
    def _normalized_vertex_radius(self):
        """Vertex radius normalized to screen space."""
        return self._vertex_size * self._normalized_scale_factor / 2

    def _set_view_slice(self):
        """Set the view given the slicing indices."""
        with self._data_view.batched_updates():
            ndisplay = self._slice_input.ndisplay
            if ndisplay != self._ndisplay_stored:
                self.selected_data = set()
                self._data_view.ndisplay = min(self.ndim, ndisplay)
                self._ndisplay_stored = ndisplay
                self._clipboard = {}

            if self._slice_input.order != self._display_order_stored:
                self.selected_data = set()
                self._data_view.update_dims_order(self._slice_input.order)
                self._display_order_stored = copy(self._slice_input.order)
                # Clear clipboard if dimensions swap
                self._clipboard = {}

            slice_key = np.array(self._data_slice.point)[
                self._slice_input.not_displayed
            ]
            if not np.array_equal(slice_key, self._data_view.slice_key):
                self.selected_data = set()
            self._data_view.slice_key = slice_key

    def interaction_box(self, index):
        """Create the interaction box around a shape or list of shapes.
        If a single index is passed then the boudning box will be inherited
        from that shapes interaction box. If list of indices is passed it will
        be computed directly.

        Parameters
        ----------
        index : int | list
            Index of a single shape, or a list of shapes around which to
            construct the interaction box

        Returns
        -------
        box : np.ndarray
            10x2 array of vertices of the interaction box. The first 8 points
            are the corners and midpoints of the box in clockwise order
            starting in the upper-left corner. The 9th point is the center of
            the box, and the last point is the location of the rotation handle
            that can be used to rotate the box
        """
        if isinstance(index, (list, np.ndarray, set)):
            if len(index) == 0:
                box = None
            elif len(index) == 1:
                box = copy(self._data_view.shapes[next(iter(index))]._box)
            else:
                indices = np.isin(self._data_view.displayed_index, list(index))
                box = create_box(self._data_view.displayed_vertices[indices])
        else:
            box = copy(self._data_view.shapes[index]._box)

        if box is not None:
            rot = box[Box.TOP_CENTER]
            length_box = np.linalg.norm(
                box[Box.BOTTOM_LEFT] - box[Box.TOP_LEFT]
            )
            if length_box > 0:
                r = (
                    self._rotation_handle_length
                    * self._normalized_scale_factor
                )
                rot = (
                    rot
                    - r
                    * (box[Box.BOTTOM_LEFT] - box[Box.TOP_LEFT])
                    / length_box
                )
            box = np.append(box, [rot], axis=0)

        return box

    def _outline_shapes(self):
        """Find outlines of any selected or hovered shapes.

        Returns
        -------
        vertices : None | np.ndarray
            Nx2 array of any vertices of outline or None
        triangles : None | np.ndarray
            Mx3 array of any indices of vertices for triangles of outline or
            None
        """
        if self._value is not None and (
            self._value[0] is not None or len(self.selected_data) > 0
        ):
            if len(self.selected_data) > 0:
                index = list(self.selected_data)
                if self._value[0] is not None:
                    if self._value[0] in index:
                        pass
                    else:
                        index.append(self._value[0])
                index.sort()
            else:
                index = self._value[0]

            centers, offsets, triangles = self._data_view.outline(index)
            vertices = centers + (
                self._normalized_scale_factor * self._highlight_width * offsets
            )
            vertices = vertices[:, ::-1]
        else:
            vertices = None
            triangles = None

        return vertices, triangles

    def _compute_vertices_and_box(self):
        """Compute location of highlight vertices and box for rendering.

        Returns
        -------
        vertices : np.ndarray
            Nx2 array of any vertices to be rendered as Markers
        face_color : str
            String of the face color of the Markers
        edge_color : str
            String of the edge color of the Markers and Line for the box
        pos : np.ndarray
            Nx2 array of vertices of the box that will be rendered using a
            Vispy Line
        width : float
            Width of the box edge
        """
        if len(self.selected_data) > 0:
            if self._mode == Mode.SELECT:
                # If in select mode just show the interaction boudning box
                # including its vertices and the rotation handle
                box = self._selected_box[Box.WITH_HANDLE]
                if self._value[0] is None or self._value[1] is None:
                    face_color = 'white'
                else:
                    face_color = self._highlight_color
                edge_color = self._highlight_color
                vertices = box[:, ::-1]
                # Use a subset of the vertices of the interaction_box to plot
                # the line around the edge
                pos = box[Box.LINE_HANDLE][:, ::-1]
                width = 1.5
            elif self._mode in (
                [
                    Mode.DIRECT,
                    Mode.ADD_PATH,
                    Mode.ADD_POLYGON,
                    Mode.ADD_POLYGON_LASSO,
                    Mode.ADD_RECTANGLE,
                    Mode.ADD_ELLIPSE,
                    Mode.ADD_LINE,
                    Mode.VERTEX_INSERT,
                    Mode.VERTEX_REMOVE,
                ]
            ):
                # If in one of these mode show the vertices of the shape itself
                inds = np.isin(
                    self._data_view.displayed_index, list(self.selected_data)
                )
                vertices = self._data_view.displayed_vertices[inds][:, ::-1]
                # If currently adding path don't show box over last vertex
                if self._mode == Mode.ADD_PATH:
                    vertices = vertices[:-1]

                if self._value[0] is None or self._value[1] is None:
                    face_color = 'white'
                else:
                    face_color = self._highlight_color
                edge_color = self._highlight_color
                pos = None
                width = 0
            else:
                # Otherwise show nothing
                vertices = np.empty((0, 2))
                face_color = 'white'
                edge_color = 'white'
                pos = None
                width = 0
        elif self._is_selecting:
            # If currently dragging a selection box just show an outline of
            # that box
            vertices = np.empty((0, 2))
            edge_color = self._highlight_color
            face_color = 'white'
            box = create_box(self._drag_box)
            width = 1.5
            # Use a subset of the vertices of the interaction_box to plot
            # the line around the edge
            pos = box[Box.LINE][:, ::-1]
        else:
            # Otherwise show nothing
            vertices = np.empty((0, 2))
            face_color = 'white'
            edge_color = 'white'
            pos = None
            width = 0

        return vertices, face_color, edge_color, pos, width

    def _set_highlight(self, force=False) -> None:
        """Render highlights of shapes.

        Includes boundaries, vertices, interaction boxes, and the drag
        selection box when appropriate.

        Parameters
        ----------
        force : bool
            Bool that forces a redraw to occur when `True`
        """
        # Check if any shape or vertex ids have changed since last call
        if (
            self.selected_data == self._selected_data_stored
            and np.array_equal(self._value, self._value_stored)
            and np.array_equal(self._drag_box, self._drag_box_stored)
        ) and not force:
            return
        self._selected_data_stored = copy(self.selected_data)
        self._value_stored = copy(self._value)
        self._drag_box_stored = copy(self._drag_box)
        self.events.highlight()

    def _finish_drawing(self, event=None):
        """Reset properties used in shape drawing."""
        index = copy(self._moving_value[0])
        self._is_moving = False
        self._drag_start = None
        self._drag_box = None
        self._is_selecting = False
        self._fixed_vertex = None
        self._value = (None, None)
        self._moving_value = (None, None)
        if self._is_creating is True:
            if self._mode == Mode.ADD_PATH:
                vertices = self._data_view.shapes[index].data
                if len(vertices) <= 2:
                    self._data_view.remove(index)
                else:
                    self._data_view.edit(index, vertices[:-1])
            if self._mode in {Mode.ADD_POLYGON, Mode.ADD_POLYGON_LASSO}:
                vertices = self._data_view.shapes[index].data
                if len(vertices) <= 3:
                    self._data_view.remove(index)
                elif self._mode == Mode.ADD_POLYGON:
                    self._data_view.edit(index, vertices[:-1])
                else:
                    vertices = rdp(
                        vertices,
                        epsilon=get_settings().experimental.rdp_epsilon,
                    )
                    self._data_view.edit(
                        index,
                        vertices[:-1],
                        new_type=shape_classes[ShapeType.POLYGON],
                    )
        # handles the case that
        if index is not None:
            self.events.data(
                value=self.data,
                action=ActionType.ADDED,
                data_indices=(-1,),
                vertex_indices=((),),
            )
        self._is_creating = False
        self._update_dims()

    @contextmanager
    def block_thumbnail_update(self):
        """Use this context manager to block thumbnail updates"""
        previous = self._allow_thumbnail_update
        self._allow_thumbnail_update = False
        try:
            yield
        finally:
            self._allow_thumbnail_update = previous

    def _update_thumbnail(self, event=None):
        """Update thumbnail with current shapes and colors."""
        # Set the thumbnail to black, opacity 1
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1
        # if the shapes layer is empty, don't update, just leave it black
        if len(self.data) == 0:
            self.thumbnail = colormapped
        # don't update the thumbnail if dragging a shape
        elif self._is_moving is False and self._allow_thumbnail_update is True:
            # calculate min vals for the vertices and pad with 0.5
            # the offset is needed to ensure that the top left corner of the shapes
            # corresponds to the top left corner of the thumbnail
            de = self._extent_data
            offset = (
                np.array([de[0, d] for d in self._slice_input.displayed]) + 0.5
            )
            # calculate range of values for the vertices and pad with 1
            # padding ensures the entire shape can be represented in the thumbnail
            # without getting clipped
            shape = np.ceil(
                [de[1, d] - de[0, d] + 1 for d in self._slice_input.displayed]
            ).astype(int)
            zoom_factor = np.divide(
                self._thumbnail_shape[:2], shape[-2:]
            ).min()

            colormapped = self._data_view.to_colors(
                colors_shape=self._thumbnail_shape[:2],
                zoom_factor=zoom_factor,
                offset=offset[-2:],
                max_shapes=self._max_shapes_thumbnail,
            )

            self.thumbnail = colormapped

    def remove_selected(self):
        """Remove any selected shapes."""
        index = list(self.selected_data)
        to_remove = sorted(index, reverse=True)

        if len(index) > 0:
            self.events.data(
                value=self.data,
                action=ActionType.REMOVING,
                data_indices=tuple(
                    index,
                ),
                vertex_indices=((),),
            )
            for ind in to_remove:
                self._data_view.remove(ind)

            self._feature_table.remove(index)
            self.text.remove(index)
            self._data_view._edge_color = np.delete(
                self._data_view._edge_color, index, axis=0
            )
            self._data_view._face_color = np.delete(
                self._data_view._face_color, index, axis=0
            )
            self.events.data(
                value=self.data,
                action=ActionType.REMOVED,
                data_indices=tuple(
                    index,
                ),
                vertex_indices=((),),
            )
        self.selected_data.clear()
        self._finish_drawing()

    def _rotate_box(self, angle, center=(0, 0)):
        """Perform a rotation on the selected box.

        Parameters
        ----------
        angle : float
            angle specifying rotation of shapes in degrees.
        center : list
            coordinates of center of rotation.
        """
        theta = np.radians(angle)
        transform = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )
        box = self._selected_box - center
        self._selected_box = box @ transform.T + center

    def _scale_box(self, scale, center=(0, 0)):
        """Perform a scaling on the selected box.

        Parameters
        ----------
        scale : float, list
            scalar or list specifying rescaling of shape.
        center : list
            coordinates of center of rotation.
        """
        if not isinstance(scale, (list, np.ndarray)):
            scale = [scale, scale]
        box = self._selected_box - center
        box = np.array(box * scale)
        if not np.array_equal(box[Box.TOP_CENTER], box[Box.HANDLE]):
            r = self._rotation_handle_length * self._normalized_scale_factor
            handle_vec = box[Box.HANDLE] - box[Box.TOP_CENTER]
            cur_len = np.linalg.norm(handle_vec)
            box[Box.HANDLE] = box[Box.TOP_CENTER] + r * handle_vec / cur_len
        self._selected_box = box + center

    def _transform_box(self, transform, center=(0, 0)):
        """Perform a linear transformation on the selected box.

        Parameters
        ----------
        transform : np.ndarray
            2x2 array specifying linear transform.
        center : list
            coordinates of center of rotation.
        """
        box = self._selected_box - center
        box = box @ transform.T
        if not np.array_equal(box[Box.TOP_CENTER], box[Box.HANDLE]):
            r = self._rotation_handle_length * self._normalized_scale_factor
            handle_vec = box[Box.HANDLE] - box[Box.TOP_CENTER]
            cur_len = np.linalg.norm(handle_vec)
            box[Box.HANDLE] = box[Box.TOP_CENTER] + r * handle_vec / cur_len
        self._selected_box = box + center

    def _update_draw(
        self, scale_factor, corner_pixels_displayed, shape_threshold
    ):
        prev_scale = self.scale_factor
        super()._update_draw(
            scale_factor, corner_pixels_displayed, shape_threshold
        )
        # update highlight only if scale has changed, otherwise causes a cycle
        self._set_highlight(force=(prev_scale != self.scale_factor))

    def _get_value(self, position):
        """Value of the data at a position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        shape : int | None
            Index of shape if any that is at the coordinates. Returns `None`
            if no shape is found.
        vertex : int | None
            Index of vertex if any that is at the coordinates. Returns `None`
            if no vertex is found.
        """
        if self._slice_input.ndisplay == 3:
            return (None, None)

        if self._is_moving:
            return self._moving_value

        coord = [position[i] for i in self._slice_input.displayed]

        # Check selected shapes
        value = None
        selected_index = list(self.selected_data)

        if len(selected_index) > 0:
            self.scale[self._slice_input.displayed]
            # Get the vertex sizes. They need to be rescaled by a few parameters:
            # - scale_factor, because vertex sizes are zoom-invariant
            # - scale, because vertex sizes are not affected by scale (unlike in Points)
            # - 2, because the radius is what we need

            if self._mode == Mode.SELECT:
                # Check if inside vertex of interaction box or rotation handle
                box = self._selected_box[Box.WITH_HANDLE]
                distances = abs(box - coord)

                # Check if any matching vertices
                matches = np.all(
                    distances <= self._normalized_vertex_radius, axis=1
                ).nonzero()
                if len(matches[0]) > 0:
                    value = (selected_index[0], matches[0][-1])
            elif self._mode in (
                [Mode.DIRECT, Mode.VERTEX_INSERT, Mode.VERTEX_REMOVE]
            ):
                # Check if inside vertex of shape
                inds = np.isin(self._data_view.displayed_index, selected_index)
                vertices = self._data_view.displayed_vertices[inds]
                distances = abs(vertices - coord)

                # Check if any matching vertices
                matches = np.all(
                    distances <= self._normalized_vertex_radius, axis=1
                ).nonzero()[0]
                if len(matches) > 0:
                    index = inds.nonzero()[0][matches[-1]]
                    shape = self._data_view.displayed_index[index]
                    vals, idx = np.unique(
                        self._data_view.displayed_index, return_index=True
                    )
                    shape_in_list = list(vals).index(shape)
                    value = (shape, index - idx[shape_in_list])

        if value is None:
            # Check if mouse inside shape
            shape = self._data_view.inside(coord)
            value = (shape, None)

        return value

    def _get_value_3d(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Tuple[Union[float, int, None], None]:
        """Get the layer data value along a ray

        Parameters
        ----------
        start_point : np.ndarray
            The start position of the ray used to interrogate the data.
        end_point : np.ndarray
            The end position of the ray used to interrogate the data.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the Viewer.

        Returns
        -------
        value
            The data value along the supplied ray.
        vertex : None
            Index of vertex if any that is at the coordinates. Always returns `None`.
        """
        value, _ = self._get_index_and_intersection(
            start_point=start_point,
            end_point=end_point,
            dims_displayed=dims_displayed,
        )

        return value, None

    def _get_index_and_intersection(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Tuple[Union[None, float, int], Union[None, np.ndarray]]:
        """Get the shape index and intersection point of the first shape
        (i.e., closest to start_point) along the specified 3D line segment.

        Note: this method is meant to be used for 3D intersection and returns
        (None, None) when used in 2D (i.e., len(dims_displayed) is 2).

        Parameters
        ----------
        start_point : np.ndarray
            The start position of the ray used to interrogate the data in
            layer coordinates.
        end_point : np.ndarray
            The end position of the ray used to interrogate the data in
            layer coordinates.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the Viewer.

        Returns
        -------
        value Union[None, float, int]
            The data value along the supplied ray.
        intersection_point : Union[None, np.ndarray]
            (n,) array containing the point where the ray intersects the first shape
            (i.e., the shape most in the foreground). The coordinate is in layer
            coordinates.
        """
        if len(dims_displayed) != 3:
            # return None if in 2D mode
            return None, None
        if (start_point is None) or (end_point is None):
            # return None if the ray doesn't intersect the data bounding box
            return None, None

        # Get the normal vector of the click plane
        start_position, ray_direction = nd_line_segment_to_displayed_data_ray(
            start_point=start_point,
            end_point=end_point,
            dims_displayed=dims_displayed,
        )
        value, intersection = self._data_view._inside_3d(
            start_position, ray_direction
        )

        # add the full nD coords to intersection
        intersection_point = start_point.copy()
        intersection_point[dims_displayed] = intersection

        return value, intersection_point

    def get_index_and_intersection(
        self,
        position: np.ndarray,
        view_direction: np.ndarray,
        dims_displayed: List[int],
    ) -> Tuple[Union[float, int, None], Union[npt.NDArray, None]]:
        """Get the shape index and intersection point of the first shape
        (i.e., closest to start_point) "under" a mouse click.

        See examples/add_points_on_nD_shapes.py for example usage.

        Parameters
        ----------
        position : tuple
            Position in either data or world coordinates.
        view_direction : Optional[np.ndarray]
            A unit vector giving the direction of the ray in nD world coordinates.
            The default value is None.
        dims_displayed : Optional[List[int]]
            A list of the dimensions currently being displayed in the viewer.
            The default value is None.

        Returns
        -------
        value
            The data value along the supplied ray.
        intersection_point : np.ndarray
            (n,) array containing the point where the ray intersects the first shape
            (i.e., the shape most in the foreground). The coordinate is in layer
            coordinates.
        """
        start_point, end_point = self.get_ray_intersections(
            position, view_direction, dims_displayed
        )
        if (start_point is not None) and (end_point is not None):
            shape_index, intersection_point = self._get_index_and_intersection(
                start_point=start_point,
                end_point=end_point,
                dims_displayed=dims_displayed,
            )
        else:
            shape_index = None
            intersection_point = None
        return shape_index, intersection_point

    def move_to_front(self):
        """Moves selected objects to be displayed in front of all others."""
        if len(self.selected_data) == 0:
            return
        new_z_index = max(self._data_view._z_index) + 1
        for index in self.selected_data:
            self._data_view.update_z_index(index, new_z_index)
        self.refresh()

    def move_to_back(self):
        """Moves selected objects to be displayed behind all others."""
        if len(self.selected_data) == 0:
            return
        new_z_index = min(self._data_view._z_index) - 1
        for index in self.selected_data:
            self._data_view.update_z_index(index, new_z_index)
        self.refresh()

    def _copy_data(self):
        """Copy selected shapes to clipboard."""
        if len(self.selected_data) > 0:
            index = list(self.selected_data)
            self._clipboard = {
                'data': [
                    deepcopy(self._data_view.shapes[i])
                    for i in self._selected_data
                ],
                'edge_color': deepcopy(self._data_view._edge_color[index]),
                'face_color': deepcopy(self._data_view._face_color[index]),
                'features': deepcopy(self.features.iloc[index]),
                'indices': self._data_slice.point,
                'text': self.text._copy(index),
            }
        else:
            self._clipboard = {}

    def _paste_data(self):
        """Paste any shapes from clipboard and then selects them."""
        cur_shapes = self.nshapes
        if len(self._clipboard.keys()) > 0:
            # Calculate offset based on dimension shifts
            offset = [
                self._data_slice.point[i] - self._clipboard['indices'][i]
                for i in self._slice_input.not_displayed
            ]

            self._feature_table.append(self._clipboard['features'])
            self.text._paste(**self._clipboard['text'])

            # Add new shape data
            for i, s in enumerate(self._clipboard['data']):
                shape = deepcopy(s)
                data = copy(shape.data)
                not_disp = self._slice_input.not_displayed
                data[:, not_disp] = data[:, not_disp] + np.array(offset)
                shape.data = data
                face_color = self._clipboard['face_color'][i]
                edge_color = self._clipboard['edge_color'][i]
                self._data_view.add(
                    shape, face_color=face_color, edge_color=edge_color
                )

            self.selected_data = set(
                range(cur_shapes, cur_shapes + len(self._clipboard['data']))
            )

            self.move_to_front()

    def to_masks(self, mask_shape=None):
        """Return an array of binary masks, one for each shape.

        Parameters
        ----------
        mask_shape : np.ndarray | tuple | None
            tuple defining shape of mask to be generated. If non specified,
            takes the max of all the vertices

        Returns
        -------
        masks : np.ndarray
            Array where there is one binary mask for each shape
        """
        if mask_shape is None:
            # See https://github.com/napari/napari/issues/2778
            # Point coordinates land on pixel centers. We want to find the
            # smallest shape that will hold the largest point in the data,
            # using rounding.
            mask_shape = np.round(self._extent_data[1]) + 1

        mask_shape = np.ceil(mask_shape).astype('int')
        masks = self._data_view.to_masks(mask_shape=mask_shape)

        return masks

    def to_labels(self, labels_shape=None):
        """Return an integer labels image.

        Parameters
        ----------
        labels_shape : np.ndarray | tuple | None
            Tuple defining shape of labels image to be generated. If non
            specified, takes the max of all the vertiecs

        Returns
        -------
        labels : np.ndarray
            Integer array where each value is either 0 for background or an
            integer up to N for points inside the shape at the index value - 1.
            For overlapping shapes z-ordering will be respected.
        """
        if labels_shape is None:
            # See https://github.com/napari/napari/issues/2778
            # Point coordinates land on pixel centers. We want to find the
            # smallest shape that will hold the largest point in the data,
            # using rounding.
            labels_shape = np.round(self._extent_data[1]) + 1

        labels_shape = np.ceil(labels_shape).astype('int')
        labels = self._data_view.to_labels(labels_shape=labels_shape)

        return labels

    def _set_drag_start(self, coordinates: Tuple[float, ...]) -> List[float]:
        """Indicate where in data space a drag event started.

        Sets the coordinates relative to the center of the bounding box of a shape and returns the position
        of where a drag event of a shape started.

        Parameters
        ----------
        coordinates : Tuple[float, ...]
            The position in image data space where dragging started.

        Returns
        -------
        coord: List[float, ...]
            The coordinates of where a shape drag event started.
        """
        coord = [coordinates[i] for i in self._slice_input.displayed]
        if self._drag_start is None and len(self.selected_data) > 0:
            center = self._selected_box[Box.CENTER]
            self._drag_start = coord - center
        return coord

    def _move_active_element_under_cursor(
        self, coordinates: Tuple[float, ...]
    ) -> None:
        """Moves object at given mouse position and set of indices.

        Parameters
        ----------
        coordinates : Tuple[float, ...]
            Position of mouse cursor in data coordinates.
        """
        # If nothing selected return
        if len(self.selected_data) == 0:
            return

        # This is None when moving whole shape(s), but not when moving a vertex
        vertex = self._moving_value[1]

        if self._mode in (
            [Mode.SELECT, Mode.ADD_RECTANGLE, Mode.ADD_ELLIPSE, Mode.ADD_LINE]
        ):
            # explicit is not None because of possibility of vertex being 0
            if self._mode == Mode.SELECT and not self._is_moving:
                if vertex is not None:
                    vertex_indices = ((vertex,),)
                else:
                    vertex_indices = tuple(
                        tuple(
                            vertex_index
                            for vertex_index, coord in enumerate(self.data[i])
                        )
                        for i in self.selected_data
                    )
                self.events.data(
                    value=self.data,
                    action=ActionType.CHANGING,
                    data_indices=tuple(self.selected_data),
                    vertex_indices=vertex_indices,
                )

            coord = self._set_drag_start(coordinates)
            self._moving_coordinates = coordinates
            self._is_moving = True
            if vertex is None:
                # Check where dragging box from to move whole object
                center = self._selected_box[Box.CENTER]
                shift = coord - center - self._drag_start
                for index in self.selected_data:
                    self._data_view.shift(index, shift)
                self._selected_box = self._selected_box + shift
                self.refresh()
            elif vertex < Box.LEN:
                # Corner / edge vertex is being dragged so resize object
                box = self._selected_box
                if self._fixed_vertex is None:
                    self._fixed_index = (vertex + 4) % Box.LEN
                    self._fixed_vertex = box[self._fixed_index]

                handle_offset = box[Box.HANDLE] - box[Box.CENTER]
                if np.linalg.norm(handle_offset) == 0:
                    handle_offset = [1, 1]
                handle_offset_norm = handle_offset / np.linalg.norm(
                    handle_offset
                )

                rot = np.array(
                    [
                        [handle_offset_norm[0], -handle_offset_norm[1]],
                        [handle_offset_norm[1], handle_offset_norm[0]],
                    ]
                )
                inv_rot = np.linalg.inv(rot)

                fixed = self._fixed_vertex
                new = list(coord)

                box_center = box[Box.CENTER]
                if self._fixed_aspect and self._fixed_index % 2 == 0:
                    # corner
                    new = (box[vertex] - box_center) / np.linalg.norm(
                        box[vertex] - box_center
                    ) * np.linalg.norm(new - box_center) + box_center

                if self._fixed_index % 2 == 0:
                    # corner selected
                    drag_scale = (inv_rot @ (new - fixed)) / (
                        inv_rot @ (box[vertex] - fixed)
                    )
                elif self._fixed_index % 4 == 3:
                    # top or bottom selected
                    drag_scale = np.array(
                        [
                            (inv_rot @ (new - fixed))[0]
                            / (inv_rot @ (box[vertex] - fixed))[0],
                            1,
                        ]
                    )
                else:
                    # left or right selected
                    drag_scale = np.array(
                        [
                            1,
                            (inv_rot @ (new - fixed))[1]
                            / (inv_rot @ (box[vertex] - fixed))[1],
                        ]
                    )

                # prevent box from shrinking below a threshold size
                size = (np.linalg.norm(box[Box.TOP_LEFT] - box_center),)
                if (
                    np.linalg.norm(size * drag_scale)
                    < self._normalized_vertex_radius
                ):
                    drag_scale[:] = 1
                # on vertical/horizontal drags we get scale of 0
                # when we actually simply don't want to scale
                drag_scale[drag_scale == 0] = 1

                # check orientation of box
                if abs(handle_offset_norm[0]) == 1:
                    for index in self.selected_data:
                        self._data_view.scale(
                            index, drag_scale, center=self._fixed_vertex
                        )
                    self._scale_box(drag_scale, center=self._fixed_vertex)
                else:
                    scale_mat = np.array(
                        [[drag_scale[0], 0], [0, drag_scale[1]]]
                    )
                    transform = rot @ scale_mat @ inv_rot
                    for index in self.selected_data:
                        self._data_view.shift(index, -self._fixed_vertex)
                        self._data_view.transform(index, transform)
                        self._data_view.shift(index, self._fixed_vertex)
                    self._transform_box(transform, center=self._fixed_vertex)
                self.refresh()
            elif vertex == 8:
                # Rotation handle is being dragged so rotate object
                handle = self._selected_box[Box.HANDLE]
                self._fixed_vertex = self._selected_box[Box.CENTER]
                offset = handle - self._fixed_vertex
                self._drag_start = -np.degrees(
                    np.arctan2(offset[0], -offset[1])
                )

                new_offset = coord - self._fixed_vertex
                new_angle = -np.degrees(
                    np.arctan2(new_offset[0], -new_offset[1])
                )
                fixed_offset = handle - self._fixed_vertex
                fixed_angle = -np.degrees(
                    np.arctan2(fixed_offset[0], -fixed_offset[1])
                )

                if self._fixed_aspect:
                    angle = np.round(new_angle / 45) * 45 - fixed_angle
                else:
                    angle = new_angle - fixed_angle

                for index in self.selected_data:
                    self._data_view.rotate(
                        index, angle, center=self._fixed_vertex
                    )
                self._rotate_box(angle, center=self._fixed_vertex)
                self.refresh()

        elif (
            self._mode
            in [
                Mode.DIRECT,
                Mode.ADD_PATH,
                Mode.ADD_POLYGON,
                Mode.ADD_POLYGON_LASSO,
            ]
            and vertex is not None
        ):
            if not self._is_moving:
                self.events.data(
                    value=self.data,
                    action=ActionType.CHANGING,
                    data_indices=tuple(self.selected_data),
                    vertex_indices=((vertex,),),
                )
            self._moving_coordinates = coordinates
            self._is_moving = True
            index = self._moving_value[0]
            shape_type = type(self._data_view.shapes[index])
            if shape_type == Ellipse:
                # TODO: Implement DIRECT vertex moving of ellipse
                pass
            else:
                new_type = Polygon if shape_type == Rectangle else None
                vertices = self._data_view.shapes[index].data
                vertices[vertex] = coordinates
                self._data_view.edit(index, vertices, new_type=new_type)
                shapes = self.selected_data
                self._selected_box = self.interaction_box(shapes)
                self.refresh()
