import numpy as np
from vispy.gloo import VertexBuffer
from vispy.visuals.filters.base_filter import Filter


class TracksFilter(Filter):
    """TracksFilter.

    Custom vertex and fragment shaders for visualizing tracks quickly with
    vispy. The central assumption is that the tracks are rendered as a
    continuous vispy Line segment, with connections and colors defined when
    the visual is created.

    The shader simply changes the visibility and/or fading of the data according
    to the current_time and the associate time metadata for each vertex. This
    is scaled according to the tail and head length. Points ahead of the current time
    are rendered with alpha set to zero.

    Parameters
    ----------
    current_time : int, float
        the current time, which is typically the frame index, although this
        can be an arbitrary float
    tail_length : int, float
        the lower limit on length of the 'tail'
    head_length : int, float
        the upper limit on length of the 'tail'
    use_fade : bool
        this will enable/disable tail fading with time
    vertex_time : 1D array, list
        a vector describing the time associated with each vertex

    TODO
    ----
    - the track is still displayed, albeit with fading, once the track has
     finished but is still within the 'tail_length' window. Should it
     disappear?

    """

    VERT_SHADER = """
        varying vec4 v_track_color;
        void apply_track_shading() {

            float alpha;

            if ($a_vertex_time > $current_time + $head_length) {
                // this is a hack to minimize the frag shader rendering ahead
                // of the current time point due to interpolation
                // track should cut off sharply at current_time when head length is 0
                // see #6696 for details
                if ($head_length == 0){
                    // this prevents a track from being rendered ahead of the
                    // current time point incorrectly due to the interpolation
                    alpha = -1000.;
                } else {
                    alpha = 0.;
                }
            } else {
                // fade the track into the temporal distance, scaled by the
                // maximum tail and head length from the gui
                float fade = ($head_length + $current_time - $a_vertex_time) / ($tail_length + $head_length);
                alpha = clamp(1.0-fade, 0.0, 1.0);
            }

            // when use_fade is disabled, the entire track is visible
            if ($use_fade == 0) {
                alpha = 1.0;
            }

            // set the vertex alpha according to the fade
            v_track_color.a = alpha;
        }
    """

    FRAG_SHADER = """
        varying vec4 v_track_color;
        void apply_track_shading() {

            // if the alpha is below the threshold, discard the fragment
            if( v_track_color.a <= 0.0 ) {
                discard;
            }

            // interpolate
            gl_FragColor.a = clamp(v_track_color.a * gl_FragColor.a, 0.0, 1.0);
        }
    """

    def __init__(
        self,
        current_time: float = 0,
        tail_length: float = 30,
        head_length: float = 0,
        use_fade: bool = True,
        vertex_time: list | np.ndarray | None = None,
    ) -> None:
        super().__init__(
            vcode=self.VERT_SHADER, vpos=3, fcode=self.FRAG_SHADER, fpos=9
        )

        self.current_time = current_time
        self.tail_length = tail_length
        self.head_length = head_length
        self.use_fade = use_fade
        self.vertex_time = vertex_time

    @property
    def current_time(self) -> float:
        return self._current_time

    @current_time.setter
    def current_time(self, n: float):
        self._current_time = n
        if isinstance(n, slice):
            n = np.max(self._vertex_time)
        self.vshader['current_time'] = float(n)

    @property
    def use_fade(self) -> bool:
        return self._use_fade

    @use_fade.setter
    def use_fade(self, value: bool):
        self._use_fade = value
        self.vshader['use_fade'] = float(self._use_fade)

    @property
    def tail_length(self) -> float:
        return self._tail_length

    @tail_length.setter
    def tail_length(self, tail_length: float):
        self._tail_length = tail_length
        self.vshader['tail_length'] = float(self._tail_length)

    @property
    def head_length(self) -> float:
        return self._head_length

    @head_length.setter
    def head_length(self, head_length: float):
        self._head_length = head_length
        self.vshader['head_length'] = float(self._head_length)

    def _attach(self, visual):
        super()._attach(visual)

    @property
    def vertex_time(self):
        return self._vertex_time

    @vertex_time.setter
    def vertex_time(self, v_time):
        self._vertex_time = np.array(v_time).reshape(-1, 1).astype(np.float32)
        self.vshader['a_vertex_time'] = VertexBuffer(self.vertex_time)
