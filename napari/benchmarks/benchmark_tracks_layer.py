import numpy as np

from napari.layers import Tracks

from .utils import Skip


class TracksSuite:
    param_names = ['size', 'n_tracks']
    params = [(5 * np.power(10, np.arange(7))).tolist(), [1, 10, 100, 1000]]

    skip_params = Skip(
        if_in_pr=lambda size, n_tracks: size > 500 or n_tracks > 10,
        always=lambda size, n_tracks: n_tracks * 5 > size,
    )
    # we skip cases where the number of tracks times five is larger than the size as it is not useful

    def setup(self, size, n_tracks):
        """
        Create tracks data
        """
        rng = np.random.default_rng(0)

        track_ids = rng.integers(n_tracks, size=size)
        time = np.zeros(len(track_ids))

        for value, counts in zip(
            *np.unique(track_ids, return_counts=True), strict=False
        ):
            t = rng.permutation(counts)
            time[track_ids == value] = t

        coordinates = rng.uniform(size=(size, 3))

        data = np.concatenate(
            (track_ids[:, None], time[:, None], coordinates),
            axis=1,
        )

        self.data = data

        # create layer for the update benchmark
        self.layer = Tracks(self.data)

    def time_create_layer(self, *_) -> None:
        Tracks(self.data)

    def time_update_layer(self, *_) -> None:
        self.layer.data = self.data


if __name__ == '__main__':
    from utils import run_benchmark

    run_benchmark()
