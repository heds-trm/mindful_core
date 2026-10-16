import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


# Source : https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html


def radar_factory(num_vars: int, frame="circle") -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    # noinspection PyUnresolvedReferences
    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path: Path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            # noinspection PyProtectedMember
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        @staticmethod
        def _close_line(line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def make_radar_figure(values: np.ndarray, names: list[str], title: str) -> Figure:
    if (len(values) != len(names)) or len(values.shape) != 1:
        raise ValueError

    values /= values.max()

    theta = radar_factory(len(values))

    figure, axis = plt.subplots(figsize=(9, 9), nrows=1, ncols=1, subplot_kw=dict(projection="radar"), squeeze=True)
    figure: Figure

    axis.set_rgrids([0.2, 0.4, 0.6, 0.8])
    axis.set_title(title, weight="bold", size="medium", position=(0.5, 1.1),
                   horizontalalignment="center", verticalalignment="center")

    axis.plot(theta, values)
    axis.fill(theta, values, alpha=0.25)
    axis.set_varlabels(names)

    return figure


def radar_from_data_frame(data_frame: pd.DataFrame,
                          title: str,
                          color="blue",
                          tick_size=0.25,
                          figure: Figure = None,
                          axis: Axes = None
                          ) -> tuple[Figure, Axes]:
    values = np.asarray(data_frame)
    values /= values.mean(axis=0).max()
    mean = values.mean(axis=0)
    stddev = values.std(axis=0)

    lower_std = mean - stddev
    upper_std = mean + stddev

    theta = radar_factory(len(mean))
    theta_fill = np.concatenate([theta, [np.pi * 2]])

    if (figure is None) or (axis is None):
        figure, axis = plt.subplots(figsize=(9, 9), nrows=1, ncols=1, subplot_kw=dict(projection="radar"), squeeze=True)
        axis.set_title(title, weight="bold", size="medium", position=(0.5, 1.1),
                       horizontalalignment="center", verticalalignment="center")
        tick_count = int(np.ceil(upper_std.max() / tick_size))
        axis.set_rgrids([tick_size * (tick + 1) for tick in range(tick_count)])
        axis.set_varlabels(data_frame.columns)

    axis.plot(theta, mean, color=color)
    axis.plot(theta, lower_std, color=color, alpha=0.5, linewidth=0.5)
    axis.plot(theta, upper_std, color=color, alpha=0.5, linewidth=0.5)

    lower_std = np.concatenate([lower_std, lower_std[:1]])
    upper_std = np.concatenate([upper_std, upper_std[:1]])
    axis.fill_between(theta_fill, lower_std, upper_std, alpha=0.25, color=color)

    return figure, axis
