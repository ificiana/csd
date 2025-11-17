from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel
from scipy.spatial.transform import Rotation as R

from config import MAX_TAKEOFF_FRACTION, MAX_TRANSL_FRACTION, DRONE_MAX_THRUST


def _dict_to_df(
    data_dict: dict, columns: list[str], index_name: str = "time"
) -> pd.DataFrame:
    """Helper to convert dict to DataFrame with consistent formatting."""
    return pd.DataFrame.from_dict(
        data_dict, orient="index", columns=columns
    ).rename_axis(index_name)


def _vector_magnitude(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Calculate vector magnitude from DataFrame columns."""
    return (
        df[f"{prefix}x"] ** 2 + df[f"{prefix}y"] ** 2 + df[f"{prefix}z"] ** 2
    ) ** 0.5


class Drone(BaseModel):
    irl_time: dict[float, float]
    pos: dict[float, tuple[float, float, float]]
    command: dict[float, tuple[float, float, float]]
    thrust: dict[float, tuple[float, float, float]]
    feedforward: dict[float, tuple[float, float, float]]
    feedback: dict[float, tuple[float, float, float]]
    attitude: dict[float, tuple[float, float, float]]

    @property
    def irl_time_df(self) -> pd.DataFrame:
        return _dict_to_df(self.irl_time, ["irl_time"])

    @property
    def pos_df(self) -> pd.DataFrame:
        return _dict_to_df(self.pos, ["x", "y", "z"])

    @property
    def thrust_df(self) -> pd.DataFrame:
        return pd.concat(
            [
                _dict_to_df(self.thrust, ["tx", "ty", "tz"]),
                _dict_to_df(self.command, ["cx", "cy", "cz"]),
            ],
            axis=1,
        )

    @property
    def control_df(self) -> pd.DataFrame:
        """Control force breakdown for analysis."""
        return pd.concat(
            [
                _dict_to_df(self.feedforward, ["ffx", "ffy", "ffz"]),
                _dict_to_df(self.feedback, ["fbx", "fby", "fbz"]),
                _dict_to_df(self.attitude, ["attx", "atty", "attz"]),
            ],
            axis=1,
        )

    @property
    def control_magnitude_df(self) -> pd.DataFrame:
        """Magnitude of each control component."""
        df = self.control_df
        return pd.DataFrame(
            {
                "vertical feedforward": df["ffz"].abs(),
                "horizontal feedforward": (df["ffx"] ** 2 + df["ffy"] ** 2) ** 0.5,
                "position feedback": _vector_magnitude(df, "fb"),
                "attitude correction": _vector_magnitude(df, "att"),
            }
        )

    @property
    def control_proportion_df(self) -> pd.DataFrame:
        """Normalized control component magnitudes."""
        mag_df = self.control_magnitude_df
        max_takeoff = MAX_TAKEOFF_FRACTION * DRONE_MAX_THRUST
        residual_thrust = DRONE_MAX_THRUST - max_takeoff
        max_transl = residual_thrust * MAX_TRANSL_FRACTION
        max_control = residual_thrust - max_transl

        return pd.DataFrame(
            {
                "vertical feedforward": (
                    mag_df["vertical feedforward"] / max_takeoff
                    if max_takeoff > 0
                    else pd.Series(0.0, index=mag_df.index)
                ),
                "horizontal feedforward": (
                    mag_df["horizontal feedforward"] / max_transl
                    if max_transl > 0
                    else pd.Series(0.0, index=mag_df.index)
                ),
                "position feedback": (
                    mag_df["position feedback"] / max_control
                    if max_control > 0
                    else pd.Series(0.0, index=mag_df.index)
                ),
                "attitude correction": (
                    mag_df["attitude correction"] / max_control
                    if max_control > 0
                    else pd.Series(0.0, index=mag_df.index)
                ),
            }
        )

    @property
    def df(self) -> pd.DataFrame:
        return pd.concat(
            [
                self.irl_time_df,
                self.pos_df,
                self.thrust_df,
                self.control_df,
                self.control_magnitude_df,
            ],
            axis=1,
        )


class Cube(BaseModel):
    irl_time: dict[float, float]
    pos: dict[float, tuple[float, float, float]]
    rot: dict[
        float,
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
    ]
    acc: dict[float, tuple[float, float, float]]
    vel: dict[float, tuple[float, float, float]]
    ang_acc: dict[float, tuple[float, float, float]]
    ang_vel: dict[float, tuple[float, float, float]]
    wind: dict[float, tuple[float, float, float]]

    @property
    def pos_df(self) -> pd.DataFrame:
        return _dict_to_df(self.pos, ["x", "y", "z"])

    @property
    def vel_df(self) -> pd.DataFrame:
        return _dict_to_df(self.vel, ["vx", "vy", "vz"])

    @property
    def acc_df(self) -> pd.DataFrame:
        return _dict_to_df(self.acc, ["ax", "ay", "az"])

    @property
    def ang_acc_df(self) -> pd.DataFrame:
        return _dict_to_df(self.ang_acc, ["Ax", "Ay", "Az"])

    @property
    def ang_vel_df(self) -> pd.DataFrame:
        return _dict_to_df(self.ang_vel, ["Wx", "Wy", "Wz"])

    @property
    def wind_df(self) -> pd.DataFrame:
        return _dict_to_df(self.wind, ["windx", "windy", "windz"])

    @property
    def rot_df(self) -> pd.DataFrame:
        flat_rot = {
            t: [val for row in mat for val in row] for t, mat in self.rot.items()
        }
        cols = [f"R{i}{j}" for i in range(3) for j in range(3)]
        df_rot = _dict_to_df(flat_rot, cols)

        rot_matrices = df_rot.values.reshape(-1, 3, 3)
        r = R.from_matrix(rot_matrices)
        df_rot[["roll", "pitch", "yaw"]] = r.as_euler("xyz", degrees=True)
        return df_rot

    @property
    def irl_time_df(self) -> pd.DataFrame:
        df = _dict_to_df(self.irl_time, ["irl_time"])
        df["eff"] = df.index / df["irl_time"]
        return df

    @property
    def df(self) -> pd.DataFrame:
        return pd.concat(
            [
                self.irl_time_df,
                self.pos_df,
                self.vel_df,
                self.acc_df,
                self.ang_vel_df,
                self.ang_acc_df,
                self.rot_df,
                self.wind_df,
            ],
            axis=1,
        )


class Data(BaseModel):
    drone_0: Drone
    drone_1: Drone
    drone_2: Drone
    drone_3: Drone
    cube: Cube

    @property
    def drones(self) -> list[Drone]:
        return [self.drone_0, self.drone_1, self.drone_2, self.drone_3]

    @property
    def drone_dfs(self) -> dict[int, pd.DataFrame]:
        return {i: drone.df for i, drone in enumerate(self.drones)}


def plot_time_series(
    df: pd.DataFrame,
    col: str | list[str],
    *,
    title: str | None = None,
    xlabel: str = "Time (s)",
    ylabel: str | None = None,
    traj: bool = False,
    merge: bool = False,
    bw: bool = False,
):
    """
    Fully interactive zoomable plot using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with a time-like index.
    col : str | list[str]
        Column name(s) to plot.
    title : str, optional
        Plot title.
    xlabel : str, default "Time (s)"
        X-axis label.
    ylabel : str, optional
        Y-axis label (ignored for trajectory plots).
    traj : bool, default False
        If True:
          - For 2 columns → plot 2D trajectory (col[0] vs col[1])
          - For 3 columns → plot 3D trajectory (col[0] vs col[1] vs col[2])
    merge : bool, default False
        If True, plot all series in a single subplot instead of separate subplots.
    bw : bool, default False
        If True, use black & white print-friendly styling.
    """

    # Normalize columns
    cols = [col] if isinstance(col, str) else col

    # Validate columns
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in DataFrame.")

    merge = merge or len(cols) == 1

    # Styling constants - BW or color
    if bw:
        STYLES: dict[str, list | None] = {
            'line_styles': ['solid', 'dash', 'dot', 'dashdot'],
            'line_widths': [2.5, 2.5, 2.5, 2.5],
            'colors': ['black', 'rgb(50,50,50)', 'rgb(100,100,100)', 'rgb(150,150,150)'],
        }
        plot_bg: str | None = 'white'
        paper_bg: str | None = 'white'
        grid_color: str | None = 'lightgray'
        zero_color: str | None = 'gray'
        text_color: str | None = 'black'
        marker_color: str | None = 'black'
        line_color_single: str | None = 'black'
        legend_border: str | None = 'black'
    else:
        STYLES = {
            'line_styles': ['solid', 'solid', 'solid', 'solid'],
            'line_widths': [2, 2, 2, 2],
            'colors': None,  # Use plotly default colors
        }
        plot_bg = None
        paper_bg = None
        grid_color = None
        zero_color = None
        text_color = None
        marker_color = None
        line_color_single = None
        legend_border = None

    # ----- TRAJECTORY MODE -----
    if traj:
        # Sort by index (time)
        df_sorted = df.sort_index()

        if len(cols) == 2:
            line_dict: dict[str, Any] = {'width': 2}
            marker_dict: dict[str, Any] = {'size': 4, 'symbol': 'circle'}
            if bw:
                line_dict['color'] = 'black'
                marker_dict['color'] = 'black'
                
            fig = go.Figure(
                go.Scatter(
                    x=df_sorted[cols[0]],
                    y=df_sorted[cols[1]],
                    mode="lines+markers",
                    name=f"{cols[0]} vs {cols[1]}",
                    line=line_dict,
                    marker=marker_dict,
                )
            )
            x_label = (
                f"{cols[0]} ({ylabel.split('(')[1].split(')')[0]})"
                if ylabel and "(" in ylabel
                else cols[0]
            )
            y_label = (
                f"{cols[1]} ({ylabel.split('(')[1].split(')')[0]})"
                if ylabel and "(" in ylabel
                else cols[1]
            )
            
            layout_dict_2d = {
                'title': title or f"2D Trajectory: {cols[0]} vs {cols[1]}",
                'xaxis_title': x_label,
                'yaxis_title': y_label,
                'template': "plotly_white",
                'height': 600,
            }
            if bw:
                layout_dict_2d['title'] = {'text': layout_dict_2d['title'], 'font': {'size': 16, 'color': text_color}} # type: ignore
                layout_dict_2d['paper_bgcolor'] = paper_bg # type: ignore
                layout_dict_2d['plot_bgcolor'] = plot_bg # type: ignore
                layout_dict_2d['font'] = {'color': text_color} # type: ignore
                layout_dict_2d['xaxis'] = {'showgrid': True, 'gridcolor': grid_color, 'zeroline': True, 'zerolinecolor': zero_color} # type: ignore
                layout_dict_2d['yaxis'] = {'showgrid': True, 'gridcolor': grid_color, 'zeroline': True, 'zerolinecolor': zero_color} # type: ignore
            fig.update_layout(**layout_dict_2d) # type: ignore

        elif len(cols) == 3:
            marker_dict: dict[str, Any] = {'size': 3, 'symbol': 'circle'}
            line_dict: dict[str, Any] = {'width': 3}
            if bw:
                marker_dict['color'] = 'black'
                line_dict['color'] = 'black'
                
            fig = go.Figure(
                go.Scatter3d(
                    x=df_sorted[cols[0]],
                    y=df_sorted[cols[1]],
                    z=df_sorted[cols[2]],
                    mode="lines+markers",
                    marker=marker_dict,
                    line=line_dict,
                    name=f"{cols[0]}-{cols[1]}-{cols[2]}",
                )
            )
            unit = (
                ylabel.split("(")[1].split(")")[0] if ylabel and "(" in ylabel else ""
            )
            
            scene_dict: dict[str, Any] = {
                'xaxis_title': f"{cols[0]} ({unit})" if unit else cols[0],
                'yaxis_title': f"{cols[1]} ({unit})" if unit else cols[1],
                'zaxis_title': f"{cols[2]} ({unit})" if unit else cols[2],
            }
            if bw:
                scene_dict['xaxis'] = {'showgrid': True, 'gridcolor': grid_color, 'backgroundcolor': 'white'}
                scene_dict['yaxis'] = {'showgrid': True, 'gridcolor': grid_color, 'backgroundcolor': 'white'}
                scene_dict['zaxis'] = {'showgrid': True, 'gridcolor': grid_color, 'backgroundcolor': 'white'}
            
            layout_dict_3d = {
                'title': title or f"3D Trajectory: {cols[0]} vs {cols[1]} vs {cols[2]}",
                'scene': scene_dict,
                'template': "plotly_white",
                'height': 600,
            }
            if bw:
                layout_dict_3d['title'] = {'text': layout_dict_3d['title'], 'font': {'size': 16, 'color': text_color}} # type: ignore
                layout_dict_3d['paper_bgcolor'] = paper_bg # type: ignore
                layout_dict_3d['font'] = {'color': text_color} # type: ignore
            fig.update_layout(**layout_dict_3d) # type: ignore
        else:
            raise ValueError("Trajectory mode only supports 2 or 3 columns.")
        fig.show()
        return

    # ----- TIME SERIES MODE -----
    if merge:
        # Single merged plot
        fig = go.Figure()
        for idx, c in enumerate(cols):
            style_idx = idx % len(STYLES['line_styles'])  # type: ignore
            trace_dict_merge: dict[str, Any] = {
                'x': df.index,
                'y': df[c],
                'mode': "lines",
                'name': c,
            }
            if bw:
                trace_dict_merge['line'] = {
                    'color': STYLES['colors'][style_idx],  # type: ignore
                    'width': STYLES['line_widths'][style_idx],  # type: ignore
                    'dash': STYLES['line_styles'][style_idx]  # type: ignore
                }
            else:
                trace_dict_merge['line'] = {'width': STYLES['line_widths'][style_idx]}  # type: ignore
            fig.add_trace(go.Scatter(**trace_dict_merge))
        
        layout_dict_merge = {
            'title': title or f"{', '.join(cols)} vs Time",
            'xaxis_title': xlabel or "Time (s)",
            'yaxis_title': ylabel or ", ".join(cols),
            'hovermode': "x unified",
            'template': "plotly_white",
            'height': 400,
        }
        if bw:
            layout_dict_merge['title'] = {'text': layout_dict_merge['title'], 'font': {'size': 16, 'color': text_color}} # type: ignore
            layout_dict_merge['paper_bgcolor'] = paper_bg # type: ignore
            layout_dict_merge['plot_bgcolor'] = plot_bg # type: ignore
            layout_dict_merge['font'] = {'color': text_color} # type: ignore
            layout_dict_merge['xaxis'] = {'showgrid': True, 'gridcolor': grid_color, 'zeroline': True, 'zerolinecolor': zero_color} # type: ignore
            layout_dict_merge['yaxis'] = {'showgrid': True, 'gridcolor': grid_color, 'zeroline': True, 'zerolinecolor': zero_color} # type: ignore
            layout_dict_merge['legend'] = {  # type: ignore
                'bgcolor': 'white',
                'bordercolor': legend_border,
                'borderwidth': 1,
                'font': {'color': text_color}
            }
        fig.update_layout(**layout_dict_merge) # type: ignore
    else:
        # Separate subplots
        fig = make_subplots(
            rows=len(cols),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=cols,
        )

        for i, c in enumerate(cols, start=1):
            trace_dict_sub = {
                'x': df.index,
                'y': df[c],
                'mode': "lines",
                'name': c,
            }
            if bw:
                trace_dict_sub['line'] = {'color': line_color_single, 'width': 2} # type: ignore
            else:
                trace_dict_sub['line'] = {'width': 2} # type: ignore
            fig.add_trace(go.Scatter(**trace_dict_sub), row=i, col=1)

        layout_dict_sub = {
            'title': title or f"{', '.join(cols)} vs Time",
            'height': 200 * len(cols) + 100,
            'hovermode': "x unified",
            'template': "plotly_white",
            'showlegend': False,
        }
        if bw:
            layout_dict_sub['title'] = {'text': layout_dict_sub['title'], 'font': {'size': 16, 'color': text_color}} # type: ignore
            layout_dict_sub['paper_bgcolor'] = paper_bg # type: ignore
            layout_dict_sub['plot_bgcolor'] = plot_bg # type: ignore
            layout_dict_sub['font'] = {'color': text_color} # type: ignore
        fig.update_layout(**layout_dict_sub) # type: ignore
        # Label axes with grid styling
        xaxis_dict: dict[str, Any] = {'title_text': xlabel or "Time (s)"}
        if bw:
            xaxis_dict['showgrid'] = True
            xaxis_dict['gridcolor'] = grid_color
            xaxis_dict['zeroline'] = True
            xaxis_dict['zerolinecolor'] = zero_color
        fig.update_xaxes(**xaxis_dict, row=len(cols), col=1)
        
        if ylabel:
            yaxis_dict_y: dict[str, Any] = {'title_text': ylabel}
            if bw:
                yaxis_dict_y['showgrid'] = True
                yaxis_dict_y['gridcolor'] = grid_color
                yaxis_dict_y['zeroline'] = True
                yaxis_dict_y['zerolinecolor'] = zero_color
            for i in range(1, len(cols) + 1):
                fig.update_yaxes(**yaxis_dict_y, row=i, col=1)
        else:
            for i, c in enumerate(cols, start=1):
                yaxis_dict_c: dict[str, Any] = {'title_text': c}
                if bw:
                    yaxis_dict_c['showgrid'] = True
                    yaxis_dict_c['gridcolor'] = grid_color
                    yaxis_dict_c['zeroline'] = True
                    yaxis_dict_c['zerolinecolor'] = zero_color
                fig.update_yaxes(**yaxis_dict_c, row=i, col=1)

    fig.show()


def plot_drone(drone: Drone, drone_id: int, bw: bool = False):
    """Plot drone telemetry (position, command, thrust, control forces).
    
    Args:
        drone: Drone instance with telemetry data.
        drone_id: Drone identifier (0-3).
        bw: If True, use black & white print-friendly styling.
    """
    # plot_time_series(
    #     drone.df,
    #     ["x", "y", "z"],
    #     title=f"Drone {drone_id} Position",
    #     ylabel="Position (m)",
    #     bw=bw,
    # )
    # plot_time_series(
    #     drone.df, ["x", "y", "z"], title=f"Drone {drone_id} Position", traj=True, bw=bw
    # )
    # plot_time_series(
    #     drone.df,
    #     ["cx", "cy", "cz"],
    #     title=f"Drone {drone_id} Command",
    #     ylabel="Command (N)",
    #     bw=bw,
    # )
    plot_time_series(
        drone.df,
        ["tx", "ty", "tz"],
        title=f"Drone {drone_id} Thrust",
        ylabel="Thrust (N)",
        bw=bw,
    )
    # plot_time_series(
    #     drone.control_magnitude_df,
    #     [
    #         "vertical feedforward",
    #         "horizontal feedforward",
    #         "position feedback",
    #         "attitude correction",
    #     ],
    #     title=f"Drone {drone_id} - Control Force Magnitudes",
    #     ylabel="Force (N)",
    #     merge=True,
    #     bw=bw,
    # )
    # plot_time_series(
    #     drone.control_proportion_df,
    #     [
    #         "vertical feedforward",
    #         "horizontal feedforward",
    #         "position feedback",
    #         "attitude correction",
    #     ],
    #     title=f"Drone {drone_id} - Control Force Proportions",
    #     ylabel="Proportion",
    #     merge=True,
    #     bw=bw,
    # )
    # for prefix, name in [
    #     ("ff", "Feedforward"),
    #     ("fb", "Feedback"),
    #     ("att", "Attitude Correction"),
    # ]:
    #     plot_time_series(
    #         drone.df,
    #         [f"{prefix}x", f"{prefix}y", f"{prefix}z"],
    #         title=f"Drone {drone_id} - {name} Forces",
    #         ylabel="Force (N)",
    #         bw=bw,
    #     )


def plot_graphs(data: Data, bw: bool = False):
    """Plot all telemetry data for drones and payload.
    
    Args:
        data: Data instance containing all telemetry.
        bw: If True, use black & white print-friendly styling.
    """
    for i, drone in enumerate(data.drones):
        plot_drone(drone, i, bw=bw)

    cube_plots = [
        (["x", "y", "z"], "Cube COM position", False, "Position (m)"),
        (["x", "y", "z"], "Cube COM position", True, "Position (m)"),
        (["vx", "vy", "vz"], "Cube velocity", False, "Velocity (m/s)"),
        (["ax", "ay", "az"], "Cube acceleration", False, "Acceleration (m/s²)"),
        (
            ["Wx", "Wy", "Wz"],
            "Cube angular velocity",
            False,
            "Angular velocity (rad/s)",
        ),
        (
            ["Ax", "Ay", "Az"],
            "Cube angular acceleration",
            False,
            "Angular acceleration (rad/s²)",
        ),
        (["yaw", "pitch", "roll"], "Yaw/Pitch/Roll", False, "Angle (deg)"),
        (["windx", "windy", "windz"], "Wind velocity", False, "Velocity (m/s)"),
    ]

    for cols, title, traj, ylabel in cube_plots:
        plot_time_series(data.cube.df, cols, title=title, traj=traj, ylabel=ylabel, bw=bw)
