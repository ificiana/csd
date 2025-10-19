import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel
from scipy.spatial.transform import Rotation as R


class Drone(BaseModel):
    irl_time: dict[float, float]
    pos: dict[float, tuple[float, float, float]]
    thrust: dict[float, tuple[float, float, float]]

    @property
    def irl_time_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            self.irl_time, orient="index", columns=["irl_time"]
        ).rename_axis("time")

    @property
    def pos_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            self.pos, orient="index", columns=["x", "y", "z"]
        ).rename_axis("time")

    @property
    def thrust_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            self.thrust, orient="index", columns=["tx", "ty", "tz"]
        ).rename_axis("time")

    @property
    def df(self) -> pd.DataFrame:
        return pd.concat([self.irl_time_df, self.pos_df, self.thrust_df], axis=1)


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

    @property
    def pos_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            self.pos, orient="index", columns=["x", "y", "z"]
        ).rename_axis("time")

    @property
    def vel_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            self.vel, orient="index", columns=["vx", "vy", "vz"]
        ).rename_axis("time")

    @property
    def acc_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            self.acc, orient="index", columns=["ax", "ay", "az"]
        ).rename_axis("time")

    @property
    def rot_df(self) -> pd.DataFrame:
        # Flatten rotation matrices into 9 columns
        flat_rot = {
            t: [val for row in mat for val in row] for t, mat in self.rot.items()
        }
        cols = [f"R{i}{j}" for i in range(3) for j in range(3)]
        df_rot = pd.DataFrame.from_dict(
            flat_rot, orient="index", columns=cols
        ).rename_axis("time")

        # Compute Euler angles (roll, pitch, yaw) in radians
        rot_matrices = df_rot.values.reshape(-1, 3, 3)
        r = R.from_matrix(rot_matrices)
        euler_angles = r.as_euler(
            "xyz", degrees=True
        )  # can use degrees=True for readability

        # Add Euler angles as new columns
        df_rot[["roll", "pitch", "yaw"]] = euler_angles
        return df_rot

    @property
    def ang_acc_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            self.ang_acc, orient="index", columns=["Ax", "Ay", "Az"]
        ).rename_axis("time")

    @property
    def ang_vel_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            self.ang_vel, orient="index", columns=["Wx", "Wy", "Wz"]
        ).rename_axis("time")
    
    @property
    def irl_time_df(self) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(
            self.irl_time, orient="index", columns=["irl_time"]
        ).rename_axis("time")
        df["eff"] = df.index / df["irl_time"]
        return df

    @property
    def df(self) -> pd.DataFrame:
        dfs = [
            self.irl_time_df,
            self.pos_df,
            self.vel_df,
            self.acc_df,
            self.ang_vel_df,
            self.ang_acc_df,
            self.rot_df,
        ]
        return pd.concat(dfs, axis=1)


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
    """

    # Normalize columns
    cols = [col] if isinstance(col, str) else col

    # Validate columns
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in DataFrame.")

    merge = merge or len(cols) == 1

    # ----- TRAJECTORY MODE -----
    if traj:
        # Sort by index (time)
        df_sorted = df.sort_index()

        if len(cols) == 2:
            fig = go.Figure(
                go.Scatter(
                    x=df_sorted[cols[0]],
                    y=df_sorted[cols[1]],
                    mode="lines+markers",
                    name=f"{cols[0]} vs {cols[1]}",
                )
            )
            fig.update_layout(
                title=title or f"2D Trajectory: {cols[0]} vs {cols[1]}",
                xaxis_title=cols[0],
                yaxis_title=cols[1],
                template="plotly_white",
                height=600,
            )

        elif len(cols) == 3:
            fig = go.Figure(
                go.Scatter3d(
                    x=df_sorted[cols[0]],
                    y=df_sorted[cols[1]],
                    z=df_sorted[cols[2]],
                    mode="lines+markers",
                    marker=dict(size=4),
                    name=f"{cols[0]}-{cols[1]}-{cols[2]}",
                )
            )
            fig.update_layout(
                title=title or f"3D Trajectory: {cols[0]} vs {cols[1]} vs {cols[2]}",
                scene=dict(
                    xaxis_title=cols[0],
                    yaxis_title=cols[1],
                    zaxis_title=cols[2],
                ),
                template="plotly_white",
                height=400,
            )
        else:
            raise ValueError("Trajectory mode only supports 2 or 3 columns.")
        fig.show()
        return

    # ----- TIME SERIES MODE -----
    if merge:
        # Single merged plot
        fig = go.Figure()
        for c in cols:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
        fig.update_layout(
            title=title or f"{', '.join(cols)} vs Time",
            xaxis_title=xlabel,
            yaxis_title=ylabel or ", ".join(cols),
            hovermode="x unified",
            template="plotly_white",
            height=400,
        )
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
            fig.add_trace(
                go.Scatter(x=df.index, y=df[c], mode="lines", name=c), row=i, col=1
            )

        fig.update_layout(
            title=title or f"{', '.join(cols)} vs Time",
            height=200 * len(cols) + 100,
            hovermode="x unified",
            template="plotly_white",
            showlegend=False,
        )

        # Label axes
        if ylabel:
            for i in range(1, len(cols) + 1):
                fig.update_yaxes(title_text=ylabel, row=i, col=1)
        else:
            for i, c in enumerate(cols, start=1):
                fig.update_yaxes(title_text=c, row=i, col=1)

    fig.show()


def plot_graphs(data: Data):
    plot_time_series(data.drone_0.df, ["x", "y", "z"], title="Drone 0 Position")
    plot_time_series(
        data.drone_0.df, ["x", "y", "z"], title="Drone 0 Position", traj=True
    )
    plot_time_series(data.drone_1.df, ["x", "y", "z"], title="Drone 1 Position")
    plot_time_series(
        data.drone_1.df, ["x", "y", "z"], title="Drone 1 Position", traj=True
    )
    plot_time_series(data.drone_2.df, ["x", "y", "z"], title="Drone 2 Position")
    plot_time_series(
        data.drone_2.df, ["x", "y", "z"], title="Drone 2 Position", traj=True
    )
    plot_time_series(data.drone_3.df, ["x", "y", "z"], title="Drone 3 Position")
    plot_time_series(
        data.drone_3.df, ["x", "y", "z"], title="Drone 3 Position", traj=True
    )
    plot_time_series(data.cube.df, ["x", "y", "z"], title="Cube COM position")
    plot_time_series(
        data.cube.df, ["x", "y", "z"], title="Cube COM position", traj=True
    )
    plot_time_series(data.cube.df, ["vx", "vy", "vz"], title="Cube velocity")
    plot_time_series(data.cube.df, ["ax", "ay", "az"], title="Cube acceleration")
    plot_time_series(data.cube.df, ["Wx", "Wy", "Wz"], title="Cube angular velocity")
    plot_time_series(
        data.cube.df, ["Ax", "Ay", "Az"], title="Cube angular acceleration"
    )
    plot_time_series(data.cube.df, ["yaw", "pitch", "roll"], title="Yaw/Pitch/Roll")
    plot_time_series(data.cube.df, "irl_time", title="IRL vs SIM")
    plot_time_series(data.cube.df, "eff", title="IRL vs SIM eff")
