# %%
# Imports
import os
from scmrepo.git import Git
import numpy as np
import plotly.graph_objects as go

PACKAGE_ROOT = Git(root_dir=".").root_dir


# %%
# Define plotting functions
def plot_simulation(positions, frame_duration, k):
    every_k_positions = positions[::k]
    num_frames, num_uav, _ = every_k_positions.shape

    # Initialize scene bounds with maximum required extent
    center, half_extent = compute_scene_bounds(positions, buffer_ratio=0.1)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=every_k_positions[0, :, 0],
                y=every_k_positions[0, :, 1],
                z=every_k_positions[0, :, 2],
                mode="markers+text",
                text=[f"UAV {i}" for i in range(num_uav)],
                textposition="top center",
                marker=dict(size=6),
                name="UAVs",
            ),
        ],
        layout=go.Layout(
            scene=dict(
                xaxis=dict(range=[center[0] - half_extent, center[0] + half_extent]),
                yaxis=dict(range=[center[1] - half_extent, center[1] + half_extent]),
                zaxis=dict(range=[center[2] - half_extent, center[2] + half_extent]),
                aspectmode="cube",
            ),
            title="UAV Simulation Viewer",
            annotations=[],
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {
                                        "duration": frame_duration,
                                        "redraw": True,
                                    },
                                    "transition": {"duration": 0},
                                    "mode": "immediate",
                                    "fromcurrent": True,
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0},
                                    "mode": "immediate",
                                },
                            ],
                        ),
                    ],
                )
            ],
        ),
        frames=[
            go.Frame(
                name=f"frame_{t}",
                data=[
                    go.Scatter3d(
                        x=every_k_positions[t, :, 0],
                        y=every_k_positions[t, :, 1],
                        z=every_k_positions[t, :, 2],
                        mode="markers+text",
                        text=[f"UAV {i}" for i in range(num_uav)],
                        textposition="top center",
                        marker=dict(size=6),
                    ),
                    go.Scatter3d(
                        x=every_k_positions[: t + 1, :, 0].flatten(),
                        y=every_k_positions[: t + 1, :, 1].flatten(),
                        z=every_k_positions[: t + 1, :, 2].flatten(),
                        mode="lines",
                        line=dict(width=2, color="lightgray"),
                    ),
                ],
                layout=go.Layout(
                    annotations=[
                        dict(
                            text=f"Step {t * k}",
                            x=0.5,
                            y=1.05,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=24, color="red"),
                        )
                    ]
                ),
            )
            for t in range(0, num_frames)
        ],
    )

    fig.show()

def compute_scene_bounds(positions, buffer_ratio=0.1):
    # Positions: (T, N, 3)
    mins = positions.min(axis=(0, 1))  # (3,)
    maxs = positions.max(axis=(0, 1))  # (3,)

    center = (mins + maxs) / 2
    half_extent = np.max(maxs - mins) / 2
    half_extent *= 1 + buffer_ratio

    return center, half_extent


# %%
# Plot the simulation environment logs
env_logs = f"{PACKAGE_ROOT}/outputs/env_logs"

positions = np.load(f"{env_logs}/positions_log.npy")
plot_simulation(
    positions, frame_duration=5, k=50
)  # Show only every 50 frames, with 5 ms between frames

# %%
