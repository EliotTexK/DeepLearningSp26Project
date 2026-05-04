# %%
# Imports
import os
from scmrepo.git import Git
import numpy as np
import plotly.graph_objects as go

PACKAGE_ROOT = Git(root_dir=".").root_dir


# %%
# Define plotting functions


def make_tetrahedron(scale=1.0):
    """
    Returns vertices of a tetrahedron pointing along +Z, centered at origin.
    Tip at +Z, base triangle at -Z.
    """
    tip = np.array([0, 0, scale])
    b0 = np.array([0, scale * 0.6, -scale * 0.5])
    b1 = np.array([-scale * 0.52, -scale * 0.3, -scale * 0.5])
    b2 = np.array([scale * 0.52, -scale * 0.3, -scale * 0.5])
    # Faces: (tip,b0,b1), (tip,b1,b2), (tip,b2,b0), (b0,b2,b1)  ← base wound CCW from outside
    verts = np.array([tip, b0, b1, b2])  # (4, 3)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])  # (4, 3)
    return verts, faces


def rotation_matrix_from_z_to_v(v):
    """
    Returns a (3,3) rotation matrix that rotates the +Z axis to align with v.
    v: (3,) unit vector
    """
    z = np.array([0.0, 0.0, 1.0])
    v = v / (np.linalg.norm(v) + 1e-12)
    cross = np.cross(z, v)
    dot = np.dot(z, v)
    c_norm = np.linalg.norm(cross)

    if c_norm < 1e-9:
        # Already aligned or anti-aligned
        return np.eye(3) if dot > 0 else np.diag([1, -1, -1])

    kx, ky, kz = cross / c_norm
    sin_a = c_norm
    cos_a = dot
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)


def build_mesh_for_frame(positions, velocities, scale=1.0):
    """
    Builds concatenated Mesh3d arrays for all UAVs at one frame.

    positions:  (N, 3)
    velocities: (N, 3)  — already unit vectors
    Returns x, y, z, i, j, k arrays suitable for go.Mesh3d.
    """
    base_verts, base_faces = make_tetrahedron(scale)
    N = positions.shape[0]
    nv = len(base_verts)  # 4
    nf = len(base_faces)  # 4

    all_x = np.empty(N * nv)
    all_y = np.empty(N * nv)
    all_z = np.empty(N * nv)
    all_i = np.empty(N * nf, dtype=int)
    all_j = np.empty(N * nf, dtype=int)
    all_k = np.empty(N * nf, dtype=int)

    for n in range(N):
        R = rotation_matrix_from_z_to_v(velocities[n])
        verts = (R @ base_verts.T).T + positions[n]  # (4, 3)

        all_x[n * nv : (n + 1) * nv] = verts[:, 0]
        all_y[n * nv : (n + 1) * nv] = verts[:, 1]
        all_z[n * nv : (n + 1) * nv] = verts[:, 2]

        offset = n * nv
        all_i[n * nf : (n + 1) * nf] = base_faces[:, 0] + offset
        all_j[n * nf : (n + 1) * nf] = base_faces[:, 1] + offset
        all_k[n * nf : (n + 1) * nf] = base_faces[:, 2] + offset

    return all_x, all_y, all_z, all_i, all_j, all_k


def make_mesh3d(positions, velocities, scale=1.0, color="blue"):
    x, y, z, i, j, k = build_mesh_for_frame(positions, velocities, scale)
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=1.0,
        flatshading=True,
        lighting=dict(diffuse=0.9, specular=0.3, ambient=0.3),
    )


def compute_scene_bounds(positions, buffer_ratio=0.1):
    mins = positions.min(axis=(0, 1))
    maxs = positions.max(axis=(0, 1))
    center = (mins + maxs) / 2
    half_extent = np.max(maxs - mins) / 2 * (1 + buffer_ratio)
    return center, half_extent


def plot_simulation(positions, velocities, frame_duration, k):
    every_k_positions = positions[::k]
    norms = np.linalg.norm(velocities[::k], axis=2, keepdims=True)
    every_k_velocities = velocities[::k] / (norms + 1e-12)

    num_frames = every_k_positions.shape[0]
    center, half_extent = compute_scene_bounds(positions, buffer_ratio=0.1)
    scale = half_extent * 0.05

    N = every_k_positions.shape[1]
    half = N // 2

    data = []
    for n in range(N):
        color = "red" if n < half else "blue"
        data.append(
            make_mesh3d(
                every_k_positions[0][n : n + 1],
                every_k_velocities[0][n : n + 1],
                scale,
                color=color,
            )
        )

    fig = go.Figure(
        data=data,
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
                data=data,
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
            for t in range(num_frames)
        ],
    )

    fig.show()


# %%
# Plot the simulation environment logs
env_logs = f"{PACKAGE_ROOT}/outputs/env_logs"

positions = np.load(f"{env_logs}/positions_log.npy")
velocities = np.load(f"{env_logs}/velocities_log.npy")
plot_simulation(positions, velocities, frame_duration=10, k=1)

# %%
