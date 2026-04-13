# %%
# Imports
from scmrepo.git import Git
from scipy.stats.qmc import PoissonDisk
import os
import random
import numpy as np

PACKAGE_ROOT = Git(root_dir=".").root_dir

# make sure output folder exists
os.makedirs(f"{PACKAGE_ROOT}/outputs", exist_ok=True)

# %%
# Define simulator
DT = 0.05  # Time between steps
G = 9.81  # gravitational acceleration

# Flight constraints
T_MIN, T_MAX = -2.0, 2.0
L_MIN, L_MAX = -5.0, 5.0
BETA_MIN, BETA_MAX = -np.pi / 3.0, np.pi / 3.0
GAMMA_MIN, GAMMA_MAX = -np.pi / 4.0, np.pi / 4.0
V_MIN, V_MAX = 100.0, 200.0


class MultiUAVSimulator:
    def __init__(self, num_uav: int, dt: float = DT, g: float = G):
        self.num_uav = num_uav
        self.dt = dt
        self.g = g
        self.state = None
        # State schema: [x, y, z, v, alpha, beta, gamma] for each UAV
        self.indices = {
            "x": 0,
            "y": 1,
            "z": 2,
            "v": 3,
            "alpha": 4,
            "beta": 5,
            "gamma": 6,
        }

    def reset(self):
        """
        Initialize all UAVs at (0, 0, 0), heading east (alpha = pi/2),
        zero roll and pitch, minimum speed.
        """
        self.state = self.initialize_uavs(
            np.array([[-10000.0,-10000.0,-10000.0],[10000.0,10000.0,10000.0]]),
            5000.0,
            V_MIN,
            np.pi / 2.0,
            0.0,
            0.0
        )
        return self.state.copy()

    def clip_actions(self, actions):
        """
        Clip actions to their allowed ranges.
        """
        T, L, beta = np.split(actions, 3, axis=-1)
        T = np.clip(T, T_MIN, T_MAX)
        L = np.clip(L, L_MIN, L_MAX)
        beta = np.clip(beta, BETA_MIN, BETA_MAX)
        return np.concatenate([T, L, beta], axis=-1)

    def dynamics(self, state, actions):
        """
        Compute time derivatives of all state variables given current state and actions.
        """
        x, y, z, v, alpha, beta, gamma = np.split(state, 7, axis=-1)
        T, L, beta = np.split(actions, 3, axis=-1)

        # Avoid division by zero in dynamics (v in denominator)
        v_safe = np.clip(v, 1e-6, None)

        # Dynamic model
        dv = self.g * (T - np.sin(gamma))
        dgamma = (self.g / v_safe) * (L * np.cos(beta) - np.cos(gamma))
        dalpha = (self.g * L * np.sin(beta)) / (v_safe * np.cos(gamma))

        # Motion model
        dx = v * np.cos(gamma) * np.sin(alpha)
        dy = v * np.cos(gamma) * np.cos(alpha)
        dz = v * np.sin(gamma)

        dbeta = np.zeros_like(
            beta
        )  # beta is directly controlled by the agent and has no dynamics

        dst = np.concatenate([dx, dy, dz, dv, dalpha, dbeta, dgamma], axis=-1)
        return dst

    def rk4(self, state, actions):
        """
        RK4 integration - faster than scipy's more complex solver
        """
        dt = self.dt

        k1 = self.dynamics(state, actions)
        k2 = self.dynamics(state + 0.5 * dt * k1, actions)
        k3 = self.dynamics(state + 0.5 * dt * k2, actions)
        k4 = self.dynamics(state + dt * k3, actions)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self, actions):
        """
        Advance the simulator by one time step using RK4 integration.
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.num_uav, 3):
            raise ValueError(
                f"Expected actions shape {(self.num_uav, 3)}, got {actions.shape}"
            )

        # Clip actions to constraints
        actions = self.clip_actions(actions)

        s_new = self.rk4(self.state, actions)

        # Unpack for clamping
        x, y, z, v, alpha, beta, gamma = np.split(s_new, 7, axis=-1)

        # Clamp state variables
        v = np.clip(v, V_MIN, V_MAX)
        beta = np.clip(beta, BETA_MIN, BETA_MAX)
        gamma = np.clip(gamma, GAMMA_MIN, GAMMA_MAX)

        # Wrap alpha to [-pi, pi] for sanity
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        # Repack state
        self.state = np.concatenate([x, y, z, v, alpha, beta, gamma], axis=-1)
        return self.state.copy()

    def get_velocity_vectors(self):
        """
        Compute 3D velocity vectors for all UAVs from (v, alpha, gamma).
        """
        x, y, z, v, alpha, beta, gamma = np.split(self.state, 7, axis=-1)
        vx = v * np.cos(gamma) * np.sin(alpha)
        vy = v * np.cos(gamma) * np.cos(alpha)
        vz = v * np.sin(gamma)
        return np.concatenate([vx, vy, vz], axis=-1)

    def pprint_state(self):
        string = ""
        for i in range(self.num_uav):
            string += f"UAV {i}:\n"
            string += f"  Position: ({self.state[i, self.indices['x']]:.2f}, {self.state[i, self.indices['y']]:.2f}, {self.state[i, self.indices['z']]:.2f})\n"
            string += f"  Velocity: {self.state[i, self.indices['v']]:.2f} m/s\n"
            string += f"  Heading (alpha): {np.degrees(self.state[i, self.indices['alpha']]):.2f} deg\n"
            string += f"  Roll (beta): {np.degrees(self.state[i, self.indices['beta']]):.2f} deg\n"
            string += f"  Pitch (gamma): {np.degrees(self.state[i, self.indices['gamma']]):.2f} deg\n"
        return string
    

    def initialize_uavs(
        self,
        bounds: np.ndarray,
        radius: float,
        v: float = 0.0,
        alpha: float = 0.0,
        beta: float = 0.0,
        gamma: float = 0.0,
    ) -> np.ndarray:
        """
        Distribute UAVs throughout a 3D cuboid using Poisson disc sampling.

        Args:
            bounds: (2, 3) array specifying cuboid bounds:
                    [[x_min, y_min, z_min],
                    [x_max, y_max, z_max]]
            radius: Minimum distance between any two UAVs (same units as bounds).
            v:      Initial speed assigned to all UAVs (default 0.0).
            alpha:  Initial roll assigned to all UAVs (default 0.0).
            beta:   Initial pitch assigned to all UAVs (default 0.0).
            gamma:  Initial yaw assigned to all UAVs (default 0.0).

        Returns:
            initial_states: (N, 7) array of UAV states [x, y, z, v, alpha, beta, gamma].
        """
        lower = bounds[0]
        upper = bounds[1]

        sampler = PoissonDisk(d=3, radius=radius, l_bounds=lower, u_bounds=upper)
        positions = sampler.random(self.num_uav)

        N = len(positions)
        kinematic_cols = np.full((N, 4), [v, alpha, beta, gamma])

        return np.hstack([positions, kinematic_cols])

# %%
# Run simulator
env = MultiUAVSimulator(num_uav=4, dt=DT, g=G)
state = env.reset()
print("Initial state:", state)

# Simple example: constant moderate thrust, small positive lift, small roll
actions = np.array([
        [0.5, 1.0, 0.1],
        [0.5, 1.0, 0.1],
        [0.5, 1.0, 0.1],
        [0.5, 1.0, 0.1]
    ],
    dtype=np.float64
)

for t in range(10):
    state = env.step(actions)
    vel = env.get_velocity_vectors()
    print(f"Step {t+1}: {env.pprint_state()}")

# %%
