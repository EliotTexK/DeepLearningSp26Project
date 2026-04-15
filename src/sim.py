# %%
# Imports
from scmrepo.git import Git
from scipy.stats.qmc import PoissonDisk
import os
import json
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
        self.team_1_idxs = np.arange(0, num_uav // 2, dtype=int)
        self.team_2_idxs = np.arange(num_uav // 2, num_uav, dtype=int)

    def reset(self):
        """
        Initialize all UAVs at (0, 0, 0), heading east (alpha = pi/2),
        zero roll and pitch, minimum speed.
        """
        self.state = self.initialize_uavs(
            np.array([[-10000.0, -10000.0, -10000.0], [10000.0, 10000.0, 10000.0]]),
            5000.0,
            V_MIN,
            np.pi / 2.0,
            0.0,
            0.0,
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

    def in_team_1(self, i):
        return i in self.team_1_idxs

    def get_state_json(self):
        """
        Get a JSON-serializable dict describing the full simulation state
        """
        state_dict = {"num_uav": self.num_uav, "dt": self.dt, "g": self.g, "uavs": []}

        vel = self.get_velocity_vectors()

        for i in range(self.num_uav):
            uav_info = {
                "id": i,
                "position": {
                    "x": float(self.state[i, self.indices["x"]]),
                    "y": float(self.state[i, self.indices["y"]]),
                    "z": float(self.state[i, self.indices["z"]]),
                },
                "speed": float(self.state[i, self.indices["v"]]),
                "angles": {
                    "alpha": float(self.state[i, self.indices["alpha"]]),
                    "beta": float(self.state[i, self.indices["beta"]]),
                    "gamma": float(self.state[i, self.indices["gamma"]]),
                },
                "velocity_vector": {
                    "vx": float(vel[i, 0]),
                    "vy": float(vel[i, 1]),
                    "vz": float(vel[i, 2]),
                },
                "observation_vector": self.get_observations(i).tolist(),
            }
            state_dict["uavs"].append(uav_info)
        return state_dict

    def save_current_state(self, filename: str):
        """
        Save the current state to a JSON file.
        """
        with open(filename, "w") as f:
            json.dump(self.get_state_json(), f, indent=4)

    def pprint_state(self):
        data = self.get_state_json()
        string = ""
        for uav in data["uavs"]:
            string += f"UAV {uav['id']}:\n"
            string += f"  Position: ({uav['position']['x']:.2f}, {uav['position']['y']:.2f}, {uav['position']['z']:.2f})\n"
            string += f"  Velocity: {uav['speed']:.2f} m/s\n"
            string += (
                f"  Heading (alpha): {np.degrees(uav['angles']['alpha']):.2f} deg\n"
            )
            string += f"  Roll (beta): {np.degrees(uav['angles']['beta']):.2f} deg\n"
            string += f"  Pitch (gamma): {np.degrees(uav['angles']['gamma']):.2f} deg\n"
            string += f"  State vector: {uav['observation_vector']}\n"
        return string

    def pairwise_observations(self, i, idxs, opponents: bool):
        """
        For current UAV i and indexes idxs of other UAVs, construct observations of those UAVs
        opponents = True: Include opponent-only state variables
        """
        if len(idxs) == 0:
            return {}  # No other UAVs to observe

        state = self.state
        vel = self.get_velocity_vectors()

        observed = {}
        # Current UAV n
        p_i = state[i, 0:3]  # (3,)
        v_i = state[i, 3]  # scalar speed
        vv_i = vel[i]  # (3,)

        # Each other UAV m
        p_m = state[idxs, 0:3]  # (K, 3)
        v_m = state[idxs, 3]  # (K,)
        vv_m = vel[idxs]  # (K, 3)

        # Relative position R_{im} = p_i - p_m
        R_im = p_i[None, :] - p_m  # (K, 3)
        d_im = np.linalg.norm(R_im, axis=-1, keepdims=True)  # (K, 1)

        # Relative speed and velocity
        V_im = (v_i - v_m)[:, None]  # (K, 1)
        vec_V_im = vv_i[None, :] - vv_m  # (K, 3)

        # Norms with epsilon for safety
        eps = 1e-8
        norm_vi = np.linalg.norm(vv_i) + eps  # scalar
        norm_vm = np.linalg.norm(vv_m, axis=-1) + eps  # (K,)
        norm_R = np.linalg.norm(R_im, axis=-1) + eps  # (K,)

        observed.update(
            {
                "d": d_im,  # (K, 1)
                "R": R_im,  # (K, 3)
                "V_rel": V_im,  # (K, 1)
                "vec_V_rel": vec_V_im,  # (K, 3)
            }
        )

        if opponents:
            # Heading crossing angle \phi^{HCA}
            dot_vi_vm = np.sum(vv_i[None, :] * vv_m, axis=-1)  # (K,)
            cos_hca = np.clip(dot_vi_vm / (norm_vi * norm_vm), -1.0, 1.0)
            phi_HCA = np.arccos(cos_hca)[:, None]  # (K, 1)

            # Aspect angle \phi^{AA} = arccos( (v_m · R_im) / (||v_m|| * ||R_im||) )
            dot_vm_R = np.sum(vv_m * R_im, axis=-1)  # (K,)
            cos_aa = np.clip(dot_vm_R / (norm_vm * norm_R), -1.0, 1.0)
            phi_AA = np.arccos(cos_aa)[:, None]  # (K, 1)

            # Angle-off \phi^{AO} = arccos( (v_i * R_im) / (||v_i|| * ||R_im||) )
            dot_vi_R = np.sum(vv_i[None, :] * R_im, axis=-1)  # (K,)
            cos_ao = np.clip(dot_vi_R / (norm_vi * norm_R), -1.0, 1.0)
            phi_AO = np.arccos(cos_ao)[:, None]  # (K, 1)

            observed.update(
                {
                    "phi_HCA": phi_HCA,
                    "phi_AA": phi_AA,
                    "phi_AO": phi_AO,
                }
            )

        return observed

    def get_observations(self, i):
        """
        Build observation vector for UAV i
        """
        if self.in_team_1(i):
            ally_indices = self.team_1_idxs
            ally_indices = ally_indices[ally_indices != i]  # Exclude self
            opponent_indices = self.team_2_idxs
        else:
            ally_indices = self.team_2_idxs
            ally_indices = ally_indices[ally_indices != i]  # Exclude self
            opponent_indices = self.team_1_idxs

        v = self.state[i, self.indices["v"]]
        alpha = self.state[i, self.indices["alpha"]]
        beta = self.state[i, self.indices["beta"]]
        gamma = self.state[i, self.indices["gamma"]]
        own_state = np.array([v, alpha, beta, gamma], dtype=np.float64)  # (4,)

        # Concatenate per-opponent features
        opp_observations = self.pairwise_observations(
            i, np.array(opponent_indices, dtype=int), opponents=True
        )
        if opp_observations:  # No observations if no opponents
            opp_feats = np.concatenate(
                [
                    opp_observations["d"],  # (K,1)
                    opp_observations["R"],  # (K,3)
                    opp_observations["V_rel"],  # (K,1)
                    opp_observations["vec_V_rel"],  # (K,3)
                    opp_observations["phi_HCA"],  # (K,1)
                    opp_observations["phi_AA"],  # (K,1)
                    opp_observations["phi_AO"],  # (K,1)
                ],
                axis=-1,
            ).reshape(
                -1
            )  # (11K,)
        else:
            opp_feats = np.array([], dtype=np.float64)

        # Concatenate per-ally features
        ally_observations = self.pairwise_observations(
            i, np.array(ally_indices, dtype=int), opponents=False
        )
        if ally_observations:  # No observations if no allies
            ally_feats = np.concatenate(
                [
                    ally_observations["d"],  # (L,1)
                    ally_observations["R"],  # (L,3)
                    ally_observations["V_rel"],  # (L,1)
                    ally_observations["vec_V_rel"],  # (L,3)
                ],
                axis=-1,
            ).reshape(
                -1
            )  # (8L,)
        else:
            ally_feats = np.array([], dtype=np.float64)

        # Combine observations
        obs = np.concatenate(
            [own_state, opp_feats, ally_feats], axis=0
        )  # (4 + 11K + 8L,)
        return obs

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
actions = np.array(
    [[0.5, 1.0, 0.1], [0.5, 1.0, 0.1], [0.5, 1.0, 0.1], [0.5, 1.0, 0.1]],
    dtype=np.float64,
)

env_logs = f"{PACKAGE_ROOT}/outputs/env_logs"
if not os.path.exists(env_logs):
    os.makedirs(env_logs)
    print(f"Created directory {env_logs} for environment logs.")
else:
    for file in os.listdir(env_logs):
        os.remove(os.path.join(env_logs, file))
    print(f"Cleared existing logs in {env_logs}.")

for t in range(10):
    state = env.step(actions)
    vel = env.get_velocity_vectors()
    env.save_current_state(f"{env_logs}/step_{t}_state.json")
print(f"Saved environment states to {env_logs}")

# %%
