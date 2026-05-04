import copy
import random
import os
from collections import deque
from scmrepo.git import Git
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")          # headless-safe; swap to "TkAgg" for live window
import matplotlib.pyplot as plt

# Real UAV combat simulator — place uav_sim.py on your PYTHONPATH or in the
# same directory as this file.
from sim import (
    MultiUAVSimulator,
    DT, G,
    T_MIN, T_MAX,
    L_MIN, L_MAX,
    BETA_MIN, BETA_MAX,
    V_MIN,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PACKAGE_ROOT = Git(root_dir=".").root_dir

# make sure output folder exists
os.makedirs(f"{PACKAGE_ROOT}/outputs", exist_ok=True)

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_HEADER_WIDTH = 72

def _print_header(title: str) -> None:
    print("\n" + "=" * _HEADER_WIDTH)
    print(f"  {title}")
    print("=" * _HEADER_WIDTH)


def _print_episode_row(ep: int, ep_return: float, avg_return: float,
                       steps: int, alive: int, n_agents: int) -> None:
    """Single-line episode summary."""
    print(
        f"  Ep {ep:>7,d} │ "
        f"Return: {ep_return:>10.3f} │ "
        f"Avg(last): {avg_return:>10.3f} │ "
        f"Steps: {steps:>4d} │ "
        f"Alive: {alive}/{n_agents}"
    )


def _print_separator() -> None:
    print("  " + "─" * (_HEADER_WIDTH - 2))


def _print_attack_event(step: int, rewards: np.ndarray) -> None:
    r_str = "  ".join(f"A{i}={r:+.2f}" for i, r in enumerate(rewards))
    print(f"     Attack  step={step:>4d}  │  {r_str}")


# ---------------------------------------------------------------------------
# Reward plot
# ---------------------------------------------------------------------------

def save_reward_plot(episode_returns: list[float],
                     log_interval: int,
                     path: str = "reward_history.png") -> None:
    """
    Save a plot of per-episode return and a smoothed rolling average.
    """
    returns = np.array(episode_returns)
    eps     = np.arange(1, len(returns) + 1)

    # Rolling mean with window = log_interval (or all data if shorter)
    window  = min(log_interval, len(returns))
    kernel  = np.ones(window) / window
    smooth  = np.convolve(returns, kernel, mode="valid")
    smooth_eps = eps[window - 1:]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eps, returns, color="#9ecae1", linewidth=0.7,
            alpha=0.6, label="Episode return")
    ax.plot(smooth_eps, smooth, color="#2171b5", linewidth=2.0,
            label=f"Rolling mean (w={window})")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Return", fontsize=12)
    ax.set_title("MATD3 – UAV Combat: Return over Training", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  📊  Reward plot saved → {path}")


# ---------------------------------------------------------------------------
# UAV environment adapter
# ---------------------------------------------------------------------------

class _FixedSizeMultiUAVSim(MultiUAVSimulator):
    """
    Thin subclass that disables agent removal.

    ``MultiUAVSimulator.remove_destroyed()`` shrinks its internal state array
    whenever a UAV is shot down, which would produce variable-length
    observation tensors incompatible with MATD3's fixed batch dimensions.
    Overriding it to a no-op keeps the state array at its original
    (num_uav, 7) shape for the full episode duration.  Destroyed agents
    are flagged in ``self.destroyed`` and are masked by the wrapper.
    """

    def remove_destroyed(self) -> None:  # type: ignore[override]
        pass  # intentional no-op — masking handled by MultiUAVEnvWrapper


class MultiUAVEnvWrapper:
    """
    Adapter that presents ``MultiUAVSimulator`` to ``MATD3.train()``.

    MATD3 requires a simple two-method interface::

        states               = env.reset()          # (N, state_dim)
        next_states, r, d, _ = env.step(actions)    # (N, state_dim), (N,), (N,), dict

    Observation dimension is fixed for the full episode.  Because
    ``_FixedSizeMultiUAVSim`` never shrinks its team-index arrays,
    each UAV always observes exactly ``num_uav//2`` opponents and
    ``num_uav//2 - 1`` allies, giving::

        state_dim = 4 + 11 * (num_uav // 2) + 8 * (num_uav // 2 - 1)

    Destroyed agents receive:
    * zero action vector  (so their physics drift harmlessly)
    * zero observation vector
    * ``done = True``     (so the critic target is masked to 0)
    * zero reward
    """

    def __init__(self, num_uav: int = 4, dt: float = DT, g: float = G,
                 spawn_range=3000.0, spawn_radius=800.0):
        if num_uav % 2 != 0:
            raise ValueError("num_uav must be even (equal-sized teams).")
        self.num_uav = num_uav
        self._dt = dt
        self._g = g

        n_opp  = num_uav // 2        # opponents observed per agent
        n_ally = num_uav // 2 - 1   # allies observed per agent (self excluded)
        self.state_dim = 4 + 11 * n_opp + 8 * n_ally

        self._sim: _FixedSizeMultiUAVSim | None = None
        self._spawn_range  = spawn_range
        self._spawn_radius = spawn_radius

    # ------------------------------------------------------------------
    # MATD3 interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Initialise a new episode and return the first joint observation."""
        self._sim = _FixedSizeMultiUAVSim(num_uav=self.num_uav,
                                          dt=self._dt, g=self._g)
        self._sim.state = self._sim.initialize_uavs(
            bounds=np.array([
                [-self._spawn_range] * 3,
                [ self._spawn_range] * 3,
            ]),
            radius=self._spawn_radius,
            v=V_MIN,
            alpha=np.pi / 2.0,
            beta=0.0,
            gamma=0.0,
        )
        self._sim.destroyed = np.zeros(self.num_uav, dtype=bool)
        return self._build_obs()

    def step(self, actions: np.ndarray):
        """
        Advance the simulation one step.

        Args:
            actions: ``(N, 3)`` float array — ``[T, L, beta]`` per agent.
                     Destroyed agents' rows are zeroed before forwarding.

        Returns:
            next_states : ``(N, state_dim)`` float32 ndarray
            rewards     : ``(N,)``           float32 ndarray
            dones       : ``(N,)``           bool ndarray
            info        : dict forwarded from the simulator
        """
        actions = np.asarray(actions, dtype=np.float64).reshape(self.num_uav, 3)
        actions[self._sim.destroyed] = 0.0

        _, rewards, episode_done, info = self._sim.step(actions)

        next_states = self._build_obs()

        per_agent_done = self._sim.destroyed.copy()
        if episode_done:
            per_agent_done[:] = True

        return next_states, rewards.astype(np.float32), per_agent_done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        out = np.zeros((self.num_uav, self.state_dim), dtype=np.float32)
        for i in range(self.num_uav):
            if not self._sim.destroyed[i]:
                raw = self._sim.get_observations(i)
                n = min(len(raw), self.state_dim)
                out[i, :n] = raw[:n].astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:

    def __init__(self, capacity: int = 1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, joint_state, joint_action, rewards, next_joint_state, dones):
        self.buffer.append((
            np.array(joint_state,      dtype=np.float32),
            np.array(joint_action,     dtype=np.float32),
            np.array(rewards,          dtype=np.float32),
            np.array(next_joint_state, dtype=np.float32),
            np.array(dones,            dtype=np.float32),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        joint_s, joint_a, r, joint_ns, d = zip(*batch)
        return (
            torch.FloatTensor(np.stack(joint_s)).to(DEVICE),
            torch.FloatTensor(np.stack(joint_a)).to(DEVICE),
            torch.FloatTensor(np.stack(r)).to(DEVICE),
            torch.FloatTensor(np.stack(joint_ns)).to(DEVICE),
            torch.FloatTensor(np.stack(d)).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Actor / Critic networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Decentralized policy: a_i = mu_i(s_i)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    """Centralized Q-function over joint (s, a)."""

    def __init__(self, joint_state_dim: int, joint_action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        in_dim = joint_state_dim + joint_action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_state: torch.Tensor,
                joint_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([joint_state, joint_action], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Per-agent MATD3 logic
# ---------------------------------------------------------------------------

class MATD3Agent:

    def __init__(self,
                 agent_id: int,
                 state_dim: int,
                 action_dim: int,
                 joint_state_dim: int,
                 joint_action_dim: int,
                 action_low: np.ndarray,
                 action_high: np.ndarray,
                 hidden_dim: int     = 256,
                 lr: float           = 3e-4,
                 gamma: float        = 0.95,
                 tau: float          = 0.005,
                 policy_noise: float = 0.2,
                 noise_clip: float   = 0.5,
                 policy_delay: int   = 2):

        self.id           = agent_id
        self.action_dim   = action_dim
        self.action_low   = torch.FloatTensor(action_low).to(DEVICE)
        self.action_high  = torch.FloatTensor(action_high).to(DEVICE)
        self.gamma        = gamma
        self.tau          = tau
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.policy_delay = policy_delay
        self._critic_updates = 0

        self.actor        = Actor(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt    = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic1        = Critic(joint_state_dim, joint_action_dim, hidden_dim).to(DEVICE)
        self.critic2        = Critic(joint_state_dim, joint_action_dim, hidden_dim).to(DEVICE)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic1_opt    = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt    = optim.Adam(self.critic2.parameters(), lr=lr)

    @torch.no_grad()
    def select_action(self, state: np.ndarray,
                      explore_noise: float = 0.1) -> np.ndarray:
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        a = self.actor(s).squeeze(0)
        a = self._scale(a)
        if explore_noise > 0:
            noise = torch.randn_like(a) * explore_noise
            a = torch.clamp(a + noise, self.action_low, self.action_high)
        return a.cpu().numpy()

    def _scale(self, a: torch.Tensor) -> torch.Tensor:
        lo, hi = self.action_low, self.action_high
        return lo + (a + 1.0) * 0.5 * (hi - lo)

    def update(self,
               joint_states: torch.Tensor,
               joint_actions: torch.Tensor,
               rewards: torch.Tensor,
               next_joint_states: torch.Tensor,
               dones: torch.Tensor,
               next_actions_all: torch.Tensor,
               own_states: torch.Tensor) -> dict:

        self._critic_updates += 1
        losses = {}

        with torch.no_grad():
            noise = (torch.randn_like(next_actions_all) * self.policy_noise
                    ).clamp(-self.noise_clip, self.noise_clip)
            next_a_smooth = next_actions_all + noise

            q1_next = self.critic1_target(next_joint_states, next_a_smooth)
            q2_next = self.critic2_target(next_joint_states, next_a_smooth)
            q_next  = torch.min(q1_next, q2_next)
            y       = rewards + self.gamma * (1.0 - dones) * q_next

        q1 = self.critic1(joint_states, joint_actions)
        q2 = self.critic2(joint_states, joint_actions)

        critic1_loss = F.mse_loss(q1, y)
        critic2_loss = F.mse_loss(q2, y)

        self.critic1_opt.zero_grad(); critic1_loss.backward(); self.critic1_opt.step()
        self.critic2_opt.zero_grad(); critic2_loss.backward(); self.critic2_opt.step()

        losses["critic1"] = critic1_loss.item()
        losses["critic2"] = critic2_loss.item()

        if self._critic_updates % self.policy_delay == 0:
            a_i = self._scale(self.actor(own_states))
            joint_actions_for_grad = joint_actions.clone()
            start = self.id * self.action_dim
            end   = start + self.action_dim
            joint_actions_for_grad[:, start:end] = a_i

            actor_loss = -self.critic1(joint_states, joint_actions_for_grad).mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            losses["actor"] = actor_loss.item()

            _soft_update(self.actor,   self.actor_target,   self.tau)
            _soft_update(self.critic1, self.critic1_target, self.tau)
            _soft_update(self.critic2, self.critic2_target, self.tau)

        return losses


# ---------------------------------------------------------------------------
# MATD3 coordinator
# ---------------------------------------------------------------------------

class MATD3:
    """Multi-Agent Twin Delayed DDPG (CTDE)."""

    def __init__(self,
                 n_agents: int,
                 state_dim: int,
                 action_dim: int,
                 action_low: np.ndarray,
                 action_high: np.ndarray,
                 hidden_dim: int     = 256,
                 lr: float           = 3e-4,
                 gamma: float        = 0.95,
                 tau: float          = 0.005,
                 policy_noise: float = 0.2,
                 noise_clip: float   = 0.5,
                 policy_delay: int   = 2,
                 buffer_size: int    = 1_000_000,
                 batch_size: int     = 256):

        self.n_agents   = n_agents
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size

        joint_state_dim  = n_agents * state_dim
        joint_action_dim = n_agents * action_dim

        self.agents = [
            MATD3Agent(
                agent_id         = i,
                state_dim        = state_dim,
                action_dim       = action_dim,
                joint_state_dim  = joint_state_dim,
                joint_action_dim = joint_action_dim,
                action_low       = action_low,
                action_high      = action_high,
                hidden_dim       = hidden_dim,
                lr               = lr,
                gamma            = gamma,
                tau              = tau,
                policy_noise     = policy_noise,
                noise_clip       = noise_clip,
                policy_delay     = policy_delay,
            )
            for i in range(n_agents)
        ]

        self.buffer = ReplayBuffer(capacity=buffer_size)

    def select_actions(self, states: np.ndarray,
                       explore_noise: float = 0.1) -> np.ndarray:
        return np.array([
            agent.select_action(states[i], explore_noise)
            for i, agent in enumerate(self.agents)
        ])

    def store_transition(self, states, actions, rewards, next_states, dones):
        self.buffer.push(states, actions, rewards, next_states, dones)

    def train_step(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return {}

        joint_s, joint_a, rewards, joint_ns, dones = self.buffer.sample(
            self.batch_size)

        B = joint_s.shape[0]
        flat_js  = joint_s.view(B, -1)
        flat_ja  = joint_a.view(B, -1)
        flat_njs = joint_ns.view(B, -1)

        with torch.no_grad():
            next_actions = []
            for i, agent in enumerate(self.agents):
                ns_i   = joint_ns[:, i, :]
                a_next = agent.actor_target(ns_i)
                a_next = agent._scale(a_next)
                next_actions.append(a_next)
            next_joint_a = torch.cat(next_actions, dim=-1)

        all_losses = {}
        for i, agent in enumerate(self.agents):
            r_i    = rewards[:, i].unsqueeze(1)
            done_i = dones[:, i].unsqueeze(1)
            own_si = joint_s[:, i, :]

            losses = agent.update(
                joint_states      = flat_js,
                joint_actions     = flat_ja,
                rewards           = r_i,
                next_joint_states = flat_njs,
                dones             = done_i,
                next_actions_all  = next_joint_a,
                own_states        = own_si,
            )
            all_losses[i] = losses

        return all_losses

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self,
              env,
              n_episodes: int     = 100_000,
              max_steps: int      = 300,
              explore_noise: float = 0.2,
              log_interval: int   = 1000,
              plot_path: str      = "reward_history.png") -> list[float]:
        """
        Main training loop with neat per-episode logging and reward plot.

        Args:
            env           : environment with reset() / step() interface
            n_episodes    : total training episodes
            max_steps     : max steps per episode
            explore_noise : Gaussian exploration noise std
            log_interval  : print rolling-average summary every N episodes
            plot_path     : where to save the final reward graph

        Returns:
            episode_returns : list of total returns (one float per episode)
        """
        episode_returns: list[float] = []

        _print_header(
            f"MATD3 Training  │  {self.n_agents} agents  │  "
            f"state_dim={self.state_dim}  │  action_dim={self.action_dim}  │  "
            f"device={DEVICE}"
        )
        print(
            f"  Episodes: {n_episodes:,}  │  max_steps/ep: {max_steps}  │  "
            f"batch: {self.batch_size}  │  explore_noise: {explore_noise}\n"
        )
        _print_separator()

        for ep in range(1, n_episodes + 1):
            states    = env.reset()
            ep_return = 0.0
            step      = 0

            print(f"  ┌─ Ep {ep:>7,d} " + "─" * 50)

            for step in range(1, max_steps + 1):
                actions = self.select_actions(states, explore_noise)
                next_states, rewards, dones, info = env.step(actions)
                '''
                # ── Print notable reward events ──────────────────────────
                if np.any(rewards != 0):
                    _print_attack_event(step, rewards)
                '''
                self.store_transition(states, actions, rewards,
                                      next_states, dones)
                self.train_step()

                ep_return += float(np.sum(rewards))
                states     = next_states

                if all(dones):
                    break

            episode_returns.append(ep_return)

            # ── Per-episode summary line ─────────────────────────────────
            alive = int(np.sum(~env._sim.destroyed))
            avg   = float(np.mean(episode_returns[-log_interval:]))
            print(
                f"  └─ Return: {ep_return:>10.3f} │ "
                f"Avg(last {log_interval}): {avg:>10.3f} │ "
                f"Steps: {step:>4d} │ "
                f"Alive: {alive}/{self.n_agents}"
            )

            # ── Rolling-average banner every log_interval episodes ───────
            if ep % log_interval == 0:
                _print_separator()
                print(
                    f"    Checkpoint ep {ep:,}  │  "
                    f"Avg return (last {log_interval}): {avg:+.4f}  │  "
                    f"Buffer: {len(self.buffer):,} transitions"
                )
                _print_separator()
                save_reward_plot(episode_returns, log_interval, path=plot_path)
                
                # ── NEW: save weights ────────────────────────────────────────
                checkpoint = {
                    i: {
                        "actor":          agent.actor.state_dict(),
                        "actor_target":   agent.actor_target.state_dict(),
                        "critic1":        agent.critic1.state_dict(),
                        "critic2":        agent.critic2.state_dict(),
                        "critic1_target": agent.critic1_target.state_dict(),
                        "critic2_target": agent.critic2_target.state_dict(),
                    }
                    for i, agent in enumerate(self.agents)
                }
                torch.save(checkpoint, f"checkpoint_ep{ep}.pt")
                print(f"  💾  Checkpoint saved → checkpoint_ep{ep}.pt")

        # ── Final plot ───────────────────────────────────────────────────
        _print_separator()
        print(f"\n  Training complete.  Total episodes: {n_episodes:,}")
        save_reward_plot(episode_returns, log_interval, path=plot_path)

        return episode_returns
    
    def test(self, max_steps: int = 300):
        env_logs = f"{PACKAGE_ROOT}/outputs/env_logs"

        _print_header(
            f"MATD3 Testing  │  {self.n_agents} agents  │  "
            f"state_dim={self.state_dim}  │  action_dim={self.action_dim}  │  "
            f"device={DEVICE}"
        )
        _print_separator()

        positions, velocities, attacked, destroyed = [], [], [], []
        states    = env.reset()
        ep_return = 0

        for step in range(1, max_steps + 1):

            actions = self.select_actions(states, explore_noise=0.0)
            next_states, rewards, dones, info = env.step(actions)

            positions.append(info["positions"])
            velocities.append(info["velocity_vectors"])
            attacked.append(info["attacked"])
            destroyed.append(info["destroyed"])

            '''
            # ── Print notable reward events ──────────────────────────
            if np.any(rewards != 0):
                _print_attack_event(step, rewards)
            '''
            self.store_transition(states, actions, rewards,
                                    next_states, dones)
            self.train_step()

            ep_return += float(np.sum(rewards))
            states     = next_states

            if all(dones):
                break
        
        np.save(f"{env_logs}/positions_log.npy", np.array(positions))
        np.save(f"{env_logs}/velocities_log.npy", np.array(velocities))
        np.save(f"{env_logs}/attacked_log.npy", np.array(attacked))
        np.save(f"{env_logs}/destroyed_log.npy", np.array(destroyed))
        print(f"Saved environment states to {env_logs}")

    def load(self, path: str) -> None:
        """
        Load agent weights from a checkpoint saved during training.

        Args:
            path : path to the checkpoint file (e.g. "checkpoint_ep1000.pt")
        """
        checkpoint = torch.load(path, map_location=DEVICE)

        for i, agent in enumerate(self.agents):
            weights = checkpoint[i]
            agent.actor.load_state_dict(weights["actor"])
            agent.actor_target.load_state_dict(weights["actor_target"])
            agent.critic1.load_state_dict(weights["critic1"])
            agent.critic2.load_state_dict(weights["critic2"])
            agent.critic1_target.load_state_dict(weights["critic1_target"])
            agent.critic2_target.load_state_dict(weights["critic2_target"])

        print(f"  ✅  Checkpoint loaded ← {path}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _soft_update(net: nn.Module, target: nn.Module, tau: float) -> None:
    """Polyak averaging: θ_target ← τ·θ + (1-τ)·θ_target."""
    for p, tp in zip(net.parameters(), target.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    NUM_UAV    = 4
    ACTION_DIM = 3   # [T, L, beta]

    action_low  = np.array([T_MIN, L_MIN, BETA_MIN])
    action_high = np.array([T_MAX, L_MAX, BETA_MAX])

    env = MultiUAVEnvWrapper(
        num_uav      = NUM_UAV,
        dt           = DT,
        g            = G,
        spawn_range  = 3000.0,
        spawn_radius = 800.0,
    )
    STATE_DIM = env.state_dim

    matd3 = MATD3(
        n_agents     = NUM_UAV,
        state_dim    = STATE_DIM,
        action_dim   = ACTION_DIM,
        action_low   = action_low,
        action_high  = action_high,
        hidden_dim   = 256,
        lr           = 3e-4,
        gamma        = 0.95,
        tau          = 0.005,
        policy_noise = 0.2,
        noise_clip   = 0.5,
        policy_delay = 2,
        buffer_size  = 1_000_000,
        batch_size   = 256,
    )

    checkpoint_path = input("input file name to load checkpoint, or leave blank for a fresh run: ")
    if checkpoint_path != '': matd3.load(checkpoint_path)
    testing_mode = input('testing mode (y/n)? ')

    if testing_mode:
        matd3.test(max_steps = 1000)
    else:
        returns = matd3.train(
            env,
            n_episodes    = 10000,
            max_steps     = 300,
            log_interval  = 50,
            plot_path     = "reward_history.png",
        )

        print(f"\n  Final episode return: {returns[-1]:.3f}")