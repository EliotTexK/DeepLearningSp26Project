import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

    def __init__(self, num_uav: int = 4, dt: float = DT, g: float = G, spawn_range=3000.0, spawn_radius=800.0):
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
        self._sim = _FixedSizeMultiUAVSim(num_uav=self.num_uav, dt=self._dt, g=self._g)
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
            dones       : ``(N,)``           bool ndarray —
                          True for individually destroyed agents, or all-True
                          when the episode terminates (one side wiped out).
            info        : dict forwarded from the simulator
        """
        actions = np.asarray(actions, dtype=np.float64).reshape(self.num_uav, 3)
        actions[self._sim.destroyed] = 0.0   # mask out dead-agent inputs

        _, rewards, episode_done, info = self._sim.step(actions)

        next_states = self._build_obs()

        # Per-agent done flags: True if this agent was destroyed at any point,
        # or if the entire episode has ended (one team wiped out).
        per_agent_done = self._sim.destroyed.copy()
        if episode_done:
            per_agent_done[:] = True

        return next_states, rewards.astype(np.float32), per_agent_done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """
        Build a ``(num_uav, state_dim)`` observation array.
        Rows for destroyed agents are left as zeros.
        """
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
        """
        Store one transition.
          joint_state / next_joint_state : (N, state_dim) arrays
          joint_action                   : (N, action_dim) arrays
          rewards / dones                : (N,) arrays
        """
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
            torch.FloatTensor(np.stack(joint_s)).to(DEVICE),   # (B, N, state_dim)
            torch.FloatTensor(np.stack(joint_a)).to(DEVICE),   # (B, N, action_dim)
            torch.FloatTensor(np.stack(r)).to(DEVICE),         # (B, N)
            torch.FloatTensor(np.stack(joint_ns)).to(DEVICE),  # (B, N, state_dim)
            torch.FloatTensor(np.stack(d)).to(DEVICE),         # (B, N)
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Actor / Critic networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """
    Each agent's local policy: a_i = mu_i(s_i).

    Takes only the agent's own observation as input (decentralized execution).
    Three hidden layers of size 256 as per Table 2.
    """

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
    """
    Centralized Q-function: Q_i(s, a) where s and a are the *joint*
    state and action of all agents.

    The joint input is the concatenation of all agents' states and actions,
    which is the standard CTDE formulation used in MATD3/MADDPG.
    Three hidden layers of size 256 as per Table 2.
    """

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
        """
        Args:
            joint_state  : (B, joint_state_dim)
            joint_action : (B, joint_action_dim)
        Returns:
            Q-value      : (B, 1)
        """
        x = torch.cat([joint_state, joint_action], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Per-agent MATD3 logic
# ---------------------------------------------------------------------------

class MATD3Agent:
    """
    One agent in MATD3.

    Owns:
      - one actor  + its target
      - two critics + their targets  (clipped double-Q)

    The critics are centralized: they receive the full joint (s, a).
    The actor is decentralized: it only sees its own local state s_i.
    """

    def __init__(self,
                 agent_id: int,
                 state_dim: int,
                 action_dim: int,
                 joint_state_dim: int,
                 joint_action_dim: int,
                 action_low: np.ndarray,
                 action_high: np.ndarray,
                 hidden_dim: int    = 256,
                 lr: float          = 3e-4,
                 gamma: float       = 0.95,
                 tau: float         = 0.005,
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

        # ---- Actor ----
        self.actor        = Actor(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt    = optim.Adam(self.actor.parameters(), lr=lr)

        # ---- Critic 1 & 2 ----
        self.critic1        = Critic(joint_state_dim, joint_action_dim, hidden_dim).to(DEVICE)
        self.critic2        = Critic(joint_state_dim, joint_action_dim, hidden_dim).to(DEVICE)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic1_opt    = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt    = optim.Adam(self.critic2.parameters(), lr=lr)

    @torch.no_grad()
    def select_action(self, state: np.ndarray,
                      explore_noise: float = 0.1) -> np.ndarray:
        """Decentralized execution: only own state s_i is used."""
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        a = self.actor(s).squeeze(0)
        a = self._scale(a)
        if explore_noise > 0:
            noise = torch.randn_like(a) * explore_noise
            a = torch.clamp(a + noise, self.action_low, self.action_high)
        return a.cpu().numpy()

    def _scale(self, a: torch.Tensor) -> torch.Tensor:
        """Map tanh output [-1,1] to [action_low, action_high]."""
        lo, hi = self.action_low, self.action_high
        return lo + (a + 1.0) * 0.5 * (hi - lo)

    def update(self,
               joint_states: torch.Tensor,        # (B, joint_state_dim)
               joint_actions: torch.Tensor,        # (B, joint_action_dim)
               rewards: torch.Tensor,              # (B, 1)
               next_joint_states: torch.Tensor,    # (B, joint_state_dim)
               dones: torch.Tensor,                # (B, 1)
               next_actions_all: torch.Tensor,     # (B, joint_action_dim)  — from target actors
               own_states: torch.Tensor,           # (B, state_dim)  — for actor gradient
               ) -> dict:
        """
        One full MATD3 update for this agent.

        Step 1 — Critic update (Eq. 3):
          y = r_i + gamma * min_{k=1,2} Q'^k_i(s', a')|_{a'=mu'(s')}
          Loss = E[(Q^j_i(s,a) - y)^2]   for j = 1, 2

        Step 2 — Delayed actor update (Eq. 4):
          Every `policy_delay` critic steps:
          grad = E[grad_{theta_i} mu_i * grad_{a_i} Q^1_i(s, a)]
        """
        self._critic_updates += 1
        losses = {}

        # ---- Critic update ----
        with torch.no_grad():
            # Target policy smoothing (Eq. 5)
            noise = (torch.randn_like(next_actions_all) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)
            next_a_smooth = (next_actions_all + noise).clamp(
                self.action_low.repeat(next_actions_all.shape[-1] //
                                       self.action_dim),
                self.action_high.repeat(next_actions_all.shape[-1] //
                                        self.action_dim),
            )

            q1_next = self.critic1_target(next_joint_states, next_a_smooth)
            q2_next = self.critic2_target(next_joint_states, next_a_smooth)
            q_next  = torch.min(q1_next, q2_next)            # clipped double-Q
            y       = rewards + self.gamma * (1.0 - dones) * q_next

        q1 = self.critic1(joint_states, joint_actions)
        q2 = self.critic2(joint_states, joint_actions)

        critic1_loss = F.mse_loss(q1, y)
        critic2_loss = F.mse_loss(q2, y)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        losses["critic1"] = critic1_loss.item()
        losses["critic2"] = critic2_loss.item()

        # ---- Delayed actor update ----
        if self._critic_updates % self.policy_delay == 0:
            # Recompute this agent's action from its own state
            a_i = self._scale(self.actor(own_states))         # (B, action_dim)

            # Splice a_i back into the joint action tensor
            joint_actions_for_grad = joint_actions.clone()
            start = self.id * self.action_dim
            end   = start + self.action_dim
            joint_actions_for_grad[:, start:end] = a_i

            # Maximize Q^1 w.r.t. actor parameters (gradient ascent)
            actor_loss = -self.critic1(joint_states,
                                       joint_actions_for_grad).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            losses["actor"] = actor_loss.item()

            # Soft target network updates
            _soft_update(self.actor,   self.actor_target,   self.tau)
            _soft_update(self.critic1, self.critic1_target, self.tau)
            _soft_update(self.critic2, self.critic2_target, self.tau)

        return losses


# ---------------------------------------------------------------------------
# MATD3 coordinator
# ---------------------------------------------------------------------------

class MATD3:
    """
    Multi-Agent Twin Delayed DDPG.

    Coordinates N agents sharing one replay buffer.
    Implements centralized training / decentralized execution (CTDE).
    """

    def __init__(self,
                 n_agents: int,
                 state_dim: int,
                 action_dim: int,
                 action_low: np.ndarray,
                 action_high: np.ndarray,
                 hidden_dim: int    = 256,       # Table 2
                 lr: float          = 3e-4,       # Table 2
                 gamma: float       = 0.95,       # Table 2
                 tau: float         = 0.005,
                 policy_noise: float = 0.2,
                 noise_clip: float   = 0.5,
                 policy_delay: int   = 2,         # Table 2
                 buffer_size: int    = 1_000_000, # Table 2
                 batch_size: int     = 256):      # Table 2

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
        """
        Decentralized execution: each agent acts on its own state.

        Args:
            states : (N, state_dim)
        Returns:
            actions: (N, action_dim)
        """
        return np.array([
            agent.select_action(states[i], explore_noise)
            for i, agent in enumerate(self.agents)
        ])

    def store_transition(self, states, actions, rewards, next_states, dones):
        """Push one time-step into the shared replay buffer."""
        self.buffer.push(states, actions, rewards, next_states, dones)

    def train_step(self) -> dict:
        """
        Sample a mini-batch and update all agents.
        Returns a dict of losses keyed by agent index.
        """
        if len(self.buffer) < self.batch_size:
            return {}

        joint_s, joint_a, rewards, joint_ns, dones = self.buffer.sample(
            self.batch_size)

        # joint_s  : (B, N, state_dim)  -> flatten to (B, N*state_dim)
        B = joint_s.shape[0]
        flat_js  = joint_s.view(B, -1)    # (B, N*state_dim)
        flat_ja  = joint_a.view(B, -1)    # (B, N*action_dim)
        flat_njs = joint_ns.view(B, -1)   # (B, N*state_dim)

        # Compute next joint actions from all target actors (for critic target)
        with torch.no_grad():
            next_actions = []
            for i, agent in enumerate(self.agents):
                ns_i   = joint_ns[:, i, :]          # (B, state_dim)
                a_next = agent.actor_target(ns_i)   # (B, action_dim)
                a_next = agent._scale(a_next)
                next_actions.append(a_next)
            next_joint_a = torch.cat(next_actions, dim=-1)  # (B, N*action_dim)

        all_losses = {}
        for i, agent in enumerate(self.agents):
            r_i    = rewards[:, i].unsqueeze(1)   # (B, 1)
            done_i = dones[:, i].unsqueeze(1)     # (B, 1)
            own_si = joint_s[:, i, :]             # (B, state_dim)

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

    def train(self, env, n_episodes: int = 100_000,
              max_steps: int = 300,
              explore_noise: float = 0.1,
              log_interval: int = 1000) -> list[float]:
        """
        Main training loop.

        Args:
            env          : environment exposing
                               reset() -> (N, state_dim) ndarray
                               step(actions) -> (next_states, rewards, dones, info)
                           Compatible with both ``MultiUAVEnvWrapper`` and
                           ``DummyMultiAgentEnv``.
            n_episodes   : total training episodes
            max_steps    : max steps per episode
            explore_noise: Gaussian exploration noise std
            log_interval : print average return every N episodes

        Returns:
            episode_returns: list of total returns per episode
        """
        episode_returns = []

        for ep in range(1, n_episodes + 1):
            states    = env.reset()                 # (N, state_dim)
            ep_return = 0.0

            for _ in range(max_steps):
                actions = self.select_actions(states, explore_noise)
                next_states, rewards, dones, info = env.step(actions)  # ← name it info
                if any(rewards > 0):
                    print(f"  Attack event at step {info}, rewards: {rewards}")

                self.store_transition(states, actions, rewards, next_states, dones)
                self.train_step()

                ep_return += float(np.sum(rewards))
                states     = next_states

                if all(dones):
                    break

            episode_returns.append(ep_return)

            if ep % log_interval == 0:
                avg = np.mean(episode_returns[-log_interval:])
                print(f"Episode {ep:>7d} | Avg Return (last {log_interval}): {avg:.7f}")

        return episode_returns


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _soft_update(net: nn.Module, target: nn.Module, tau: float):
    """Polyak averaging: theta_target <- tau*theta + (1-tau)*theta_target."""
    for p, tp in zip(net.parameters(), target.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    NUM_UAV    = 4   # must be even (equal teams)
    ACTION_DIM = 3   # [T, L, beta]

    action_low  = np.array([T_MIN, L_MIN, BETA_MIN])
    action_high = np.array([T_MAX, L_MAX, BETA_MAX])

    # Real UAV environment — state_dim is derived from the team configuration:
    #   4 + 11*(N/2) + 8*(N/2-1) = 34 for N=4
    env = MultiUAVEnvWrapper(num_uav=NUM_UAV, dt=DT, g=G, spawn_range=3000.0, spawn_radius=800.0)
    STATE_DIM = env.state_dim

    matd3 = MATD3(
        n_agents     = NUM_UAV,
        state_dim    = STATE_DIM,
        action_dim   = ACTION_DIM,
        action_low   = action_low,
        action_high  = action_high,
        # Hyper-parameters from Table 2
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

    print(f"MATD3 + UAV Sim | {NUM_UAV} agents | state_dim={STATE_DIM} | action_dim={ACTION_DIM}")
    print("Running smoke test (100 episodes)...")

    returns = matd3.train(env, n_episodes=100, max_steps=2000, log_interval=50)
    print(f"Smoke test complete. Final episode return: {returns[-1]:.3f}")
