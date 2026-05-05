import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Logging helpers  (unchanged from MATD3)
# ---------------------------------------------------------------------------

_HEADER_WIDTH = 72

def _print_header(title: str) -> None:
    print("\n" + "=" * _HEADER_WIDTH)
    print(f"  {title}")
    print("=" * _HEADER_WIDTH)

def _print_separator() -> None:
    print("  " + "─" * (_HEADER_WIDTH - 2))

def _print_attack_event(step: int, rewards: np.ndarray) -> None:
    r_str = "  ".join(f"A{i}={r:+.2f}" for i, r in enumerate(rewards))
    print(f"     Attack  step={step:>4d}  │  {r_str}")


# ---------------------------------------------------------------------------
# Reward plot  (unchanged)
# ---------------------------------------------------------------------------

def save_reward_plot(episode_returns: list[float],
                     log_interval: int,
                     path: str = "reward_history.png") -> None:
    returns = np.array(episode_returns)
    eps     = np.arange(1, len(returns) + 1)
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
    ax.set_title("ATT-MATD3 – UAV Combat: Return over Training", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  📊  Reward plot saved → {path}")


# ---------------------------------------------------------------------------
# UAV environment adapter  (unchanged from MATD3)
# ---------------------------------------------------------------------------

class _FixedSizeMultiUAVSim(MultiUAVSimulator):
    """Keeps state array fixed-size by zeroing destroyed agents instead of removing them."""
    
    def remove_destroyed(self) -> None:
        # Zero out state rows for destroyed agents but keep array size intact
        if hasattr(self, 'destroyed') and hasattr(self, 'state'):
            for i in range(len(self.destroyed)):
                if self.destroyed[i]:
                    self.state[i] = 0.0


class MultiUAVEnvWrapper:
    """
    Adapter that presents MultiUAVSimulator to ATT-MATD3.train().

    The wrapper additionally exposes per-observation slice dimensions so
    that the attention networks can split the flat state vector back into
    (own, opponents, allies) segments:

        own_dim   = 4   (speed + 3 attitude angles)
        opp_dim   = 11  (per-opponent features)
        ally_dim  = 8   (per-ally features)
        n_opp     = num_uav // 2
        n_ally    = num_uav // 2 - 1
        state_dim = own_dim + opp_dim * n_opp + ally_dim * n_ally
    """

    OWN_DIM  = 4
    OPP_DIM  = 11
    ALLY_DIM = 8

    def __init__(self, num_uav: int = 4, dt: float = DT, g: float = G,
                 spawn_range: float = 3000.0, spawn_radius: float = 800.0):
        if num_uav % 2 != 0:
            raise ValueError("num_uav must be even (equal-sized teams).")
        self.num_uav = num_uav
        self._dt = dt
        self._g  = g

        self.n_opp  = num_uav // 2
        self.n_ally = num_uav // 2 - 1
        self.state_dim = (self.OWN_DIM
                          + self.OPP_DIM  * self.n_opp
                          + self.ALLY_DIM * self.n_ally)

        self._sim: _FixedSizeMultiUAVSim | None = None
        self._spawn_range  = spawn_range
        self._spawn_radius = spawn_radius

    def reset(self) -> np.ndarray:
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
        actions = np.asarray(actions, dtype=np.float64).reshape(self.num_uav, 3)
        actions[self._sim.destroyed] = 0.0

        _, rewards, episode_done, info = self._sim.step(actions)
        next_states = self._build_obs()

        per_agent_done = self._sim.destroyed.copy()
        if episode_done:
            per_agent_done[:] = True

        return next_states, rewards.astype(np.float32), per_agent_done, info

    def _build_obs(self) -> np.ndarray:
        out = np.zeros((self.num_uav, self.state_dim), dtype=np.float32)
        for i in range(self.num_uav):
            if not self._sim.destroyed[i]:
                raw = self._sim.get_observations(i)
                n   = min(len(raw), self.state_dim)
                out[i, :n] = raw[:n].astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# Replay buffer  (unchanged)
# ---------------------------------------------------------------------------

class ReplayBuffer:

    def __init__(self, capacity: int = 1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, joint_state, joint_action, rewards,
             next_joint_state, dones):
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
# Attention module  (Eq. 7 / 8 in the paper)
# ---------------------------------------------------------------------------

class AttentionModule(nn.Module):
    """
    Single-head dot-product attention with death masking.

    Given a query (the agent's own embedding) and a set of key/value
    pairs (opponent or ally embeddings), computes:

        alpha_j = (q^T W_q^T W_k k_j) / sqrt(d)
        mask:    alpha_j = -inf  for destroyed entities
        v_agg   = sum_j  softmax(alpha_j) * W_v * x_j

    Args:
        query_dim : dimensionality of the query vector (own-state encoding)
        kv_dim    : dimensionality of each key/value vector (entity encoding)
        embed_dim : projected dimensionality (d in the paper)
    """

    def __init__(self, query_dim: int, kv_dim: int, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(query_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(kv_dim,   embed_dim, bias=False)
        self.W_v = nn.Linear(kv_dim,   embed_dim, bias=False)

    def forward(self,
                query: torch.Tensor,          # (B, query_dim)
                keys:  torch.Tensor,          # (B, N_entities, kv_dim)
                mask:  torch.Tensor | None = None  # (B, N_entities) bool, True = dead
               ) -> torch.Tensor:             # (B, embed_dim)
        """
        Args:
            query : own-state encoding, shape (B, query_dim)
            keys  : entity state encodings, shape (B, N, kv_dim)
            mask  : boolean tensor, True where entity is destroyed (B, N)

        Returns:
            Weighted sum of value vectors, shape (B, embed_dim)
        """
        B, N, _ = keys.shape

        q = self.W_q(query)                          # (B, embed_dim)
        k = self.W_k(keys)                           # (B, N, embed_dim)
        v = self.W_v(keys)                           # (B, N, embed_dim)

        # Scaled dot-product scores: (B, N)
        scores = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) / (self.embed_dim ** 0.5)

        # Death masking: set destroyed entity scores to -inf
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)      # (B, N)

        # Guard against all-masked rows (all -inf → NaN after softmax)
        weights = torch.nan_to_num(weights, nan=0.0)

        # Weighted sum: (B, embed_dim)
        agg = torch.bmm(weights.unsqueeze(1), v).squeeze(1)
        return agg


# ---------------------------------------------------------------------------
# ATT-MATD3 Actor  (policy network, Fig. 3 / Eq. 6-7)
# ---------------------------------------------------------------------------

class AttentionActor(nn.Module):
    """
    Decentralized policy with attention over opponents and allies.

    Architecture (mirrors Fig. 3):
        1. Linear encoder for own state  →  own_enc  (embed_dim)
        2. Linear encoder for each opponent state  →  opp_enc_i  (embed_dim)
        3. Attention(own_enc, opp_enc_*) with death mask  →  v_oppo  (embed_dim)
        4. Linear encoder for each ally state  →  ally_enc_i  (embed_dim)
        5. Attention(own_enc, ally_enc_*) with death mask  →  v_ally  (embed_dim)
        6. Concat([v_oppo, v_ally])  →  2-layer MLP  →  tanh  →  action

    Args:
        own_dim    : dimension of agent's own-state slice (4 in the paper)
        opp_dim    : dimension of per-opponent feature slice (11 in the paper)
        ally_dim   : dimension of per-ally feature slice (8 in the paper)
        n_opp      : number of opponents observed
        n_ally     : number of allies observed (self excluded)
        action_dim : output action dimension (3 in the paper)
        embed_dim  : attention projection dimension
        hidden_dim : MLP hidden layer width
    """

    def __init__(self,
                 own_dim: int,
                 opp_dim: int,
                 ally_dim: int,
                 n_opp: int,
                 n_ally: int,
                 action_dim: int,
                 embed_dim: int  = 64,
                 hidden_dim: int = 256):
        super().__init__()
        self.own_dim  = own_dim
        self.opp_dim  = opp_dim
        self.ally_dim = ally_dim
        self.n_opp    = n_opp
        self.n_ally   = n_ally

        # Encoders that project raw features into the shared embed space
        self.own_encoder  = nn.Sequential(nn.Linear(own_dim,  embed_dim), nn.ReLU())
        self.opp_encoder  = nn.Sequential(nn.Linear(opp_dim,  embed_dim), nn.ReLU())
        self.ally_encoder = nn.Sequential(nn.Linear(ally_dim, embed_dim), nn.ReLU())

        # Two attention pools — one per entity type
        self.opp_att  = AttentionModule(embed_dim, embed_dim, embed_dim)
        self.ally_att = AttentionModule(embed_dim, embed_dim, embed_dim)

        # 2-layer MLP that maps concatenated embeddings → action
        in_dim = embed_dim * 2  # v_oppo ++ v_ally
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def forward(self,
                state: torch.Tensor,             # (B, state_dim)
                opp_mask:  torch.Tensor | None = None,   # (B, n_opp)  bool
                ally_mask: torch.Tensor | None = None,   # (B, n_ally) bool
               ) -> torch.Tensor:                # (B, action_dim)

        own, opps, allies = self._split(state)   # own (B,4), opps (B,n,11), allies (B,n,8)

        # Encode
        own_enc  = self.own_encoder(own)                         # (B, embed_dim)
        opp_enc  = self.opp_encoder(opps.reshape(-1, self.opp_dim))
        opp_enc  = opp_enc.reshape(state.shape[0], self.n_opp, -1)  # (B, n_opp, embed_dim)

        # Attend
        v_oppo = self.opp_att(own_enc, opp_enc, opp_mask)       # (B, embed_dim)

        if self.n_ally > 0:
            ally_enc = self.ally_encoder(allies.reshape(-1, self.ally_dim))
            ally_enc = ally_enc.reshape(state.shape[0], self.n_ally, -1)
            v_ally   = self.ally_att(own_enc, ally_enc, ally_mask)  # (B, embed_dim)
        else:
            v_ally = torch.zeros_like(v_oppo)

        out = self.mlp(torch.cat([v_oppo, v_ally], dim=-1))
        return out

    def _split(self, state: torch.Tensor):
        """Split flat state vector into (own, opponents, allies) tensors."""
        B = state.shape[0]
        ptr = 0

        own   = state[:, ptr : ptr + self.own_dim];  ptr += self.own_dim
        opps  = state[:, ptr : ptr + self.opp_dim * self.n_opp]
        opps  = opps.reshape(B, self.n_opp, self.opp_dim);  ptr += self.opp_dim * self.n_opp
        allies = state[:, ptr : ptr + self.ally_dim * self.n_ally]
        allies = allies.reshape(B, self.n_ally, self.ally_dim)

        return own, opps, allies


# ---------------------------------------------------------------------------
# ATT-MATD3 Critic  (centralized Q-function, Eq. 8)
# ---------------------------------------------------------------------------

class AttentionCritic(nn.Module):
    """
    Centralized Q-function with attention over allied (state, action) pairs.

    Architecture (mirrors Eq. 8):
        1. For each ally j: encode (s_j, a_j) → e_j   (embed_dim)
        2. Attention(e_i, e_j for j≠i) with death mask → weighted sum
        3. 2-layer MLP → Q-value scalar

    In practice we include ALL agents (self + allies) as the key set and
    let the attention learn whom to focus on, consistent with the paper's
    centralized critic that receives joint (s, a).

    Args:
        n_agents      : total number of agents
        own_state_dim : per-agent state dimension
        action_dim    : per-agent action dimension
        embed_dim     : attention projection size
        hidden_dim    : MLP hidden width
    """

    def __init__(self,
                 n_agents: int,
                 own_state_dim: int,
                 action_dim: int,
                 embed_dim: int  = 64,
                 hidden_dim: int = 256):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = own_state_dim
        self.action_dim = action_dim

        entity_dim = own_state_dim + action_dim
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim,  embed_dim), nn.ReLU(),
        )

        self.att = AttentionModule(embed_dim, embed_dim, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self,
                joint_state:  torch.Tensor,          # (B, N * state_dim)
                joint_action: torch.Tensor,          # (B, N * action_dim)
                agent_id: int,
                done_mask: torch.Tensor | None = None  # (B, N) bool
               ) -> torch.Tensor:                    # (B, 1)

        B = joint_state.shape[0]
        N = self.n_agents

        states  = joint_state.reshape(B, N, self.state_dim)
        actions = joint_action.reshape(B, N, self.action_dim)

        # Concatenate (s_j, a_j) for each agent j
        sa = torch.cat([states, actions], dim=-1)   # (B, N, state+action)

        # Encode every entity
        enc = self.entity_encoder(sa.reshape(B * N, -1))
        enc = enc.reshape(B, N, -1)                 # (B, N, embed_dim)

        # Query is this agent's own encoding
        query = enc[:, agent_id, :]                 # (B, embed_dim)

        # Attend over ALL agents (self-attention gives global context)
        agg = self.att(query, enc, done_mask)        # (B, embed_dim)

        return self.mlp(agg)                         # (B, 1)


# ---------------------------------------------------------------------------
# Per-agent ATT-MATD3 logic
# ---------------------------------------------------------------------------

class ATTMATD3Agent:
    """
    Single agent in the ATT-MATD3 framework.

    Replaces the flat Actor/Critic from MATD3Agent with the
    attention-based AttentionActor / AttentionCritic.
    Everything else (double-Q, delayed policy update, target
    smoothing, soft updates) is identical to MATD3.
    """

    def __init__(self,
                 agent_id: int,
                 # observation split dimensions
                 own_dim: int,
                 opp_dim: int,
                 ally_dim: int,
                 n_opp: int,
                 n_ally: int,
                 # action
                 action_dim: int,
                 action_low: np.ndarray,
                 action_high: np.ndarray,
                 # critic joint dimensions
                 n_agents: int,
                 state_dim: int,
                 # hyper-params
                 embed_dim: int      = 64,
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

        # --- Actor ---
        actor_kwargs = dict(
            own_dim=own_dim, opp_dim=opp_dim, ally_dim=ally_dim,
            n_opp=n_opp, n_ally=n_ally,
            action_dim=action_dim, embed_dim=embed_dim, hidden_dim=hidden_dim,
        )
        self.actor        = AttentionActor(**actor_kwargs).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt    = optim.Adam(self.actor.parameters(), lr=lr)

        # --- Critic ---
        critic_kwargs = dict(
            n_agents=n_agents, own_state_dim=state_dim,
            action_dim=action_dim, embed_dim=embed_dim, hidden_dim=hidden_dim,
        )
        self.critic1        = AttentionCritic(**critic_kwargs).to(DEVICE)
        self.critic2        = AttentionCritic(**critic_kwargs).to(DEVICE)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic1_opt    = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt    = optim.Adam(self.critic2.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, state: np.ndarray,
                      explore_noise: float = 0.1) -> np.ndarray:
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        a = self.actor(s)           # no mask at inference (live agents only)
        a = self._scale(a).squeeze(0)
        if explore_noise > 0:
            noise = torch.randn_like(a) * explore_noise
            a = torch.clamp(a + noise, self.action_low, self.action_high)
        return a.cpu().numpy()

    def _scale(self, a: torch.Tensor) -> torch.Tensor:
        lo, hi = self.action_low, self.action_high
        return lo + (a + 1.0) * 0.5 * (hi - lo)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self,
               joint_states: torch.Tensor,        # (B, N*state_dim)
               joint_actions: torch.Tensor,       # (B, N*action_dim)
               rewards: torch.Tensor,             # (B, 1)
               next_joint_states: torch.Tensor,   # (B, N*state_dim)
               dones: torch.Tensor,               # (B, 1)
               next_actions_all: torch.Tensor,    # (B, N*action_dim)
               own_states: torch.Tensor,          # (B, state_dim)
               done_mask: torch.Tensor | None,    # (B, N) bool — for critic attention
               next_done_mask: torch.Tensor | None,
               ) -> dict:

        self._critic_updates += 1
        losses = {}

        # ── Critic update ─────────────────────────────────────────────
        with torch.no_grad():
            noise = (torch.randn_like(next_actions_all) * self.policy_noise
                    ).clamp(-self.noise_clip, self.noise_clip)
            next_a_smooth = next_actions_all + noise

            q1_next = self.critic1_target(next_joint_states, next_a_smooth,
                                          self.id, next_done_mask)
            q2_next = self.critic2_target(next_joint_states, next_a_smooth,
                                          self.id, next_done_mask)
            q_next  = torch.min(q1_next, q2_next)
            y       = rewards + self.gamma * (1.0 - dones) * q_next

        q1 = self.critic1(joint_states, joint_actions, self.id, done_mask)
        q2 = self.critic2(joint_states, joint_actions, self.id, done_mask)

        c1_loss = F.mse_loss(q1, y)
        c2_loss = F.mse_loss(q2, y)
        self.critic1_opt.zero_grad(); c1_loss.backward(); self.critic1_opt.step()
        self.critic2_opt.zero_grad(); c2_loss.backward(); self.critic2_opt.step()
        losses["critic1"] = c1_loss.item()
        losses["critic2"] = c2_loss.item()

        # ── Delayed actor update ──────────────────────────────────────
        if self._critic_updates % self.policy_delay == 0:
            a_i = self._scale(self.actor(own_states))  # (B, action_dim)

            # Rebuild joint action with this agent's fresh action
            joint_a_grad = joint_actions.clone()
            start = self.id * self.action_dim
            end   = start + self.action_dim
            joint_a_grad[:, start:end] = a_i

            actor_loss = -self.critic1(joint_states, joint_a_grad,
                                       self.id, done_mask).mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            losses["actor"] = actor_loss.item()

            _soft_update(self.actor,   self.actor_target,   self.tau)
            _soft_update(self.critic1, self.critic1_target, self.tau)
            _soft_update(self.critic2, self.critic2_target, self.tau)

        return losses


# ---------------------------------------------------------------------------
# ATT-MATD3 coordinator
# ---------------------------------------------------------------------------

class ATTMATD3:
    """
    Multi-Agent Twin Delayed DDPG with Attention Mechanism (ATT-MATD3).

    Drop-in replacement for MATD3; call signature of train() is identical.
    """

    def __init__(self,
                 n_agents: int,
                 state_dim: int,
                 action_dim: int,
                 action_low: np.ndarray,
                 action_high: np.ndarray,
                 # observation split (must match MultiUAVEnvWrapper)
                 own_dim: int   = MultiUAVEnvWrapper.OWN_DIM,
                 opp_dim: int   = MultiUAVEnvWrapper.OPP_DIM,
                 ally_dim: int  = MultiUAVEnvWrapper.ALLY_DIM,
                 n_opp: int     = 0,   # set from env
                 n_ally: int    = 0,   # set from env
                 # hyper-params
                 embed_dim: int      = 64,
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

        self.agents = [
            ATTMATD3Agent(
                agent_id     = i,
                own_dim      = own_dim,
                opp_dim      = opp_dim,
                ally_dim     = ally_dim,
                n_opp        = n_opp,
                n_ally       = n_ally,
                action_dim   = action_dim,
                action_low   = action_low,
                action_high  = action_high,
                n_agents     = n_agents,
                state_dim    = state_dim,
                embed_dim    = embed_dim,
                hidden_dim   = hidden_dim,
                lr           = lr,
                gamma        = gamma,
                tau          = tau,
                policy_noise = policy_noise,
                noise_clip   = noise_clip,
                policy_delay = policy_delay,
            )
            for i in range(n_agents)
        ]

        self.buffer = ReplayBuffer(capacity=buffer_size)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def select_actions(self, states: np.ndarray,
                       explore_noise: float = 0.1) -> np.ndarray:
        return np.array([
            agent.select_action(states[i], explore_noise)
            for i, agent in enumerate(self.agents)
        ])

    def store_transition(self, states, actions, rewards, next_states, dones):
        self.buffer.push(states, actions, rewards, next_states, dones)

    @staticmethod
    def _build_done_mask(dones: torch.Tensor, n_agents: int) -> torch.Tensor:
        """
        Convert per-agent done vector (B, N) to a boolean mask for the
        attention critic.  Shape: (B, N).  True = agent is dead.
        """
        return dones.bool()  # dones already stored as float; cast to bool

    def train_step(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return {}

        joint_s, joint_a, rewards, joint_ns, dones = self.buffer.sample(
            self.batch_size)

        B  = joint_s.shape[0]
        N  = self.n_agents
        SD = self.state_dim
        AD = self.action_dim

        # Flatten for critic (critic receives flat joint tensors internally)
        flat_js  = joint_s.view(B, N * SD)
        flat_ja  = joint_a.view(B, N * AD)
        flat_njs = joint_ns.view(B, N * SD)

        # Death masks for attention (B, N) bool
        done_mask      = self._build_done_mask(dones, N)       # (B, N)
        next_done_mask = done_mask                              # same step's dones

        # Compute next actions from target actors
        with torch.no_grad():
            next_actions = []
            for i, agent in enumerate(self.agents):
                ns_i   = joint_ns[:, i, :]                     # (B, state_dim)
                a_next = agent.actor_target(ns_i)
                a_next = agent._scale(a_next)
                next_actions.append(a_next)
            next_joint_a = torch.cat(next_actions, dim=-1)     # (B, N*AD)

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
                done_mask         = done_mask,
                next_done_mask    = next_done_mask,
            )
            all_losses[i] = losses

        return all_losses

    # ------------------------------------------------------------------
    # Main training loop  (identical interface to MATD3.train)
    # ------------------------------------------------------------------

    def train(self,
              env,
              n_episodes: int      = 100_000,
              max_steps: int       = 300,
              explore_noise: float = 0.2,
              log_interval: int    = 1000,
              plot_path: str       = "reward_history.png") -> list[float]:

        episode_returns: list[float] = []

        _print_header(
            f"ATT-MATD3 Training  │  {self.n_agents} agents  │  "
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

                self.store_transition(states, actions, rewards,
                                      next_states, dones)
                self.train_step()

                ep_return += float(np.sum(rewards))
                states     = next_states

                if all(dones):
                    break

            episode_returns.append(ep_return)

            alive = int(np.sum(~env._sim.destroyed))
            avg   = float(np.mean(episode_returns[-log_interval:]))
            print(
                f"  └─ Return: {ep_return:>10.3f} │ "
                f"Avg(last {log_interval}): {avg:>10.3f} │ "
                f"Steps: {step:>4d} │ "
                f"Alive: {alive}/{self.n_agents}"
            )

            if ep % log_interval == 0:
                _print_separator()
                print(
                    f"    Checkpoint ep {ep:,}  │  "
                    f"Avg return (last {log_interval}): {avg:+.4f}  │  "
                    f"Buffer: {len(self.buffer):,} transitions"
                )
                _print_separator()
                save_reward_plot(episode_returns, log_interval, path=plot_path)

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

        _print_separator()
        print(f"\n  Training complete.  Total episodes: {n_episodes:,}")
        save_reward_plot(episode_returns, log_interval, path=plot_path)
        return episode_returns


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _soft_update(net: nn.Module, target: nn.Module, tau: float) -> None:
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

    att_matd3 = ATTMATD3(
        n_agents     = NUM_UAV,
        state_dim    = env.state_dim,
        action_dim   = ACTION_DIM,
        action_low   = action_low,
        action_high  = action_high,
        # observation split — must match env
        own_dim      = MultiUAVEnvWrapper.OWN_DIM,
        opp_dim      = MultiUAVEnvWrapper.OPP_DIM,
        ally_dim     = MultiUAVEnvWrapper.ALLY_DIM,
        n_opp        = env.n_opp,
        n_ally       = env.n_ally,
        # hyper-params (Table 2 in the paper)
        embed_dim    = 64,
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

    returns = att_matd3.train(
        env,
        n_episodes    = 10_000,
        max_steps     = 300,
        log_interval  = 50,
        plot_path     = "reward_history_a.png",
    )

    print(f"\n  Final episode return: {returns[-1]:.3f}")