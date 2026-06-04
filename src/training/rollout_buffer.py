"""
rollout_buffer.py — Buffer de datos para PPO (single-env y multi-env).

Para n_envs=1 el comportamiento es idéntico al original.
Para n_envs>1 almacena (rollout_length, n_envs, ...) internamente,
calcula GAE por env y aplana a (T*N,) para los minibatches.
"""

import numpy as np
import torch


class RolloutBuffer:

    def __init__(
        self,
        buffer_size: int = 2048,
        obs_dim: int = 44,
        gamma: float = 0.999,
        lam: float = 0.95,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.lam = lam
        self.n_envs = n_envs

        N = n_envs
        self.observations = np.zeros((buffer_size, N, obs_dim), dtype=np.float32)
        self.actions      = np.zeros((buffer_size, N), dtype=np.int64)
        self.rewards      = np.zeros((buffer_size, N), dtype=np.float32)
        self.dones        = np.zeros((buffer_size, N), dtype=np.float32)
        self.log_probs    = np.zeros((buffer_size, N), dtype=np.float32)
        self.values       = np.zeros((buffer_size, N), dtype=np.float32)
        self.advantages   = np.zeros((buffer_size, N), dtype=np.float32)
        self.returns      = np.zeros((buffer_size, N), dtype=np.float32)

        self.ptr = 0

    def add(self, obs, action, reward, done, log_prob, value):
        """
        Guarda una transición.
        Para n_envs=1: obs (obs_dim,), action scalar, reward scalar, etc.
        Para n_envs>1: obs (N, obs_dim), action (N,), reward (N,), etc.
        """
        N = self.n_envs
        if N == 1:
            self.observations[self.ptr, 0] = obs
            self.actions[self.ptr, 0]      = action
            self.rewards[self.ptr, 0]      = reward
            self.dones[self.ptr, 0]        = float(done)
            self.log_probs[self.ptr, 0]    = log_prob
            self.values[self.ptr, 0]       = value
        else:
            self.observations[self.ptr] = obs           # (N, obs_dim)
            self.actions[self.ptr]      = action        # (N,)
            self.rewards[self.ptr]      = reward        # (N,)
            self.dones[self.ptr]        = done.astype(np.float32)  # (N,)
            self.log_probs[self.ptr]    = log_prob      # (N,)
            self.values[self.ptr]       = value         # (N,)
        self.ptr += 1

    def compute_gae(self, last_value, last_done):
        """
        Calcula GAE por env y luego computa returns.

        Args:
            last_value: scalar (n_envs=1) o array (N,) (n_envs>1)
            last_done:  bool   (n_envs=1) o array (N,) bool
        """
        N = self.n_envs
        n = self.ptr

        # Normalizar a arrays (N,)
        lv = np.full(N, last_value) if np.isscalar(last_value) else np.asarray(last_value, dtype=np.float32)
        ld = np.full(N, float(last_done)) if np.isscalar(last_done) else np.asarray(last_done, dtype=np.float32)

        last_gae = np.zeros(N, dtype=np.float32)

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = lv
                next_done  = ld
            else:
                next_value = self.values[t + 1]   # (N,)
                next_done  = self.dones[t + 1]    # (N,)

            next_non_terminal = 1.0 - next_done
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, batch_size: int = 512):
        """
        Aplana (T, N) → (T*N,) y genera minibatches aleatorios.
        """
        T = self.ptr
        N = self.n_envs
        total = T * N

        flat_obs       = self.observations[:T].reshape(total, self.obs_dim)
        flat_actions   = self.actions[:T].reshape(total)
        flat_log_probs = self.log_probs[:T].reshape(total)
        flat_adv       = self.advantages[:T].reshape(total)
        flat_returns   = self.returns[:T].reshape(total)

        indices = np.random.permutation(total)
        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs":          torch.FloatTensor(flat_obs[idx]),
                "actions":      torch.LongTensor(flat_actions[idx]),
                "old_log_probs": torch.FloatTensor(flat_log_probs[idx]),
                "advantages":   torch.FloatTensor(flat_adv[idx]),
                "returns":      torch.FloatTensor(flat_returns[idx]),
            }

    def reset(self):
        self.ptr = 0
