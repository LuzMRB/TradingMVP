"""
dual_buffer.py — Rollout buffer para entrenamiento dual Bull/Bear + Arbitrador.

Guarda trayectorias con dos tipos de reward:
  - arb_reward:  M2M PnL inmediato (para el arbitrador, GAE estándar)
  - bull/bear:   cambio de precio en t+k (reward retardado, asignado post-rollout)

Los targets de Bull/Bear son binarios:
  bull_target = 1 si mid_price[t+k] > mid_price[t], else 0
  bear_target = 1 - bull_target
"""

import numpy as np
import torch


class DualRolloutBuffer:

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        gamma: float = 0.999,
        lam: float   = 0.95,
        n_envs: int  = 1,
        k_delay: int = 5,
    ):
        self.T       = buffer_size
        self.N       = n_envs
        self.gamma   = gamma
        self.lam     = lam
        self.k       = k_delay
        self.ptr     = 0

        # ── Datos estándar PPO (arbitrador) ──────────────────
        self.obs         = np.zeros((self.T, self.N, obs_dim), dtype=np.float32)
        self.actions     = np.zeros((self.T, self.N),          dtype=np.int64)
        self.arb_rewards = np.zeros((self.T, self.N),          dtype=np.float32)
        self.dones       = np.zeros((self.T, self.N),          dtype=np.float32)
        self.log_probs   = np.zeros((self.T, self.N),          dtype=np.float32)
        self.values      = np.zeros((self.T, self.N),          dtype=np.float32)
        self.advantages  = np.zeros((self.T, self.N),          dtype=np.float32)
        self.returns     = np.zeros((self.T, self.N),          dtype=np.float32)

        # ── Datos direccionales (Bull/Bear) ───────────────────
        self.score_bull  = np.zeros((self.T, self.N), dtype=np.float32)
        self.score_bear  = np.zeros((self.T, self.N), dtype=np.float32)
        self.mid_prices  = np.zeros((self.T, self.N), dtype=np.float32)
        self.bull_target = np.zeros((self.T, self.N), dtype=np.float32)
        self.bear_target = np.zeros((self.T, self.N), dtype=np.float32)

    def add(
        self,
        obs, action, arb_reward, done, log_prob, value,
        score_bull, score_bear, mid_price,
    ):
        t = self.ptr
        self.obs[t]         = obs
        self.actions[t]     = action
        self.arb_rewards[t] = arb_reward
        self.dones[t]       = done
        self.log_probs[t]   = log_prob
        self.values[t]      = value
        self.score_bull[t]  = score_bull
        self.score_bear[t]  = score_bear
        self.mid_prices[t]  = mid_price
        self.ptr += 1

    def compute_directional_targets(self):
        """
        Para cada step t asigna:
            bull_target[t] = 1 si mid_price[t+k] > mid_price[t], else 0
            bear_target[t] = 1 - bull_target[t]

        Para los últimos k steps usa el último precio disponible (truncado).
        """
        T = self.ptr
        for t in range(T):
            future_t = min(t + self.k, T - 1)
            went_up  = (self.mid_prices[future_t] > self.mid_prices[t]).astype(np.float32)
            self.bull_target[t] = went_up
            self.bear_target[t] = 1.0 - went_up

    def compute_gae(self, last_value, last_done):
        """GAE sobre arb_rewards (M2M). Igual que RolloutBuffer estándar."""
        T = self.ptr

        if self.N == 1:
            last_value   = np.array([last_value],          dtype=np.float32)
            last_done_np = np.array([float(last_done)],    dtype=np.float32)
        else:
            last_value   = np.asarray(last_value,          dtype=np.float32)
            last_done_np = np.asarray(last_done,           dtype=np.float32)

        last_gae = np.zeros(self.N, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val  = last_value
                next_done = last_done_np
            else:
                next_val  = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta    = self.arb_rewards[t] + self.gamma * next_val * (1 - next_done) - self.values[t]
            last_gae = delta + self.gamma * self.lam * (1 - next_done) * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

        # Normalizar returns para que el critic no tenga que aprender escalas enormes
        T = self.ptr
        ret_flat = self.returns[:T].flatten()
        self.returns[:T] = ((self.returns[:T] - ret_flat.mean()) /
                            (ret_flat.std() + 1e-8))

    def get_batches(self, batch_size: int):
        T     = self.ptr
        total = T * self.N

        obs_flat        = torch.FloatTensor(self.obs[:T].reshape(total, -1))
        actions_flat    = torch.LongTensor(self.actions[:T].reshape(total))
        log_probs_flat  = torch.FloatTensor(self.log_probs[:T].reshape(total))
        advantages_flat = torch.FloatTensor(self.advantages[:T].reshape(total))
        returns_flat    = torch.FloatTensor(self.returns[:T].reshape(total))
        bull_tgt_flat   = torch.FloatTensor(self.bull_target[:T].reshape(total))
        bear_tgt_flat   = torch.FloatTensor(self.bear_target[:T].reshape(total))

        indices = np.random.permutation(total)
        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]
            yield {
                "obs":           obs_flat[idx],
                "actions":       actions_flat[idx],
                "old_log_probs": log_probs_flat[idx],
                "advantages":    advantages_flat[idx],
                "returns":       returns_flat[idx],
                "bull_target":   bull_tgt_flat[idx],
                "bear_target":   bear_tgt_flat[idx],
            }

    def reset(self):
        self.ptr = 0
