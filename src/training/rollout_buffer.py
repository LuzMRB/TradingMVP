"""
rollout_buffer.py — Buffer de datos para PPO
=============================================

Almacena N transiciones (obs, action, reward, done, log_prob, value)
y cuando está lleno, calcula ventajas con GAE para el update de PPO.

Flujo:
    1. Agente hace 2048 steps → se guardan aquí
    2. compute_gae() → calcula advantages y returns
    3. get_batches() → minibatches aleatorios para PPO update
    4. reset() → vacía para el siguiente rollout
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
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lam = lam

        # Pre-allocate arrays
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)

        # Se calculan con compute_gae()
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0

    def add(self, obs, action, reward, done, log_prob, value):
        """Guarda una transición. Llamado una vez por step."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, last_done: bool):
        """
        Calcula GAE (Generalized Advantage Estimation).

        Para cada step calcula: ¿fue esta acción mejor o peor
        de lo que el critic predijo?

        advantage > 0 → la acción fue mejor de lo esperado → subir su prob
        advantage < 0 → la acción fue peor de lo esperado → bajar su prob
        """
        last_gae = 0.0
        n = self.ptr

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = float(last_done)
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            next_non_terminal = 1.0 - next_done

            # TD error: reward real + valor futuro - valor predicho
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )

            # GAE acumula TD errors con descuento γλ
            last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        # Returns = advantages + values (target para entrenar el critic)
        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, batch_size: int = 512):
        """
        Genera minibatches aleatorios (barajados) para PPO update.

        Yields: dict con tensores PyTorch por cada minibatch
        """
        n = self.ptr
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            yield {
                "obs": torch.FloatTensor(self.observations[idx]),
                "actions": torch.LongTensor(self.actions[idx]),
                "old_log_probs": torch.FloatTensor(self.log_probs[idx]),
                "advantages": torch.FloatTensor(self.advantages[idx]),
                "returns": torch.FloatTensor(self.returns[idx]),
            }

    def reset(self):
        """Vacía el buffer para el siguiente rollout."""
        self.ptr = 0
