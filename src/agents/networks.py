"""
networks.py — Red Neuronal Actor-Critic para PPO
=================================================

Recibe una observación (44 números del LOB) y produce:
  - Actor: probabilidades sobre 11 acciones
  - Critic: valor escalar del estado

Arquitectura:
    obs (44) → 256 → 256 → Actor (11 logits) + Critic (1 valor)

No tiene NADA que ver con ABIDES. Es PyTorch puro.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):

    def __init__(self, obs_dim: int = 44, action_dim: int = 11, hidden_dim: int = 256):
        super().__init__()

        # Backbone compartido: extrae features de la observación
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor: produce logits para cada acción
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic: produce estimación de valor del estado
        self.critic = nn.Linear(hidden_dim, 1)

        # Inicialización orthogonal (estándar en PPO)
        self._init_weights()

    def _init_weights(self):
        for module in self.shared:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """Forward completo. Retorna logits y valor."""
        features = self.shared(obs)
        return self.actor(features), self.critic(features)

    def get_action_and_value(self, obs: torch.Tensor):
        """
        Usado durante RECOLECCIÓN de datos (rollout).
        El agente observa el mercado y decide qué hacer.

        Returns: action (int), log_prob (float), value (float)
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Usado durante ENTRENAMIENTO (update PPO).
        Re-evalúa acciones ya tomadas con los pesos actualizados.

        Returns: log_probs, entropy, values
        """
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values.squeeze(-1)
