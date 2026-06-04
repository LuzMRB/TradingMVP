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

import math
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


class TransformerActorCritic(nn.Module):
    """
    Transformer Actor-Critic para PPO.

    Tokeniza la observación de 44 features como 11 tokens de 4 dims:
      - Tokens 0-9: nivel i del LOB → (bid_bps, bid_vol, ask_bps, ask_vol)
      - Token  10:  portfolio        → (holdings, cash, pnl, time_progress)

    Esto permite al Transformer aprender relaciones entre niveles del libro
    (e.g. imbalance bid/ask, profundidad) que una MLP trata como features planas.
    """

    def __init__(
        self,
        obs_dim: int = 44,
        action_dim: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len   = 11   # 10 niveles LOB + 1 portfolio
        self.token_dim = 4    # features por token
        self.d_model   = d_model

        # Proyección token_dim → d_model
        self.input_proj = nn.Linear(self.token_dim, d_model)

        # Positional encoding fijo (sinusoidal)
        pe = torch.zeros(self.seq_len, d_model)
        pos = torch.arange(self.seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cabezas actor y critic
        self.actor  = nn.Linear(d_model, action_dim)
        self.critic = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.actor.bias,  0.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (..., 44) → (..., 11, 4) → transformer → mean pool → (..., d_model)
        batch_shape = obs.shape[:-1]
        tokens = obs.reshape(*batch_shape, self.seq_len, self.token_dim)
        flat   = tokens.reshape(-1, self.seq_len, self.token_dim)
        x = self.input_proj(flat) + self.pe
        x = self.transformer(x)
        x = x.mean(dim=1)
        return x.reshape(*batch_shape, -1)

    def forward(self, obs: torch.Tensor):
        features = self._encode(obs)
        return self.actor(features), self.critic(features)

    def get_action_and_value(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values.squeeze(-1)
