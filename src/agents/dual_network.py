"""
dual_network.py — Red dual Bull/Bear + Arbitrador para MAPPO.

Arquitectura:
    Transformer Encoder (compartido)
        ↓
    ├── Bull Head  → score_bull  ∈ [0,1]  P(precio sube en k steps)
    ├── Bear Head  → score_bear  ∈ [0,1]  P(precio baja en k steps)
    ├── Arbitrador → logits (buy/sell/hold), recibe features + scores
    └── Critic     → valor escalar (para entrenamiento del arbitrador)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class DualDirectionalNetwork(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_frames: int = 10,
        arb_hidden: int = 64,
    ):
        super().__init__()
        self.n_frames  = n_frames
        self.seq_len   = 11 * n_frames
        self.token_dim = 4
        self.d_model   = d_model

        # ── Encoder compartido ──────────────────────────────
        self.input_proj = nn.Linear(self.token_dim, d_model)

        pe  = torch.zeros(self.seq_len, d_model)
        pos = torch.arange(self.seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Cabezas direccionales ────────────────────────────
        self.bull_head = nn.Linear(d_model, 1)   # sigmoid → P(sube)
        self.bear_head = nn.Linear(d_model, 1)   # sigmoid → P(baja)

        # ── Arbitrador ───────────────────────────────────────
        # Recibe features del encoder + score_bull + score_bear
        self.arbitrator = nn.Sequential(
            nn.Linear(d_model + 2, arb_hidden),
            nn.ReLU(),
            nn.Linear(arb_hidden, action_dim),
        )

        # ── Critic (para PPO del arbitrador) ─────────────────
        self.critic = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.bull_head.weight, gain=0.01)
        nn.init.constant_(self.bull_head.bias, 0.0)
        nn.init.orthogonal_(self.bear_head.weight, gain=0.01)
        nn.init.constant_(self.bear_head.bias, 0.0)
        for layer in self.arbitrator:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        batch_shape = obs.shape[:-1]
        tokens = obs.reshape(*batch_shape, self.seq_len, self.token_dim)
        flat   = tokens.reshape(-1, self.seq_len, self.token_dim)
        x = self.input_proj(flat) + self.pe
        x = self.transformer(x)
        x = x.mean(dim=1)
        return x.reshape(*batch_shape, self.d_model)

    def get_scores(self, obs: torch.Tensor):
        """Devuelve scores bull/bear y features del encoder.
        Bull/Bear usan features.detach() para no contaminar el encoder
        con gradiente de ruido direccional."""
        features   = self._encode(obs)
        feats_stop = features.detach()
        score_bull = torch.sigmoid(self.bull_head(feats_stop)).squeeze(-1)
        score_bear = torch.sigmoid(self.bear_head(feats_stop)).squeeze(-1)
        return score_bull, score_bear, features

    def get_action_and_value(self, obs: torch.Tensor):
        """Usado en rollout — devuelve acción, log_prob, valor, scores."""
        score_bull, score_bear, features = self.get_scores(obs)
        scores    = torch.stack([score_bull, score_bear], dim=-1)
        arb_input = torch.cat([features, scores], dim=-1)
        logits    = self.arbitrator(arb_input)
        dist      = Categorical(logits=logits)
        action    = dist.sample()
        value     = self.critic(features).squeeze(-1)
        return action, dist.log_prob(action), value, score_bull, score_bear

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Usado en PPO update — re-evalúa acciones con pesos actuales."""
        score_bull, score_bear, features = self.get_scores(obs)
        scores    = torch.stack([score_bull, score_bear], dim=-1)
        arb_input = torch.cat([features, scores], dim=-1)
        logits    = self.arbitrator(arb_input)
        dist      = Categorical(logits=logits)
        value     = self.critic(features).squeeze(-1)
        return dist.log_prob(actions), dist.entropy(), value, score_bull, score_bear

    def forward(self, obs: torch.Tensor):
        """Inferencia greedy — devuelve logits y valor."""
        score_bull, score_bear, features = self.get_scores(obs)
        scores    = torch.stack([score_bull, score_bear], dim=-1)
        arb_input = torch.cat([features, scores], dim=-1)
        return self.arbitrator(arb_input), self.critic(features)
