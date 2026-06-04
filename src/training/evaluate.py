"""
evaluate.py — Evaluación determinista del agente entrenado.

Corre n_episodes completos con la política greedy (sin exploración)
y devuelve un dict de métricas listo para guardar en JSON.
"""

import numpy as np
import torch
from typing import Callable, Dict


def evaluate_model(network, env_fn: Callable, n_episodes: int = 20, device: str = "cpu") -> Dict:
    """
    Evalúa la política entrenada en n_episodes episodios completos.

    Args:
        network:    ActorCritic o TransformerActorCritic ya entrenado
        env_fn:     callable que crea un SpyGymEnv limpio
        n_episodes: número de episodios de evaluación
        device:     "cpu" o "cuda"

    Returns:
        dict con métricas agregadas y por episodio
    """
    dev = torch.device(device)
    network.eval()

    ep_rewards    = []
    ep_lengths    = []
    ep_final_hold = []
    ep_final_pnl  = []

    env = env_fn()

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                logits, _ = network.forward(obs_t)
                action = logits.argmax(dim=-1).item()   # greedy

            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        ep_rewards.append(total_reward)
        ep_lengths.append(steps)
        ep_final_hold.append(int(info.get("holdings", 0)))
        ep_final_pnl.append(float(info.get("cash", 0)))

    env.close()
    network.train()

    rewards = np.array(ep_rewards)
    sharpe = float(rewards.mean() / (rewards.std() + 1e-8))

    return {
        "n_episodes":       n_episodes,
        "mean_reward":      float(rewards.mean()),
        "std_reward":       float(rewards.std()),
        "min_reward":       float(rewards.min()),
        "max_reward":       float(rewards.max()),
        "sharpe":           sharpe,
        "pct_positive":     float((rewards > 0).mean()),
        "mean_ep_length":   float(np.mean(ep_lengths)),
        "mean_final_hold":  float(np.mean(ep_final_hold)),
        "episode_rewards":  [float(r) for r in ep_rewards],
    }
