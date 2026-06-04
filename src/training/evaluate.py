"""
evaluate.py — Evaluación determinista del agente entrenado.

Corre n_episodes completos con la política greedy (sin exploración)
y devuelve un dict de métricas listo para guardar en JSON.

Métricas clave:
    - episode_rewards:  reward total (incluye penalizaciones)
    - pnl_pct:          PnL real = (cash + holdings*precio - starting_cash) / starting_cash
"""

import numpy as np
import torch
from typing import Callable, Dict


def evaluate_model(network, env_fn: Callable, n_episodes: int = 20, device: str = "cpu") -> Dict:
    dev = torch.device(device)
    network.eval()

    ep_rewards   = []
    ep_lengths   = []
    ep_final_hold = []
    ep_pnl_pct   = []  # PnL real sin penalizaciones

    use_gru = hasattr(network, 'gru')
    env = env_fn()
    starting_cash = env.starting_cash

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        h = torch.zeros(1, 1, network.d_model, device=dev) if use_gru else None

        while not done:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                if use_gru:
                    logits, _, h = network.forward(obs_t, h)
                else:
                    logits, _ = network.forward(obs_t)
                action = logits.argmax(dim=-1).item()   # greedy

            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        holdings  = int(info.get("holdings", 0))
        cash      = float(info.get("cash", starting_cash))
        mid_price = float(info.get("mid_price", env.price_norm))

        # PnL real: cuánto dinero ganó/perdió sin contar penalizaciones
        final_value = cash + holdings * mid_price
        pnl_pct = (final_value - starting_cash) / starting_cash

        ep_rewards.append(total_reward)
        ep_lengths.append(steps)
        ep_final_hold.append(holdings)
        ep_pnl_pct.append(float(pnl_pct))

        print(
            f"  Eval ep {ep+1:>2d}/{n_episodes} | "
            f"reward: {total_reward:>8.4f} | "
            f"PnL real: {pnl_pct:>+.2%} | "
            f"holdings: {holdings} | steps: {steps}"
        )

    env.close()
    network.train()

    rewards = np.array(ep_rewards)
    pnls    = np.array(ep_pnl_pct)
    sharpe  = float(pnls.mean() / (pnls.std() + 1e-8))

    return {
        "n_episodes":       n_episodes,
        "mean_reward":      float(rewards.mean()),
        "std_reward":       float(rewards.std()),
        "mean_pnl_pct":     float(pnls.mean()),
        "std_pnl_pct":      float(pnls.std()),
        "min_pnl_pct":      float(pnls.min()),
        "max_pnl_pct":      float(pnls.max()),
        "sharpe":           sharpe,
        "pct_positive":     float((pnls > 0).mean()),
        "mean_ep_length":   float(np.mean(ep_lengths)),
        "mean_final_hold":  float(np.mean(ep_final_hold)),
        "episode_rewards":  [float(r) for r in ep_rewards],
        "episode_pnl_pct":  [float(p) for p in ep_pnl_pct],
    }
