"""
eval_mappo.py — Evaluación final standalone del modelo MAPPO Dual.
Guarda en eval_finales/{LABEL}/
"""

import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.env.spy_gym_env import SpyGymEnv
from src.env.frame_stack import FrameStackWrapper
from src.agents.dual_network import DualDirectionalNetwork
from src.training.evaluate import evaluate_model

# ══════════════════════════════════════════════
LABEL      = "MAPPO_Dual_v3_ck102400"
CHECKPOINT = "results/MAPPO_Dual_v3/checkpoints/checkpoint_102400.pt"
N_FRAMES   = 10
N_EPISODES = 100
DEVICE     = "cpu"
EVAL_DIR   = "eval_finales"

ENV_KWARGS = dict(
    background_config="rmsc04",
    mkt_close="16:00:00",
    timestep_duration="60s",
    starting_cash=1_000_000,
    order_fixed_size=10,
    inv_penalty_coef=0.0001,
    opportunity_cost_coef=0.0001,
    first_interval="00:05:00",
)
# ══════════════════════════════════════════════


def env_fn():
    return FrameStackWrapper(SpyGymEnv(**ENV_KWARGS), n_frames=N_FRAMES)


def main():
    run_dir = os.path.join(EVAL_DIR, LABEL)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  EVAL: {LABEL}")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  Episodios:  {N_EPISODES}")
    print(f"{'='*55}\n")

    env = env_fn()
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    network = DualDirectionalNetwork(obs_dim=obs_dim, action_dim=action_dim, n_frames=N_FRAMES)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    network.load_state_dict(ckpt["network_state_dict"])
    print(f"  Modelo cargado (step {ckpt.get('total_steps', '?'):,})\n")

    eval_results = evaluate_model(
        network=network, env_fn=env_fn, n_episodes=N_EPISODES, device=DEVICE,
    )

    print(f"\n{'─'*55}")
    print(f"  Mean PnL:  {eval_results['mean_pnl_pct']:>+.3%}")
    print(f"  Std PnL:   {eval_results['std_pnl_pct']:>.3%}")
    print(f"  Sharpe:    {eval_results['sharpe']:>.3f}")
    print(f"  Pct pos:   {eval_results['pct_positive']:>.1%}")
    print(f"  Min/Max:   {eval_results['min_pnl_pct']:>+.2%} / {eval_results['max_pnl_pct']:>+.2%}")
    print(f"{'─'*55}\n")

    output = {
        "label": LABEL, "checkpoint": CHECKPOINT,
        "config": {"n_frames": N_FRAMES, "n_episodes": N_EPISODES},
        "eval": eval_results,
    }
    json_path = os.path.join(run_dir, f"{LABEL}_eval_final.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  JSON: {json_path}")

    # Plot
    pnls = eval_results["episode_pnl_pct"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Eval final — {LABEL}  (n={N_EPISODES})", fontsize=12)

    ax = axes[0]
    ax.bar(range(len(pnls)), [p * 100 for p in pnls],
           color=["seagreen" if p > 0 else "tomato" for p in pnls], alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    mean_pct = eval_results["mean_pnl_pct"] * 100
    ax.axhline(mean_pct, color="navy", linewidth=1.8, label=f"mean={mean_pct:.2f}%")
    ax.set_title("PnL real por episodio (%)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("PnL (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    stats = {
        "mean_pnl%": mean_pct,
        "std_pnl%":  eval_results["std_pnl_pct"] * 100,
        "sharpe":    eval_results["sharpe"],
        "pct_pos%":  eval_results["pct_positive"] * 100,
        "min_pnl%":  eval_results["min_pnl_pct"] * 100,
        "max_pnl%":  eval_results["max_pnl_pct"] * 100,
    }
    colors = ["seagreen" if v >= 0 else "tomato" for v in stats.values()]
    bars = ax.barh(list(stats.keys()), list(stats.values()), color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, stats.values()):
        ax.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)
    ax.set_title("Métricas de evaluación")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, f"{LABEL}_eval_final.png")
    plt.savefig(plot_path, dpi=130)
    plt.close()
    print(f"  Plot: {plot_path}")
    print(f"\nListo. Resultados en {run_dir}/")


if __name__ == "__main__":
    main()
