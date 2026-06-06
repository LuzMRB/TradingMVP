"""
train_dual.py — Lanzador para arquitectura Dual Bull/Bear + Arbitrador.

Uso:
    .venv/bin/python3.9 train_dual.py

Guarda en results/{LABEL}:
    - {label}.json
    - {label}_training.png
    - {label}_eval.png
    - checkpoints/
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

from src.env.spy_gym_env import SpyGymEnv
from src.env.frame_stack import FrameStackWrapper
from src.training.dual_trainer import DualTrainer
from src.training.evaluate import evaluate_model

# ══════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════
LABEL         = "MAPPO_Dual_v3"
N_ENVS        = 10
N_FRAMES      = 10
TOTAL_STEPS   = 1_000_000
ROLLOUT_LEN   = 1024
BATCH_SIZE    = 256
UPDATE_EPOCHS = 4
LR            = 1e-4
ENTROPY_COEF  = 0.008
ALPHA         = 1.0        # peso loss Bull/Bear (encoder protegido con detach)
K_DELAY       = 5          # steps de delay para reward direccional
DEVICE        = "cpu"
RESULTS_DIR   = "results"
EVAL_EPISODES = 20

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


def save_training_plot(trainer: DualTrainer, path: str):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"Training curves — {LABEL}", fontsize=13)

    ax = axes[0, 0]
    rewards = trainer.episode_rewards
    ax.plot(rewards, alpha=0.4, color="steelblue", linewidth=0.8, label="episodio")
    if len(rewards) >= 10:
        window = max(len(rewards) // 20, 10)
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), ma, color="navy", linewidth=1.5, label=f"MA-{window}")
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episodio")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(trainer.policy_losses, color="tomato", linewidth=0.9)
    ax.set_title("Policy Loss (Arbitrador)")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(trainer.value_losses, color="seagreen", linewidth=0.9)
    ax.set_title("Value Loss (Arbitrador)")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(trainer.entropies, color="darkorange", linewidth=0.9)
    ax.set_title("Entropy")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(trainer.bull_losses, color="green", linewidth=0.9, label="Bull BCE")
    ax.plot(trainer.bear_losses, color="red",   linewidth=0.9, label="Bear BCE")
    ax.set_title("Directional Losses (Bull / Bear)")
    ax.set_xlabel("Update")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Training plot: {path}")


def save_eval_plot(eval_results: dict, path: str):
    pnls = eval_results["episode_pnl_pct"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Evaluation — {LABEL}  (n={eval_results['n_episodes']})", fontsize=12)

    ax = axes[0]
    ax.bar(range(len(pnls)), [p * 100 for p in pnls],
           color=["seagreen" if p > 0 else "tomato" for p in pnls])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    mean_pct = eval_results["mean_pnl_pct"] * 100
    ax.axhline(mean_pct, color="navy", linewidth=1.5, label=f"mean={mean_pct:.2f}%")
    ax.set_title("PnL real por episodio (%)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("PnL (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    stats = {
        "mean_pnl%": eval_results["mean_pnl_pct"] * 100,
        "std_pnl%":  eval_results["std_pnl_pct"]  * 100,
        "sharpe":    eval_results["sharpe"],
        "pct_pos%":  eval_results["pct_positive"]  * 100,
    }
    colors = ["seagreen" if v >= 0 else "tomato" for v in stats.values()]
    ax.barh(list(stats.keys()), list(stats.values()), color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Eval metrics")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Eval plot: {path}")


def main():
    run_dir = os.path.join(RESULTS_DIR, LABEL)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  {LABEL}")
    print(f"  {N_ENVS} envs | {TOTAL_STEPS:,} steps | k_delay={K_DELAY} | alpha={ALPHA}")
    print(f"{'='*55}\n")

    trainer = DualTrainer(
        env_fn=env_fn,
        n_envs=N_ENVS,
        lr=LR,
        rollout_length=ROLLOUT_LEN,
        batch_size=BATCH_SIZE,
        update_epochs=UPDATE_EPOCHS,
        entropy_coef=ENTROPY_COEF,
        alpha=ALPHA,
        k_delay=K_DELAY,
        device=DEVICE,
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        n_frames=N_FRAMES,
    )

    trainer.train(total_steps=TOTAL_STEPS, log_interval=1)

    print("\nEvaluando política entrenada...")
    eval_results = evaluate_model(
        network=trainer.network,
        env_fn=env_fn,
        n_episodes=EVAL_EPISODES,
        device=DEVICE,
    )
    print(f"  Eval PnL: {eval_results['mean_pnl_pct']:+.2%}  sharpe: {eval_results['sharpe']:.3f}  pct_pos: {eval_results['pct_positive']:.1%}")

    output = {
        "label":  LABEL,
        "config": {
            "n_envs": N_ENVS, "total_steps": TOTAL_STEPS,
            "rollout_length": ROLLOUT_LEN, "batch_size": BATCH_SIZE,
            "update_epochs": UPDATE_EPOCHS, "lr": LR,
            "entropy_coef": ENTROPY_COEF, "alpha": ALPHA, "k_delay": K_DELAY,
        },
        "train": {
            "episode_rewards": [float(r) for r in trainer.episode_rewards],
            "value_losses":    trainer.value_losses,
            "policy_losses":   trainer.policy_losses,
            "entropies":       trainer.entropies,
            "bull_losses":     trainer.bull_losses,
            "bear_losses":     trainer.bear_losses,
            "total_steps":     trainer.total_steps,
            "fps_mean":        float(np.mean(trainer._fps_samples)) if trainer._fps_samples else 0.0,
        },
        "eval": eval_results,
    }

    json_path = os.path.join(run_dir, f"{LABEL}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  JSON guardado: {json_path}")

    save_training_plot(trainer, os.path.join(run_dir, f"{LABEL}_training.png"))
    save_eval_plot(eval_results, os.path.join(run_dir, f"{LABEL}_eval.png"))

    print(f"\nListo. Resultados en {run_dir}/")


if __name__ == "__main__":
    main()
