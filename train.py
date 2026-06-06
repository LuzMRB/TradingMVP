"""
train.py — Lanzador principal de entrenamiento PPO multi-env.

Uso:
    PYTHONPATH=. .venv/bin/python3.9 train.py

Al terminar guarda en results/{LABEL}:
    - {label}.json          métricas completas (train + eval)
    - {label}_training.png  curvas de entrenamiento
    - {label}_eval.png      rewards por episodio de evaluación
    - checkpoint best_model.pt en experiments/mvp_results/checkpoints/
"""

import json
import os
import sys
import warnings

# Añade la raíz del proyecto al path para que 'src' sea importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # sin display (servidor headless)
import matplotlib.pyplot as plt
import numpy as np

from src.env.spy_gym_env import SpyGymEnv
from src.env.frame_stack import FrameStackWrapper
from src.training.ppo_trainer import PPOTrainer
from src.training.evaluate import evaluate_model

# ══════════════════════════════════════════════
# CONFIGURACIÓN — edita aquí para cada experimento
# ══════════════════════════════════════════════
LABEL        = "PPO_Transformer_v3_ent002"
ARCHITECTURE = "transformer"
N_ENVS       = 10
N_FRAMES     = 10
TOTAL_STEPS  = 1_000_000
ROLLOUT_LEN  = 1024
BATCH_SIZE   = 256
UPDATE_EPOCHS = 4
LR           = 1e-4
ENTROPY_COEF = 0.02
DEVICE       = "cpu"
RESULTS_DIR  = "results"
EVAL_EPISODES = 20

# Checkpoint para continuar entrenamiento (None = entrenar desde cero)
LOAD_CHECKPOINT = None

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


def save_training_plot(trainer: PPOTrainer, path: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Training curves — {LABEL}", fontsize=13)

    # Reward por episodio
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

    # Policy loss
    ax = axes[0, 1]
    ax.plot(trainer.policy_losses, color="tomato", linewidth=0.9)
    ax.set_title("Policy Loss")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    # Value loss
    ax = axes[1, 0]
    ax.plot(trainer.value_losses, color="seagreen", linewidth=0.9)
    ax.set_title("Value Loss")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 1]
    ax.plot(trainer.entropies, color="darkorange", linewidth=0.9)
    ax.set_title("Entropy")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Training plot: {path}")


def save_eval_plot(eval_results: dict, path: str):
    pnls = eval_results["episode_pnl_pct"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Evaluation — {LABEL}  (n={eval_results['n_episodes']})", fontsize=12)

    ax = axes[0]
    ax.bar(range(len(pnls)), [p * 100 for p in pnls], color=["seagreen" if p > 0 else "tomato" for p in pnls])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    mean_pct = eval_results["mean_pnl_pct"] * 100
    ax.axhline(mean_pct, color="navy", linewidth=1.5, linestyle="-", label=f"mean={mean_pct:.2f}%")
    ax.set_title("PnL real por episodio (%)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("PnL (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    stats = {
        "mean_pnl%":  eval_results["mean_pnl_pct"] * 100,
        "std_pnl%":   eval_results["std_pnl_pct"] * 100,
        "sharpe":     eval_results["sharpe"],
        "pct_pos%":   eval_results["pct_positive"] * 100,
    }
    colors = ["seagreen" if v >= 0 else "tomato" for v in stats.values()]
    ax.barh(list(stats.keys()), list(stats.values()), color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% / ratio")
    ax.set_title("Eval metrics")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Eval plot:     {path}")


def main():
    run_dir = os.path.join(RESULTS_DIR, LABEL)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  {LABEL}")
    print(f"  {N_ENVS} envs | {TOTAL_STEPS:,} steps | arch={ARCHITECTURE}")
    print(f"{'='*55}\n")

    trainer = PPOTrainer(
        env_fn=env_fn,
        n_envs=N_ENVS,
        lr=LR,
        rollout_length=ROLLOUT_LEN,
        batch_size=BATCH_SIZE,
        update_epochs=UPDATE_EPOCHS,
        entropy_coef=ENTROPY_COEF,
        architecture=ARCHITECTURE,
        device=DEVICE,
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        n_frames=N_FRAMES,
    )

    if LOAD_CHECKPOINT:
        trainer.load_checkpoint(LOAD_CHECKPOINT)
        print(f"  Continuando desde step {trainer.total_steps:,}\n")
        trainer.total_steps = 0   # resetear contador para entrenar TOTAL_STEPS nuevos

    trainer.train(total_steps=TOTAL_STEPS, log_interval=1)

    # ── Evaluación ──────────────────────────────
    print("\nEvaluando política entrenada...")
    eval_results = evaluate_model(
        network=trainer.network,
        env_fn=env_fn,
        n_episodes=EVAL_EPISODES,
        device=DEVICE,
    )
    print(f"  Eval PnL real: {eval_results['mean_pnl_pct']:+.2%}  sharpe: {eval_results['sharpe']:.3f}  pct_pos: {eval_results['pct_positive']:.1%}")

    # ── Guardar JSON ─────────────────────────────
    output = {
        "label": LABEL,
        "config": {
            "n_envs": N_ENVS,
            "total_steps": TOTAL_STEPS,
            "rollout_length": ROLLOUT_LEN,
            "batch_size": BATCH_SIZE,
            "update_epochs": UPDATE_EPOCHS,
            "lr": LR,
            "entropy_coef": ENTROPY_COEF,
            "architecture": ARCHITECTURE,
        },
        "train": {
            "episode_rewards": [float(r) for r in trainer.episode_rewards],
            "value_losses":    trainer.value_losses,
            "policy_losses":   trainer.policy_losses,
            "entropies":       trainer.entropies,
            "total_steps":     trainer.total_steps,
            "fps_mean":        float(np.mean(trainer._fps_samples)) if trainer._fps_samples else 0.0,
        },
        "eval": eval_results,
    }

    json_path = os.path.join(run_dir, f"{LABEL}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  JSON guardado:  {json_path}")

    # ── Plots ────────────────────────────────────
    save_training_plot(trainer, os.path.join(run_dir, f"{LABEL}_training.png"))
    save_eval_plot(eval_results, os.path.join(run_dir, f"{LABEL}_eval.png"))

    print(f"\nListo. Resultados en {run_dir}/")


if __name__ == "__main__":
    main()
