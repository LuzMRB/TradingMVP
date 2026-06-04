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
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # sin display (servidor headless)
import matplotlib.pyplot as plt
import numpy as np

from src.env.spy_gym_env import SpyGymEnv
from src.training.ppo_trainer import PPOTrainer
from src.training.evaluate import evaluate_model

# ══════════════════════════════════════════════
# CONFIGURACIÓN — edita aquí para cada experimento
# ══════════════════════════════════════════════
LABEL        = "PPO_Transformer_12envs"
ARCHITECTURE = "transformer"   # "mlp" o "transformer"
N_ENVS       = 10              # núcleos libres
TOTAL_STEPS  = 500_000
ROLLOUT_LEN  = 1024
BATCH_SIZE   = 256
UPDATE_EPOCHS = 4
LR           = 1e-4
ENTROPY_COEF = 0.05
DEVICE       = "cpu"
RESULTS_DIR  = "results"
EVAL_EPISODES = 20

ENV_KWARGS = dict(
    background_config="rmsc04",
    mkt_close="16:00:00",
    timestep_duration="60s",
    starting_cash=1_000_000,
    order_fixed_size=10,
    inv_penalty_coef=0.001,
    first_interval="00:05:00",
)
# ══════════════════════════════════════════════


def env_fn():
    return SpyGymEnv(**ENV_KWARGS)


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
    rewards = eval_results["episode_rewards"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Evaluation — {LABEL}  (n={eval_results['n_episodes']})", fontsize=12)

    ax = axes[0]
    ax.bar(range(len(rewards)), rewards, color=["seagreen" if r > 0 else "tomato" for r in rewards])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(eval_results["mean_reward"], color="navy", linewidth=1.5, linestyle="-", label=f"mean={eval_results['mean_reward']:.4f}")
    ax.set_title("Episode Rewards (eval)")
    ax.set_xlabel("Episodio")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    stats = {
        "mean":     eval_results["mean_reward"],
        "std":      eval_results["std_reward"],
        "sharpe":   eval_results["sharpe"],
        "pct_pos":  eval_results["pct_positive"],
    }
    ax.barh(list(stats.keys()), list(stats.values()), color="steelblue")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Eval metrics")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Eval plot:     {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

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
    )

    trainer.train(total_steps=TOTAL_STEPS, log_interval=10)

    # ── Evaluación ──────────────────────────────
    print("\nEvaluando política entrenada...")
    eval_results = evaluate_model(
        network=trainer.network,
        env_fn=env_fn,
        n_episodes=EVAL_EPISODES,
        device=DEVICE,
    )
    print(f"  Eval mean_reward: {eval_results['mean_reward']:.4f}  sharpe: {eval_results['sharpe']:.3f}  pct_pos: {eval_results['pct_positive']:.1%}")

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

    json_path = os.path.join(RESULTS_DIR, f"{LABEL}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  JSON guardado:  {json_path}")

    # ── Plots ────────────────────────────────────
    save_training_plot(trainer, os.path.join(RESULTS_DIR, f"{LABEL}_training.png"))
    save_eval_plot(eval_results, os.path.join(RESULTS_DIR, f"{LABEL}_eval.png"))

    print(f"\nListo. Resultados en {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
