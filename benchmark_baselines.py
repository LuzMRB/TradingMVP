"""
benchmark_baselines.py — Comparación emparejada de tres vías:
    Agente RL  vs  Buy & Hold  vs  MACD crossover

Para cada episodio simulado (seed fijada → reproducible):
  1. El agente RL opera en la simulación (greedy) y se registra la serie
     completa de mid-prices en cada wakeup (1 obs/minuto).
  2. B&H se calcula offline sobre esa misma serie:
         bh_pnl = (mid_close - mid_open) / mid_open
  3. MACD crossover se ejecuta en paper-trading offline sobre esa misma serie
     (ver MACDStrategy). Mismo capital inicial, sin short selling.

Las tres estrategias comparten exactamente el mismo mercado simulado,
episodio a episodio — comparación emparejada válida.

NOTA sobre seeds: las corridas previas (benchmark_bh.py) NO fijaban seeds,
por lo que aquellos episodios no son reproducibles. Este script fija
env.seed(BASE_SEED + ep) por episodio; para un mismo modelo la corrida es
determinista (política greedy + simulación seedeada). Entre modelos distintos
las series divergen tras la primera orden diferente (el agente impacta el
mercado), pero el emparejamiento intra-episodio sigue siendo válido.

Uso:
    .venv/bin/python3.9 benchmark_baselines.py --test    # 2 episodios, imprime señales MACD
    .venv/bin/python3.9 benchmark_baselines.py           # corrida completa (30 episodios)

Guarda en benchmarking_B&H/:
    - {LABEL}_vs_baselines.json
    - {LABEL}_vs_baselines.png
    - baselines_summary.json
"""

import argparse
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
from src.agents.networks import TransformerActorCritic
from src.agents.dual_network import DualDirectionalNetwork

# ══════════════════════════════════════════════
N_EPISODES = 30
BASE_SEED  = 42
N_FRAMES   = 10
DEVICE     = "cpu"
OUT_DIR    = "benchmarking_B&H"

# MACD
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9
MACD_WARMUP  = 35      # 26 + 9: sin señales en las primeras 35 observaciones
MACD_SHARES  = 10      # compra 10 acciones por entrada (~1x capital, igual que una
                       # orden del agente; 100 acciones implicaría ~10x apalancamiento)
STARTING_CASH = 1_000_000

ENV_KWARGS = dict(
    background_config="rmsc04",
    mkt_close="16:00:00",
    timestep_duration="60s",
    starting_cash=STARTING_CASH,
    order_fixed_size=10,
    inv_penalty_coef=0.0001,
    opportunity_cost_coef=0.0001,
    first_interval="00:05:00",
)

MODELS = [
    {
        "label":      "PPO_Transformer_v3",
        "checkpoint": "results/PPO_Transformer_v3_ent002/checkpoints/best_model.pt",
        "arch":       "transformer",
    },
    {
        "label":      "MAPPO_Dual_v3",
        "checkpoint": "results/MAPPO_Dual_v3/checkpoints/checkpoint_102400.pt",
        "arch":       "dual",
    },
]
# ══════════════════════════════════════════════


def env_fn():
    return FrameStackWrapper(SpyGymEnv(**ENV_KWARGS), n_frames=N_FRAMES)


def load_network(arch: str, checkpoint: str, obs_dim: int, action_dim: int):
    if arch == "transformer":
        network = TransformerActorCritic(obs_dim=obs_dim, action_dim=action_dim, n_frames=N_FRAMES)
    elif arch == "dual":
        network = DualDirectionalNetwork(obs_dim=obs_dim, action_dim=action_dim, n_frames=N_FRAMES)
    else:
        raise ValueError(f"Arquitectura desconocida: {arch}")

    ckpt = torch.load(checkpoint, map_location=DEVICE)
    network.load_state_dict(ckpt["network_state_dict"])
    network.eval()
    return network


# ════════════════════════════════════════════════════════════
# MACD
# ════════════════════════════════════════════════════════════

def ema(series: np.ndarray, span: int) -> np.ndarray:
    """EMA estándar (adjust=False), inicializada con el primer valor."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(series, dtype=np.float64)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def macd_lines(prices: np.ndarray):
    macd   = ema(prices, MACD_FAST) - ema(prices, MACD_SLOW)
    signal = ema(macd, MACD_SIGNAL)
    return macd, signal


def run_macd_paper(prices: np.ndarray, verbose: bool = False):
    """
    Paper-trading MACD crossover sobre la serie de mid-prices del episodio.

    - Cruce MACD por ENCIMA de la señal → comprar MACD_SHARES (si no hay posición)
    - Cruce por DEBAJO → vender toda la posición (si la hay)
    - Sin short selling. Warm-up: sin señales antes de la observación MACD_WARMUP.
    - Ejecución al mid-price del paso SIGUIENTE al cruce (asunción conservadora:
      la señal calculada al cierre del minuto t solo es accionable en t+1;
      sin slippage ni coste de transacción).

    Devuelve (pnl_pct, señales) donde señales = [(t_ejecución, tipo, precio), ...]
    """
    macd, signal = macd_lines(prices)
    cash = float(STARTING_CASH)
    position = 0
    signals = []

    for t in range(1, len(prices) - 1):
        if t < MACD_WARMUP:
            continue
        crossed_up   = macd[t - 1] <= signal[t - 1] and macd[t] > signal[t]
        crossed_down = macd[t - 1] >= signal[t - 1] and macd[t] < signal[t]

        exec_price = prices[t + 1]
        if crossed_up and position == 0:
            position = MACD_SHARES
            cash -= position * exec_price
            signals.append((t + 1, "BUY", float(exec_price)))
            if verbose:
                print(f"      MACD señal: t={t} cruce ALCISTA  → BUY  {MACD_SHARES} @ {exec_price:.2f} (ejecutado en t={t+1})")
        elif crossed_down and position > 0:
            cash += position * exec_price
            signals.append((t + 1, "SELL", float(exec_price)))
            if verbose:
                print(f"      MACD señal: t={t} cruce BAJISTA  → SELL {position} @ {exec_price:.2f} (ejecutado en t={t+1})")
            position = 0

    final_value = cash + position * prices[-1]
    pnl_pct = (final_value - STARTING_CASH) / STARTING_CASH
    return float(pnl_pct), signals


# ════════════════════════════════════════════════════════════
# Evaluación emparejada
# ════════════════════════════════════════════════════════════

def run_paired_eval(label: str, network, n_episodes: int, verbose_macd: bool = False) -> dict:
    dev = torch.device(DEVICE)
    env = env_fn()

    agent_pnl, bh_pnl, macd_pnl = [], [], []
    all_mid_series = []

    for ep in range(n_episodes):
        env.seed(BASE_SEED + ep)
        obs = env.reset()
        mid_series = [env.current_mid_price]
        done = False
        info = {}

        while not done:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                logits, _ = network.forward(obs_t)
                action = logits.argmax(dim=-1).item()
            obs, _, done, info = env.step(action)
            mid_series.append(float(info.get("mid_price", mid_series[-1])))

        prices = np.array(mid_series, dtype=np.float64)

        holdings = int(info.get("holdings", 0))
        cash     = float(info.get("cash", STARTING_CASH))
        a_pnl = (cash + holdings * prices[-1] - STARTING_CASH) / STARTING_CASH
        b_pnl = (prices[-1] - prices[0]) / prices[0]
        m_pnl, signals = run_macd_paper(prices, verbose=verbose_macd)

        agent_pnl.append(float(a_pnl))
        bh_pnl.append(float(b_pnl))
        macd_pnl.append(float(m_pnl))
        all_mid_series.append([float(p) for p in prices])

        print(
            f"  [{label}] ep {ep+1:>2d}/{n_episodes} (seed {BASE_SEED+ep}) | "
            f"agente: {a_pnl:>+7.2%} | B&H: {b_pnl:>+7.2%} | "
            f"MACD: {m_pnl:>+7.2%} ({len(signals)} señales) | "
            f"mid: {prices[0]:>9.2f} -> {prices[-1]:>9.2f} | steps: {len(prices)-1}"
        )

    env.close()

    agent_arr = np.array(agent_pnl)
    bh_arr    = np.array(bh_pnl)
    macd_arr  = np.array(macd_pnl)
    alpha_bh   = agent_arr - bh_arr
    alpha_macd = agent_arr - macd_arr

    def stats(arr):
        return {
            "mean":    float(arr.mean()),
            "std":     float(arr.std()),
            "sharpe":  float(arr.mean() / (arr.std() + 1e-8)),
            "pct_pos": float((arr > 0).mean()),
            "min":     float(arr.min()),
            "max":     float(arr.max()),
        }

    return {
        "label":        label,
        "n_episodes":   n_episodes,
        "base_seed":    BASE_SEED,
        "agent":        stats(agent_arr),
        "buy_and_hold": stats(bh_arr),
        "macd":         stats(macd_arr),
        "alpha":        stats(alpha_bh),
        "alpha_vs_macd": stats(alpha_macd),
        "pct_episodes_agent_beats_bh":   float((alpha_bh > 0).mean()),
        "pct_episodes_agent_beats_macd": float((alpha_macd > 0).mean()),
        "episode_agent_pnl": agent_pnl,
        "episode_bh_pnl":    bh_pnl,
        "episode_macd_pnl":  macd_pnl,
        "episode_alpha":     [float(a) for a in alpha_bh],
        "episode_alpha_vs_macd": [float(a) for a in alpha_macd],
        "episode_mid_series": all_mid_series,
    }


# ════════════════════════════════════════════════════════════
# Gráfico
# ════════════════════════════════════════════════════════════

def save_plot(result: dict, path: str):
    label = result["label"]
    agent = np.array(result["episode_agent_pnl"]) * 100
    bh    = np.array(result["episode_bh_pnl"]) * 100
    macd  = np.array(result["episode_macd_pnl"]) * 100
    n = len(agent)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle(f"{label} vs Buy & Hold vs MACD — comparación emparejada (n={n} episodios, mismo mercado simulado)", fontsize=12)

    ax = axes[0]
    width = 0.27
    ax.bar(x - width, agent, width, label=f"Agente (mean={agent.mean():+.2f}%)", color="navy", alpha=0.85)
    ax.bar(x,         bh,    width, label=f"B&H (mean={bh.mean():+.2f}%)",       color="gray", alpha=0.85)
    ax.bar(x + width, macd,  width, label=f"MACD (mean={macd.mean():+.2f}%)",    color="darkorange", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("PnL por episodio")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("PnL (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    alpha_bh   = agent - bh
    alpha_macd = agent - macd
    ax.plot(x, alpha_bh,   marker="o", markersize=3, linewidth=1, color="gray",       label=f"alpha vs B&H (mean={alpha_bh.mean():+.2f}%)")
    ax.plot(x, alpha_macd, marker="s", markersize=3, linewidth=1, color="darkorange", label=f"alpha vs MACD (mean={alpha_macd.mean():+.2f}%)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    pct_bh   = result["pct_episodes_agent_beats_bh"] * 100
    pct_macd = result["pct_episodes_agent_beats_macd"] * 100
    ax.set_title(f"Alpha del agente  |  gana a B&H: {pct_bh:.0f}%  ·  gana a MACD: {pct_macd:.0f}%")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Alpha (puntos %)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    cats = ["Agente", "B&H", "MACD"]
    means   = [result["agent"]["mean"] * 100, result["buy_and_hold"]["mean"] * 100, result["macd"]["mean"] * 100]
    sharpes = [result["agent"]["sharpe"],     result["buy_and_hold"]["sharpe"],     result["macd"]["sharpe"]]
    xpos = np.arange(len(cats))
    w = 0.35
    ax.bar(xpos - w/2, means,   w, label="Mean PnL (%)", color="steelblue",  alpha=0.85)
    ax.bar(xpos + w/2, sharpes, w, label="Sharpe",       color="darkorange", alpha=0.85)
    ax.set_xticks(xpos)
    ax.set_xticklabels(cats)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Resumen de métricas")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  Plot: {path}")


# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Modo verificación: 2 episodios solo con v3, imprime señales MACD")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    env_probe = env_fn()
    obs_dim    = env_probe.observation_space.shape[0]
    action_dim = env_probe.action_space.n
    env_probe.close()

    models = MODELS[:1] if args.test else MODELS
    n_eps  = 2 if args.test else N_EPISODES

    summary = {}

    for cfg in models:
        label = cfg["label"]
        print(f"\n{'='*65}")
        print(f"  {label}  vs  B&H  vs  MACD   (n={n_eps} episodios emparejados, base_seed={BASE_SEED})")
        print(f"{'='*65}\n")

        network = load_network(cfg["arch"], cfg["checkpoint"], obs_dim, action_dim)
        result = run_paired_eval(label, network, n_eps, verbose_macd=args.test)

        if args.test:
            # Sanity check: B&H desde la serie registrada == fórmula apertura/cierre
            for i, series in enumerate(result["episode_mid_series"]):
                recomputed = (series[-1] - series[0]) / series[0]
                stored = result["episode_bh_pnl"][i]
                ok = abs(recomputed - stored) < 1e-12
                print(f"\n  Sanity B&H ep {i+1}: serie→{recomputed:+.4%} vs guardado→{stored:+.4%}  [{'OK' if ok else 'FALLO'}]")
            print("\n  Modo test completado — revisa las señales MACD impresas arriba.")
            return

        print(f"\n{'─'*65}")
        print(f"  Agente -> mean: {result['agent']['mean']:>+.3%} | sharpe: {result['agent']['sharpe']:.3f} | pct_pos: {result['agent']['pct_pos']:.1%}")
        print(f"  B&H    -> mean: {result['buy_and_hold']['mean']:>+.3%} | sharpe: {result['buy_and_hold']['sharpe']:.3f} | pct_pos: {result['buy_and_hold']['pct_pos']:.1%}")
        print(f"  MACD   -> mean: {result['macd']['mean']:>+.3%} | sharpe: {result['macd']['sharpe']:.3f} | pct_pos: {result['macd']['pct_pos']:.1%}")
        print(f"  Alpha vs B&H:  {result['alpha']['mean']:>+.3%} (gana en {result['pct_episodes_agent_beats_bh']:.1%})")
        print(f"  Alpha vs MACD: {result['alpha_vs_macd']['mean']:>+.3%} (gana en {result['pct_episodes_agent_beats_macd']:.1%})")
        print(f"{'─'*65}\n")

        json_path = os.path.join(OUT_DIR, f"{label}_vs_baselines.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  JSON: {json_path}")

        save_plot(result, os.path.join(OUT_DIR, f"{label}_vs_baselines.png"))

        summary[label] = {
            "agent_mean_pnl":  result["agent"]["mean"],
            "agent_sharpe":    result["agent"]["sharpe"],
            "bh_mean_pnl":     result["buy_and_hold"]["mean"],
            "bh_sharpe":       result["buy_and_hold"]["sharpe"],
            "macd_mean_pnl":   result["macd"]["mean"],
            "macd_sharpe":     result["macd"]["sharpe"],
            "alpha_vs_bh":     result["alpha"]["mean"],
            "alpha_vs_macd":   result["alpha_vs_macd"]["mean"],
            "pct_beats_bh":    result["pct_episodes_agent_beats_bh"],
            "pct_beats_macd":  result["pct_episodes_agent_beats_macd"],
        }

    summary_path = os.path.join(OUT_DIR, "baselines_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*90}")
    print(f"  RESUMEN FINAL — Agente vs B&H vs MACD (n={N_EPISODES} episodios c/u, base_seed={BASE_SEED})")
    print(f"  {'-'*86}")
    print(f"  {'Modelo':<22} {'Agente':>9} {'B&H':>9} {'MACD':>9} {'α vs B&H':>10} {'α vs MACD':>10} {'>B&H':>7} {'>MACD':>7}")
    for label, m in summary.items():
        print(f"  {label:<22} {m['agent_mean_pnl']:>+8.2%} {m['bh_mean_pnl']:>+8.2%} {m['macd_mean_pnl']:>+8.2%}"
              f" {m['alpha_vs_bh']:>+9.2%} {m['alpha_vs_macd']:>+9.2%}"
              f" {m['pct_beats_bh']:>6.1%} {m['pct_beats_macd']:>6.1%}")
    print(f"{'='*90}\n")
    print(f"  Resumen: {summary_path}")
    print(f"\nListo. Resultados en {OUT_DIR}/")


if __name__ == "__main__":
    main()
