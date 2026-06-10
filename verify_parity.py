"""
verify_parity.py — Verificación de paridad de evaluación.

A) Paridad cruzada: el evaluate_model ORIGINAL (el de la eval de 100 episodios)
   sobre un env seedeado con 42 debe reproducir exactamente el episodio 1 de
   benchmark_baselines (agente -0.888%).
B) Control sin seed: el bucle de benchmark_baselines SIN env.seed() durante
   8 episodios — ¿se parece a la distribución de la eval de 100 (+0.31%)
   o a la de la corrida seedeada (-1.26%)?
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import torch

from benchmark_baselines import env_fn, load_network
from src.training.evaluate import evaluate_model

CHECKPOINT = "results/PPO_Transformer_v3_ent002/checkpoints/best_model.pt"

env_probe = env_fn()
obs_dim, action_dim = env_probe.observation_space.shape[0], env_probe.action_space.n
env_probe.close()

network = load_network("transformer", CHECKPOINT, obs_dim, action_dim)

# ── A) evaluate_model original sobre env con seed 42 ─────────
print("=" * 60)
print("A) evaluate_model ORIGINAL con env seedeado (seed 42, 1 ep)")
print("   Esperado si hay paridad: PnL agente = -0.8880%")
print("=" * 60)

def seeded_env_fn():
    e = env_fn()
    e.seed(42)
    return e

res = evaluate_model(network=network, env_fn=seeded_env_fn, n_episodes=1, device="cpu")
print(f"\n   Resultado: {res['episode_pnl_pct'][0]:+.4%}")
match = abs(res["episode_pnl_pct"][0] - (-0.008880)) < 1e-4
print(f"   Paridad: {'OK — mismo episodio, mismo resultado' if match else 'FALLO — los scripts difieren'}")

# ── B) bucle propio SIN seed, 8 episodios ────────────────────
print()
print("=" * 60)
print("B) Bucle de benchmark SIN seed (8 episodios)")
print("   eval-100 dio mean +0.31%; corrida seedeada dio -1.26%")
print("=" * 60)

env = env_fn()
pnls = []
for ep in range(8):
    obs = env.reset()       # SIN env.seed() — como eval_final / benchmark_bh
    done = False
    info = {}
    while not done:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            logits, _ = network.forward(obs_t)
            action = logits.argmax(dim=-1).item()
        obs, _, done, info = env.step(action)
    holdings = int(info.get("holdings", 0))
    cash = float(info.get("cash", 1_000_000))
    mid = float(info.get("mid_price", 100_000.0))
    pnl = (cash + holdings * mid - 1_000_000) / 1_000_000
    pnls.append(pnl)
    print(f"   ep {ep+1}/8 (sin seed): {pnl:+.3%}")
env.close()

arr = np.array(pnls)
print(f"\n   mean: {arr.mean():+.3%} | std: {arr.std():.3%} | >0: {(arr>0).mean():.0%}")
print("\nListo.")
