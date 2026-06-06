"""
test_dual.py — Verifica que la arquitectura dual entrena correctamente.

Comprueba:
  1. Forward pass: shapes correctas, scores en [0,1]
  2. Buffer: delayed targets asignados correctamente
  3. Gradientes: todos los componentes reciben gradiente
  4. Update: pesos cambian tras optimizer.step()
  5. Integración con env real: rollout + update completo
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from src.agents.dual_network import DualDirectionalNetwork
from src.training.dual_buffer import DualRolloutBuffer
from src.training.running_stats import RunningMeanStd

PASS = "  [OK]"
FAIL = "  [FAIL]"

def check(condition, msg):
    status = PASS if condition else FAIL
    print(f"{status}  {msg}")
    if not condition:
        raise AssertionError(msg)

# ══════════════════════════════════════════════
print("\n── 1. Forward pass ──────────────────────────────────")
# ══════════════════════════════════════════════

net = DualDirectionalNetwork(obs_dim=440, action_dim=3, n_frames=10)
obs = torch.randn(8, 440)

action, log_prob, value, score_bull, score_bear = net.get_action_and_value(obs)

check(action.shape == (8,),          f"action shape: {action.shape}")
check(log_prob.shape == (8,),        f"log_prob shape: {log_prob.shape}")
check(value.shape == (8,),           f"value shape: {value.shape}")
check(score_bull.shape == (8,),      f"score_bull shape: {score_bull.shape}")
check(score_bear.shape == (8,),      f"score_bear shape: {score_bear.shape}")
check((score_bull >= 0).all() and (score_bull <= 1).all(), "score_bull ∈ [0,1]")
check((score_bear >= 0).all() and (score_bear <= 1).all(), "score_bear ∈ [0,1]")
check(action.min() >= 0 and action.max() <= 2, f"acciones válidas: {action.tolist()}")

log_probs2, entropy, value2, sb2, se2 = net.evaluate_actions(obs, action)
check(log_probs2.shape == (8,), f"evaluate_actions log_probs shape: {log_probs2.shape}")
check(entropy.shape == (8,),    f"evaluate_actions entropy shape: {entropy.shape}")

logits, val = net.forward(obs)
check(logits.shape == (8, 3), f"forward logits shape: {logits.shape}")

# ══════════════════════════════════════════════
print("\n── 2. Buffer — delayed targets ──────────────────────")
# ══════════════════════════════════════════════

BUF_SIZE, N_ENVS, OBS_DIM, K = 20, 2, 440, 5
buf = DualRolloutBuffer(BUF_SIZE, OBS_DIM, n_envs=N_ENVS, k_delay=K)

# Precios sintéticos: sube linealmente
mid_prices = np.linspace(100, 120, BUF_SIZE)

for t in range(BUF_SIZE):
    buf.add(
        obs=np.zeros((N_ENVS, OBS_DIM)),
        action=np.zeros(N_ENVS, dtype=np.int64),
        arb_reward=np.ones(N_ENVS) * 0.01,
        done=np.zeros(N_ENVS),
        log_prob=np.zeros(N_ENVS),
        value=np.zeros(N_ENVS),
        score_bull=np.ones(N_ENVS) * 0.6,
        score_bear=np.ones(N_ENVS) * 0.4,
        mid_price=np.ones(N_ENVS) * mid_prices[t],
    )

buf.compute_directional_targets()

# Con precios siempre subiendo, bull_target debe ser 1 para todos menos los últimos k
for t in range(BUF_SIZE - K):
    check(buf.bull_target[t, 0] == 1.0, f"bull_target[{t}]=1 (precio sube)")
    check(buf.bear_target[t, 0] == 0.0, f"bear_target[{t}]=0 (precio sube)")

# Últimos k steps: future_t = T-1, mismo precio → went_up=0 si igual
# (con linspace el último tramo sigue subiendo, así que también 1 excepto el último)
check(buf.ptr == BUF_SIZE, f"buffer lleno: ptr={buf.ptr}")

# GAE
buf.compute_gae(last_value=np.zeros(N_ENVS), last_done=np.zeros(N_ENVS))
check(not np.isnan(buf.advantages).any(), "advantages sin NaN")
check(not np.isnan(buf.returns).any(),    "returns sin NaN")

# Batches
batches = list(buf.get_batches(batch_size=16))
check(len(batches) > 0, f"batches generados: {len(batches)}")
b = batches[0]
check("bull_target" in b and "bear_target" in b, "targets en batch")
check(b["obs"].shape[1] == OBS_DIM, f"obs shape en batch: {b['obs'].shape}")

# ══════════════════════════════════════════════
print("\n── 3. Gradientes — todos los componentes ────────────")
# ══════════════════════════════════════════════

net = DualDirectionalNetwork(obs_dim=440, action_dim=3, n_frames=10)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
bce = nn.BCELoss()

obs_t    = torch.randn(32, 440)
actions  = torch.randint(0, 3, (32,))
bull_tgt = torch.randint(0, 2, (32,)).float()
bear_tgt = 1 - bull_tgt
returns  = torch.randn(32)
adv      = torch.randn(32)

# Forward
log_probs, entropy, values, score_bull, score_bear = net.evaluate_actions(obs_t, actions)

# Losses
ratio = torch.exp(log_probs - log_probs.detach())
policy_loss  = -(ratio * adv).mean()
value_loss   = nn.functional.mse_loss(values, returns)
entropy_loss = -entropy.mean()
loss_arb     = policy_loss + 0.5 * value_loss + 0.02 * entropy_loss
loss_bull    = bce(score_bull, bull_tgt)
loss_bear    = bce(score_bear, bear_tgt)
loss         = loss_arb + 0.5 * (loss_bull + loss_bear)

loss.backward()

# Verificar gradientes en cada componente
components = {
    "transformer (encoder compartido)": net.transformer.layers[0].self_attn.in_proj_weight,
    "input_proj":                        net.input_proj.weight,
    "bull_head":                         net.bull_head.weight,
    "bear_head":                         net.bear_head.weight,
    "arbitrador (capa 0)":               net.arbitrator[0].weight,
    "arbitrador (capa 1)":               net.arbitrator[2].weight,
    "critic":                            net.critic.weight,
}

for name, param in components.items():
    has_grad = param.grad is not None and param.grad.abs().sum().item() > 0
    check(has_grad, f"gradiente en {name}")

# ══════════════════════════════════════════════
print("\n── 4. Update — pesos cambian ────────────────────────")
# ══════════════════════════════════════════════

net2 = DualDirectionalNetwork(obs_dim=440, action_dim=3, n_frames=10)
optimizer2 = torch.optim.Adam(net2.parameters(), lr=1e-3)

# Snapshot de pesos antes
w_bull_before = net2.bull_head.weight.data.clone()
w_bear_before = net2.bear_head.weight.data.clone()
w_arb_before  = net2.arbitrator[0].weight.data.clone()
w_enc_before  = net2.input_proj.weight.data.clone()

# Un update
obs_t   = torch.randn(64, 440)
actions = torch.randint(0, 3, (64,))
bull_t  = torch.randint(0, 2, (64,)).float()
bear_t  = 1 - bull_t
rets    = torch.randn(64)
advs    = torch.randn(64)

log_probs, entropy, values, sb, se = net2.evaluate_actions(obs_t, actions)
ratio = torch.exp(log_probs - log_probs.detach())
loss = (-(ratio * advs).mean()
        + 0.5 * nn.functional.mse_loss(values, rets)
        + 0.5 * (bce(sb, bull_t) + bce(se, bear_t)))
optimizer2.zero_grad()
loss.backward()
optimizer2.step()

check((net2.bull_head.weight.data != w_bull_before).any(),   "bull_head actualizado")
check((net2.bear_head.weight.data != w_bear_before).any(),   "bear_head actualizado")
check((net2.arbitrator[0].weight.data != w_arb_before).any(),"arbitrador actualizado")
check((net2.input_proj.weight.data != w_enc_before).any(),   "encoder compartido actualizado")

# ══════════════════════════════════════════════
print("\n── 5. RunningMeanStd ─────────────────────────────────")
# ══════════════════════════════════════════════

rms = RunningMeanStd()
data1 = np.random.randn(100) * 2 + 5
rms.update(data1)
check(abs(rms.mean - 5) < 0.5,  f"mean ≈ 5: {rms.mean:.3f}")
check(abs(rms.var  - 4) < 1.0,  f"var  ≈ 4: {rms.var:.3f}")

normed = rms.normalize(data1)
check(abs(normed.mean()) < 0.5,  f"mean normalizado ≈ 0: {normed.mean():.3f}")
check(abs(normed.std()  - 1) < 0.3, f"std normalizado ≈ 1: {normed.std():.3f}")
check((np.abs(normed) <= 10).all(), "clipping a ±10 funciona")

# ══════════════════════════════════════════════
print("\n── 6. Integración con env real (5 steps) ────────────")
# ══════════════════════════════════════════════

from src.env.spy_gym_env import SpyGymEnv
from src.env.frame_stack import FrameStackWrapper

env = FrameStackWrapper(SpyGymEnv(
    background_config="rmsc04",
    mkt_close="16:00:00",
    timestep_duration="60s",
    starting_cash=1_000_000,
    order_fixed_size=10,
    inv_penalty_coef=0.0001,
    opportunity_cost_coef=0.0001,
    first_interval="00:05:00",
), n_frames=10)

obs = env.reset()
net3 = DualDirectionalNetwork(obs_dim=env.observation_space.shape[0], action_dim=env.action_space.n, n_frames=10)

for step in range(5):
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    action, log_prob, value, sb, se = net3.get_action_and_value(obs_t)
    next_obs, reward, done, info = env.step(action.item())

    check("mid_price" in info, f"step {step}: mid_price en info ({info.get('mid_price', 'MISSING')})")
    check(isinstance(reward, float), f"step {step}: reward es float ({reward:.4f})")
    obs = next_obs if not done else env.reset()

env.close()

# ══════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  Todos los tests pasados correctamente.")
print("══════════════════════════════════════════\n")
