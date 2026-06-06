"""
dual_trainer.py — Entrenador para arquitectura Dual Bull/Bear + Arbitrador.

Dos objetivos de entrenamiento sobre el mismo encoder compartido:
  1. Bull/Bear heads: BCE contra targets binarios de dirección de precio (t+k)
  2. Arbitrador:      PPO estándar con reward M2M normalizado

Loss total:
    loss = loss_arbitrador + alpha * (loss_bull + loss_bear)
"""

import os
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.dual_network import DualDirectionalNetwork
from src.env.vec_env import SubprocVecEnv
from src.training.dual_buffer import DualRolloutBuffer
from src.training.running_stats import RunningMeanStd


class DualTrainer:

    def __init__(
        self,
        env_fn: Callable,
        n_envs: int        = 10,
        lr: float          = 1e-4,
        gamma: float       = 0.999,
        gae_lambda: float  = 0.95,
        clip_eps: float    = 0.2,
        entropy_coef: float = 0.02,
        value_coef: float  = 0.5,
        alpha: float       = 0.5,       # peso de loss Bull/Bear sobre el encoder
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int    = 256,
        rollout_length: int = 1024,
        k_delay: int       = 5,         # steps de delay para reward Bull/Bear
        checkpoint_dir: str = "experiments/mvp_results/checkpoints",
        device: str        = "cpu",
        n_frames: int      = 10,
    ):
        self.n_envs         = n_envs
        self.clip_eps       = clip_eps
        self.entropy_coef   = entropy_coef
        self.value_coef     = value_coef
        self.alpha          = alpha
        self.max_grad_norm  = max_grad_norm
        self.update_epochs  = update_epochs
        self.batch_size     = batch_size
        self.rollout_length = rollout_length
        self.checkpoint_dir = checkpoint_dir
        self.device         = torch.device(device)

        print(f"Lanzando {n_envs} workers en subprocesos...")
        self.env = SubprocVecEnv([env_fn] * n_envs)
        obs_dim    = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        print(f"  {n_envs} workers listos. obs_dim={obs_dim}, action_dim={action_dim}")

        self.network = DualDirectionalNetwork(
            obs_dim=obs_dim, action_dim=action_dim, n_frames=n_frames,
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.buffer = DualRolloutBuffer(
            buffer_size=rollout_length,
            obs_dim=obs_dim,
            gamma=gamma,
            lam=gae_lambda,
            n_envs=n_envs,
            k_delay=k_delay,
        )

        # Normalización online del reward del arbitrador (M2M)
        self.arb_rms = RunningMeanStd()

        self.total_steps   = 0
        self.total_updates = 0
        self.episode_rewards: list = []
        self.value_losses:    list = []
        self.policy_losses:   list = []
        self.entropies:       list = []
        self.bull_losses:     list = []
        self.bear_losses:     list = []
        self._fps_samples:    list = []

        self.current_episode_reward = np.zeros(n_envs, dtype=np.float32)

    # ================================================================
    def train(self, total_steps: int = 1_000_000, log_interval: int = 1,
              checkpoint_interval: int = 100_000):
        obs = self.env.reset()
        start_time = time.time()

        while self.total_steps < total_steps:
            rollout_info = self._collect_rollout(obs)
            obs = rollout_info["last_obs"]

            # Targets Bull/Bear con delay k
            self.buffer.compute_directional_targets()

            # GAE sobre arb_rewards normalizados
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).to(self.device)
                _, _, last_values, _, _ = self.network.get_action_and_value(obs_t)
                last_value_np = last_values.cpu().numpy()

            self.buffer.compute_gae(
                last_value=last_value_np,
                last_done=rollout_info["last_done"],
            )

            update_info = self._update()
            self.total_updates += 1
            self.buffer.reset()

            self.value_losses.append(update_info["value_loss"])
            self.policy_losses.append(update_info["policy_loss"])
            self.entropies.append(update_info["entropy"])
            self.bull_losses.append(update_info["bull_loss"])
            self.bear_losses.append(update_info["bear_loss"])

            elapsed = time.time() - start_time
            fps = self.total_steps / elapsed if elapsed > 0 else 0
            self._fps_samples.append(fps)

            if self.total_updates % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
                print(
                    f"Steps: {self.total_steps:>8d} | "
                    f"Updates: {self.total_updates:>4d} | "
                    f"Avg Reward: {avg_reward:>7.3f} | "
                    f"PLoss: {update_info['policy_loss']:>8.4f} | "
                    f"VLoss: {update_info['value_loss']:>7.4f} | "
                    f"Bull: {update_info['bull_loss']:>6.4f} | "
                    f"Bear: {update_info['bear_loss']:>6.4f} | "
                    f"Ent: {update_info['entropy']:>5.3f} | "
                    f"FPS: {fps:>5.0f}"
                )

            if self.total_steps % checkpoint_interval < self.rollout_length * self.n_envs:
                self.save_checkpoint(f"checkpoint_{self.total_steps}.pt")

        self.save_checkpoint("best_model.pt")
        self.env.close()
        print(f"\nEntrenamiento completado: {total_steps} steps")

    # ================================================================
    def _collect_rollout(self, obs: np.ndarray) -> dict:
        last_done = np.zeros(self.n_envs, dtype=bool)

        for _ in range(self.rollout_length):
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).to(self.device)
                action, log_prob, value, score_bull, score_bear = \
                    self.network.get_action_and_value(obs_t)

            actions_np    = action.cpu().numpy()
            log_probs_np  = log_prob.cpu().numpy()
            values_np     = value.cpu().numpy()
            score_bull_np = score_bull.cpu().numpy()
            score_bear_np = score_bear.cpu().numpy()

            next_obs, rewards, dones, infos = self.env.step(actions_np)

            # Mid price para reward Bull/Bear (retardado)
            mid_prices = np.array([
                info.get("mid_price", 1.0) for info in infos
            ], dtype=np.float32)

            self.buffer.add(
                obs=obs,
                action=actions_np,
                arb_reward=rewards,
                done=dones,
                log_prob=log_probs_np,
                value=values_np,
                score_bull=score_bull_np,
                score_bear=score_bear_np,
                mid_price=mid_prices,
            )

            self.total_steps += self.n_envs
            self.current_episode_reward += rewards
            for i, done in enumerate(dones):
                if done:
                    self.episode_rewards.append(float(self.current_episode_reward[i]))
                    self.current_episode_reward[i] = 0.0
            last_done = dones
            obs = next_obs

        return {"last_obs": obs, "last_done": last_done}

    # ================================================================
    def _update(self) -> dict:
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_bull_loss   = 0.0
        total_bear_loss   = 0.0
        num_batches       = 0

        bce = nn.BCELoss()

        for _ in range(self.update_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                obs           = batch["obs"].to(self.device)
                actions       = batch["actions"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                advantages    = batch["advantages"].to(self.device)
                returns       = batch["returns"].to(self.device)
                bull_target   = batch["bull_target"].to(self.device)
                bear_target   = batch["bear_target"].to(self.device)

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                new_log_probs, entropy, new_values, score_bull, score_bear = \
                    self.network.evaluate_actions(obs, actions)

                # ── Loss arbitrador (PPO) ──────────────────────
                ratio  = torch.exp(new_log_probs - old_log_probs)
                surr1  = ratio * advantages
                surr2  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = nn.functional.mse_loss(new_values, returns)
                entropy_loss = -entropy.mean()

                loss_arb = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # ── Loss Bull/Bear (BCE) ───────────────────────
                loss_bull = bce(score_bull, bull_target)
                loss_bear = bce(score_bear, bear_target)

                # ── Loss total ────────────────────────────────
                loss = loss_arb + self.alpha * (loss_bull + loss_bear)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.mean().item()
                total_bull_loss   += loss_bull.item()
                total_bear_loss   += loss_bear.item()
                num_batches       += 1

        n = max(num_batches, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss":  total_value_loss  / n,
            "entropy":     total_entropy     / n,
            "bull_loss":   total_bull_loss   / n,
            "bear_loss":   total_bear_loss   / n,
        }

    # ================================================================
    def save_checkpoint(self, filename: str):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "network_state_dict":   self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps":          self.total_steps,
            "total_updates":        self.total_updates,
            "episode_rewards":      self.episode_rewards,
            "arb_rms_mean":         self.arb_rms.mean,
            "arb_rms_var":          self.arb_rms.var,
            "arb_rms_count":        self.arb_rms.count,
        }, path)
        print(f"  Checkpoint guardado: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps    = checkpoint["total_steps"]
        self.total_updates  = checkpoint["total_updates"]
        self.episode_rewards = checkpoint["episode_rewards"]
        if "arb_rms_mean" in checkpoint:
            self.arb_rms.mean  = checkpoint["arb_rms_mean"]
            self.arb_rms.var   = checkpoint["arb_rms_var"]
            self.arb_rms.count = checkpoint["arb_rms_count"]
        print(f"  Checkpoint cargado: {path} (step {self.total_steps})")
