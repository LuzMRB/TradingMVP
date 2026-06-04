"""
ppo_trainer.py — Entrenador PPO con soporte multi-env (SubprocVecEnv).

Para n_envs=1 el comportamiento es idéntico al original (single env).
Para n_envs>1 lanza SubprocVecEnv y recolecta rollouts en paralelo.

Uso:
    from src.env.spy_gym_env import SpyGymEnv
    from src.training.ppo_trainer import PPOTrainer

    def env_fn():
        return SpyGymEnv(background_config="rmsc04", ...)

    trainer = PPOTrainer(env_fn=env_fn, n_envs=12, ...)
    trainer.train(total_steps=500_000)
"""

import os
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.networks import ActorCritic, TransformerActorCritic
from src.env.vec_env import SubprocVecEnv
from src.training.rollout_buffer import RolloutBuffer


class PPOTrainer:
    """
    Orquestador del entrenamiento PPO.

    Args:
        env_fn:           callable que devuelve un env Gym (se llama n_envs veces)
        n_envs:           número de envs en paralelo (1 = single-env, comportamiento original)
        lr:               learning rate para Adam
        gamma:            discount factor
        gae_lambda:       GAE lambda
        clip_eps:         PPO clip epsilon
        entropy_coef:     peso del bonus de entropía
        value_coef:       peso de la loss del critic
        max_grad_norm:    gradient clipping
        update_epochs:    pasadas por update (K)
        batch_size:       tamaño de minibatch
        rollout_length:   steps por update por env
        hidden_dim:       neuronas por capa oculta
        checkpoint_dir:   dónde guardar checkpoints
        device:           "cpu" o "cuda"
        architecture:     "mlp" o "transformer"
    """

    def __init__(
        self,
        env_fn: Callable = None,
        n_envs: int = 1,
        env=None,  # retrocompatibilidad: acepta env= directo (implica n_envs=1)
        lr: float = 3e-4,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 512,
        rollout_length: int = 2048,
        hidden_dim: int = 256,
        checkpoint_dir: str = "experiments/mvp_results/checkpoints",
        device: str = "cpu",
        architecture: str = "mlp",
        n_frames: int = 1,
    ):
        # Retrocompatibilidad: env= directo (notebook antiguo)
        if env is not None and env_fn is None:
            env_fn = lambda: env  # noqa: E731
            n_envs = 1

        self.n_envs = n_envs
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device)

        # Crear env(s)
        if n_envs == 1:
            self.env = env_fn()
            obs_dim    = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.n
        else:
            print(f"Lanzando {n_envs} workers en subprocesos...")
            self.env = SubprocVecEnv([env_fn] * n_envs)
            obs_dim    = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.n
            print(f"  {n_envs} workers listos. obs_dim={obs_dim}, action_dim={action_dim}")

        # Red neuronal
        if architecture == "transformer":
            self.network = TransformerActorCritic(obs_dim, action_dim, n_frames=n_frames).to(self.device)
        else:
            self.network = ActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Buffer — guarda rollout_length pasos por env
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            obs_dim=obs_dim,
            gamma=gamma,
            lam=gae_lambda,
            n_envs=n_envs,
        )

        self.total_steps = 0
        self.total_updates = 0
        self.episode_rewards: list = []
        self.value_losses: list = []
        self.policy_losses: list = []
        self.entropies: list = []
        self._fps_samples: list = []
        # Para n_envs>1 llevamos reward acumulado por env
        self.current_episode_reward = np.zeros(n_envs, dtype=np.float32)

        self._use_gru = False
        self.h = None

    # ================================================================
    # MÉTODO PRINCIPAL: train()
    # ================================================================
    def train(
        self,
        total_steps: int = 1_000_000,
        log_interval: int = 10,
        checkpoint_interval: int = 100_000,
    ):
        obs = self.env.reset()   # (obs_dim,) si n_envs=1; (N, obs_dim) si n_envs>1
        start_time = time.time()

        while self.total_steps < total_steps:
            rollout_info = self._collect_rollout(obs)
            obs = rollout_info["last_obs"]

            # Calcular ventajas con GAE
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).to(self.device)
                _, _, last_values = self.network.get_action_and_value(obs_t)
                if self.n_envs == 1:
                    last_value_np = last_values.item()
                else:
                    last_value_np = last_values.cpu().numpy()

            self.buffer.compute_gae(
                last_value=last_value_np,
                last_done=rollout_info["last_done"],
            )

            update_info = self._ppo_update()
            self.total_updates += 1
            self.buffer.reset()

            # Guardar curvas de entrenamiento
            self.value_losses.append(update_info["value_loss"])
            self.policy_losses.append(update_info["policy_loss"])
            self.entropies.append(update_info["entropy"])

            elapsed = time.time() - start_time
            fps = self.total_steps / elapsed if elapsed > 0 else 0
            self._fps_samples.append(fps)

            if self.total_updates % log_interval == 0:
                avg_reward = (
                    np.mean(self.episode_rewards[-10:])
                    if self.episode_rewards else 0.0
                )
                print(
                    f"Steps: {self.total_steps:>8d} | "
                    f"Updates: {self.total_updates:>4d} | "
                    f"Avg Reward: {avg_reward:>8.3f} | "
                    f"Policy Loss: {update_info['policy_loss']:>8.4f} | "
                    f"Value Loss: {update_info['value_loss']:>8.4f} | "
                    f"Entropy: {update_info['entropy']:>6.3f} | "
                    f"FPS: {fps:>6.0f}"
                )

            if self.total_steps % checkpoint_interval < self.rollout_length * self.n_envs:
                self.save_checkpoint(f"checkpoint_{self.total_steps}.pt")

        self.save_checkpoint("best_model.pt")
        if self.n_envs > 1:
            self.env.close()
        print(f"\nEntrenamiento completado: {total_steps} steps")

    # ================================================================
    # FASE 1: Recolectar rollout
    # ================================================================
    def _collect_rollout(self, obs: np.ndarray) -> dict:
        last_done = False if self.n_envs == 1 else np.zeros(self.n_envs, dtype=bool)

        for _ in range(self.rollout_length):
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).to(self.device)
                action, log_prob, value = self.network.get_action_and_value(obs_t)

            if self.n_envs == 1:
                next_obs, reward, done, _ = self.env.step(action.item())
                self.buffer.add(
                    obs=obs,
                    action=action.item(),
                    reward=reward,
                    done=done,
                    log_prob=log_prob.item(),
                    value=value.item(),
                )
                self.total_steps += 1
                self.current_episode_reward[0] += reward
                if done:
                    self.episode_rewards.append(float(self.current_episode_reward[0]))
                    self.current_episode_reward[0] = 0.0
                    next_obs = self.env.reset()
                last_done = done
            else:
                actions_np   = action.cpu().numpy()
                log_probs_np = log_prob.cpu().numpy()
                values_np    = value.cpu().numpy()

                next_obs, rewards, dones, _ = self.env.step(actions_np)

                self.buffer.add(
                    obs=obs,
                    action=actions_np,
                    reward=rewards,
                    done=dones,
                    log_prob=log_probs_np,
                    value=values_np,
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
    # FASE 3: PPO Update
    # ================================================================
    def _ppo_update(self) -> dict:
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        num_batches       = 0

        for _ in range(self.update_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                obs           = batch["obs"].to(self.device)
                actions       = batch["actions"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                advantages    = batch["advantages"].to(self.device)
                returns       = batch["returns"].to(self.device)

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                new_log_probs, entropy, new_values = self.network.evaluate_actions(obs, actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss  = nn.functional.mse_loss(new_values, returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.mean().item()
                num_batches       += 1

        return {
            "policy_loss": total_policy_loss / max(num_batches, 1),
            "value_loss":  total_value_loss  / max(num_batches, 1),
            "entropy":     total_entropy     / max(num_batches, 1),
        }

    # ================================================================
    # Guardar / Cargar modelo
    # ================================================================
    def save_checkpoint(self, filename: str):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "network_state_dict":   self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps":          self.total_steps,
                "total_updates":        self.total_updates,
                "episode_rewards":      self.episode_rewards,
            },
            path,
        )
        print(f"  Checkpoint guardado: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps    = checkpoint["total_steps"]
        self.total_updates  = checkpoint["total_updates"]
        self.episode_rewards = checkpoint["episode_rewards"]
        print(f"  Checkpoint cargado: {path} (step {self.total_steps})")
