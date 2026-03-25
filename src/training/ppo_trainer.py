"""
ppo_trainer.py — Entrenador PPO (Proximal Policy Optimization)
===============================================================

Este archivo orquesta todo el entrenamiento:
    1. Recolecta experiencia usando networks.py + spy_gym_env.py
    2. Almacena en rollout_buffer.py
    3. Calcula ventajas (GAE)
    4. Actualiza la red con el algoritmo PPO clipped

Hiperparámetros (del execution_plan.md):
    - LR:             3e-4 (Adam)
    - Clip ε:         0.2
    - K epochs:       4 pasadas por update
    - Minibatch:      512
    - Rollout length: 2048 steps
    - γ (gamma):      0.999 (horizonte largo para MM)
    - λ (GAE lambda): 0.95
    - Entropy coef:   0.01
    - Value coef:     0.5
    - Max grad norm:  0.5

Uso:
    from src.env.spy_gym_env import SpyMarketMakerEnv
    from src.training.ppo_trainer import PPOTrainer

    env = SpyMarketMakerEnv(background_config="rmsc04")
    trainer = PPOTrainer(env)
    trainer.train(total_steps=1_000_000)
"""

import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.networks import ActorCritic
from src.training.rollout_buffer import RolloutBuffer


class PPOTrainer:
    """
    Orquestador del entrenamiento PPO.

    Args:
        env:              instancia de SpyMarketMakerEnv (o cualquier Gym env)
        lr:               learning rate para Adam
        gamma:            discount factor
        gae_lambda:       GAE lambda
        clip_eps:         PPO clip epsilon
        entropy_coef:     peso del bonus de entropía (exploración)
        value_coef:       peso de la loss del critic
        max_grad_norm:    gradient clipping
        update_epochs:    pasadas por update (K)
        batch_size:       tamaño de minibatch
        rollout_length:   steps de experiencia por update
        hidden_dim:       neuronas por capa oculta
        checkpoint_dir:   dónde guardar checkpoints
        device:           "cpu" o "cuda"
    """

    def __init__(
        self,
        env,
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
    ):
        self.env = env
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device)

        # Dimensiones del env
        obs_dim = env.observation_space.shape[0]     # 44
        action_dim = env.action_space.n              # 11

        # Red neuronal
        self.network = ActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)

        # Optimizador
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Buffer
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            obs_dim=obs_dim,
            gamma=gamma,
            lam=gae_lambda,
        )

        # Tracking de métricas
        self.total_steps = 0
        self.total_updates = 0
        self.episode_rewards = []
        self.current_episode_reward = 0.0

    # ================================================================
    # MÉTODO PRINCIPAL: train()
    # ================================================================
    def train(
        self,
        total_steps: int = 1_000_000,
        log_interval: int = 10,
        checkpoint_interval: int = 100_000,
    ):
        """
        Bucle principal de entrenamiento.

        Args:
            total_steps:         total de steps en el env
            log_interval:        cada cuántos updates imprimir métricas
            checkpoint_interval: cada cuántos steps guardar modelo
        """
        obs = self.env.reset()
        start_time = time.time()

        while self.total_steps < total_steps:
            # ---- FASE 1: Recolectar experiencia ----
            rollout_info = self._collect_rollout(obs)
            obs = rollout_info["last_obs"]

            # ---- FASE 2: Calcular ventajas (GAE) ----
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                _, _, last_value = self.network.get_action_and_value(obs_tensor)
            self.buffer.compute_gae(
                last_value=last_value.item(),
                last_done=rollout_info["last_done"],
            )

            # ---- FASE 3: Update PPO ----
            update_info = self._ppo_update()
            self.total_updates += 1

            # ---- FASE 4: Logging ----
            if self.total_updates % log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.total_steps / elapsed if elapsed > 0 else 0

                # Reward medio de episodios completados
                avg_reward = (
                    np.mean(self.episode_rewards[-10:])
                    if self.episode_rewards
                    else 0.0
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

            # ---- FASE 5: Checkpoint ----
            if self.total_steps % checkpoint_interval < self.rollout_length:
                self.save_checkpoint(
                    f"checkpoint_{self.total_steps}.pt"
                )

            # ---- Vaciar buffer ----
            self.buffer.reset()

        # Guardar modelo final
        self.save_checkpoint("best_model.pt")
        print(f"\nEntrenamiento completado: {total_steps} steps")

    # ================================================================
    # FASE 1: Recolectar rollout
    # ================================================================
    def _collect_rollout(self, obs: np.ndarray) -> dict:
        """
        Ejecuta rollout_length steps en el env y guarda en el buffer.

        El agente interactúa con ABIDES:
            1. La red mira la observación → elige acción
            2. El env ejecuta la acción en ABIDES → devuelve reward
            3. Se guarda todo en el buffer
        """
        last_done = False

        for step in range(self.rollout_length):
            # Red decide acción
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                action, log_prob, value = self.network.get_action_and_value(
                    obs_tensor
                )

            # Env ejecuta acción
            next_obs, reward, done, info = self.env.step(action.item())

            # Guardar en buffer
            self.buffer.add(
                obs=obs,
                action=action.item(),
                reward=reward,
                done=done,
                log_prob=log_prob.item(),
                value=value.item(),
            )

            # Tracking
            self.total_steps += 1
            self.current_episode_reward += reward

            # Si el episodio terminó, reiniciar
            if done:
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0.0
                next_obs = self.env.reset()

            obs = next_obs
            last_done = done

        return {"last_obs": obs, "last_done": last_done}

    # ================================================================
    # FASE 3: PPO Update
    # ================================================================
    def _ppo_update(self) -> dict:
        """
        Actualiza la red usando PPO clipped objective.

        Hace K=4 pasadas sobre los datos del buffer, en minibatches
        de 512, aplicando tres losses:

        1. Policy loss (clipped):
           - Calcula ratio = prob_nueva / prob_vieja
           - Clippea el ratio a [1-ε, 1+ε] para evitar updates grandes
           - Usa el mínimo entre ratio×advantage y clipped×advantage

        2. Value loss:
           - MSE entre predicción del critic y return real

        3. Entropy bonus:
           - Incentiva exploración (que no siempre haga lo mismo)
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for epoch in range(self.update_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                # Mover a device
                obs = batch["obs"].to(self.device)
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)

                # Normalizar advantages (reduce varianza, mejora estabilidad)
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Re-evaluar acciones con pesos NUEVOS
                new_log_probs, entropy, new_values = (
                    self.network.evaluate_actions(obs, actions)
                )

                # ---- 1. Policy Loss (PPO Clipped) ----
                # ratio = π_nueva(a|s) / π_vieja(a|s)
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Surrogate losses
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * advantages
                )

                # Tomamos el mínimo (pesimista) para ser conservador
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- 2. Value Loss ----
                value_loss = nn.functional.mse_loss(new_values, returns)

                # ---- 3. Entropy Loss ----
                entropy_loss = -entropy.mean()

                # ---- Loss Total ----
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # ---- Backprop + Update ----
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Tracking
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1

        return {
            "policy_loss": total_policy_loss / max(num_batches, 1),
            "value_loss": total_value_loss / max(num_batches, 1),
            "entropy": total_entropy / max(num_batches, 1),
        }

    # ================================================================
    # Guardar / Cargar modelo
    # ================================================================
    def save_checkpoint(self, filename: str):
        """Guarda el estado completo del entrenamiento."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "total_updates": self.total_updates,
                "episode_rewards": self.episode_rewards,
            },
            path,
        )
        print(f"  Checkpoint guardado: {path}")

    def load_checkpoint(self, path: str):
        """Carga un checkpoint guardado para continuar entrenamiento."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        self.total_updates = checkpoint["total_updates"]
        self.episode_rewards = checkpoint["episode_rewards"]
        print(f"  Checkpoint cargado: {path} (step {self.total_steps})")
