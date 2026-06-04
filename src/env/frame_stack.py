"""
frame_stack.py — Wrapper que apila N observaciones consecutivas.

Convierte obs (obs_dim,) en (N * obs_dim,) apilando los últimos N steps.
Los primeros steps del episodio rellenan con ceros los frames faltantes.
"""

import numpy as np


class FrameStackWrapper:
    """
    Wrapper que apila los últimos N frames de observación.

    Args:
        env:     env Gym base
        n_frames: número de frames a apilar
    """

    def __init__(self, env, n_frames: int = 10):
        self.env = env
        self.n_frames = n_frames
        self._obs_dim = env.observation_space.shape[0]
        self._buffer = np.zeros((n_frames, self._obs_dim), dtype=np.float32)

        # Sobreescribir observation_space con la nueva shape
        import gym
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_frames * self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = env.action_space

        # Exponer atributos del env base que usa evaluate_model
        self.starting_cash = env.starting_cash
        self.price_norm = env.price_norm

    def reset(self):
        obs = self.env.reset()
        self._buffer[:] = 0.0
        self._buffer[-1] = obs
        return self._buffer.flatten()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._buffer = np.roll(self._buffer, shift=-1, axis=0)
        self._buffer[-1] = obs
        return self._buffer.flatten(), reward, done, info

    def close(self):
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)
