"""
vec_env.py — Vectorized environment con subprocesos separados.

Lanza N copias de un env Gym en procesos independientes (cada uno
con su propia instancia de ABIDES) y expone la interfaz vectorizada:
    obs, rewards, dones, infos = vec_env.step(actions)   # actions: (N,)
    obs = vec_env.reset()                                 # obs: (N, obs_dim)
"""

import multiprocessing as mp
import numpy as np
from typing import Callable, List


def _worker(conn: mp.connection.Connection, env_fn: Callable):
    """Proceso worker. Vive en un loop recibiendo comandos por pipe."""
    env = env_fn()
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                obs = env.reset()
                conn.send(obs)
            elif cmd == "step":
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                conn.send((obs, float(reward), bool(done), info))
            elif cmd == "get_spaces":
                conn.send((env.observation_space, env.action_space))
            elif cmd == "close":
                env.close()
                break
    except Exception as e:
        conn.send(("__error__", str(e)))
    finally:
        conn.close()


class SubprocVecEnv:
    """
    Vectorized environment: N workers en procesos separados.

    Args:
        env_fns: lista de callables que retornan un env Gym cada uno.
    """

    def __init__(self, env_fns: List[Callable]):
        self.n_envs = len(env_fns)
        ctx = mp.get_context("fork")
        self.parent_conns, child_conns = zip(
            *[ctx.Pipe() for _ in range(self.n_envs)]
        )
        self.processes: List[mp.Process] = []
        for child_conn, env_fn in zip(child_conns, env_fns):
            p = ctx.Process(target=_worker, args=(child_conn, env_fn), daemon=True)
            p.start()
            child_conn.close()
            self.processes.append(p)

        # Obtener espacios del primer worker
        self.parent_conns[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.parent_conns[0].recv()

    def reset(self) -> np.ndarray:
        """Resetea todos los envs y devuelve obs vectorizadas (N, obs_dim)."""
        for conn in self.parent_conns:
            conn.send(("reset", None))
        return np.stack([conn.recv() for conn in self.parent_conns])

    def step(self, actions: np.ndarray):
        """
        Ejecuta un step en cada env con su acción correspondiente.

        Args:
            actions: array (N,) de acciones enteras
        Returns:
            obs:     (N, obs_dim)
            rewards: (N,)
            dones:   (N,) bool
            infos:   list de N dicts
        """
        for conn, action in zip(self.parent_conns, actions):
            conn.send(("step", int(action)))
        results = [conn.recv() for conn in self.parent_conns]
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), list(infos)

    def close(self):
        """Cierra todos los procesos workers."""
        for conn in self.parent_conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
