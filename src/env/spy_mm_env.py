"""
SpyMarketMakerEnv — Gym environment para entrenar un agente de Market Making
sobre el simulador ABIDES (config rmsc04).

ESPACIO DE OBSERVACIÓN: 44 features
    - 10 niveles del LOB en el lado bid  (precio_bps + log_volumen) → 20
    - 10 niveles del LOB en el lado ask  (precio_bps + log_volumen) → 20
    - 4 features de portfolio: posición normalizada, cash normalizado,
      PnL no realizado, progreso temporal del episodio              →  4

    NOTA — ampliación futura (NO implementada en esta versión):
    Si el agente no aprende bien con estas 44, se pueden añadir
    features cross-time (returns, momentum, microprice histórico) o
    estructurales (imbalance del libro, spread normalizado en bps).
    Cualquier ampliación a 45-47 features REQUIERE pasar a la
    "Opción B" de decoradores — ver bloque de decoradores más abajo.

ESPACIO DE ACCIÓN: 9 acciones discretas, (bid_offset, ask_offset) ∈ {0,1,2}²
    En cada step el agente elige UNA pareja (bid_offset, ask_offset)
    que define simultáneamente a qué profundidad coloca su BID y su ASK:
        - offset = 0 → al toque del libro (más agresivo)
        - offset = 1 → 1 tick por dentro del libro
        - offset = 2 → 2 ticks por dentro del libro

    Mapeo action_id ∈ {0..8} → (bid_offset, ask_offset):
        bid_offset = action_id // 3
        ask_offset = action_id  % 3

    Antes de colocar las nuevas órdenes se cancelan las activas
    (CCL_ALL) — comportamiento estándar de un Market Maker.

REWARD POR STEP (Avellaneda & Stoikov, 2008):
    r_t = realized_pnl  -  γ · q²  +  quoting_bonus  -  adverse_selection

    donde:
        - realized_pnl viene de la marcación a mercado (Hull, Cap. 2)
        - γ · q² es la penalización cuadrática de inventario (A-S 2008)
        - quoting_bonus y adverse_selection son REWARD SHAPING heurístico
          (no están en ningún paper — ver advertencia más abajo)

REWARD TERMINAL (raw_state_to_update_reward):
    update_reward = -γ · (q / max_inventory)²

    Penalización cuadrática del inventario al cierre del episodio.
    Forma derivada del running cost de Avellaneda-Stoikov (2008): el
    término de aversión al riesgo en horizonte finito integra a una
    penalización cuadrática terminal. Sin este término, el agente
    puede aprender a acabar con inventario gigante (reward hacking).

═══════════════════════════════════════════════════════════════════════════
REFERENCIAS BIBLIOGRÁFICAS
═══════════════════════════════════════════════════════════════════════════

[1] Avellaneda, M. & Stoikov, S. (2008). "High-frequency trading in a
    limit order book". Quantitative Finance 8(3), 217-224.
    → Framework matemático fundacional del MM. De aquí salen:
      · Concepto de bid_offset δ_b y ask_offset δ_a desde el midprice
      · Penalización cuadrática de inventario  -γ · q²
      · Parámetro γ de aversión al riesgo (rango usual: 0.001 - 0.1)

[2] Ganesh, S., Vadori, N., Xu, M., Zheng, H., Reddy, P. & Veloso, M.
    (2019). "Reinforcement Learning for Market Making in a Multi-Agent
    Dealer Market". JPMC AI Research, NeurIPS 2019 Workshop on Robust
    AI in Financial Services.
    → Aplicación de RL al MM sobre ABIDES (es el paper "espejo" del MVP).
      Justifica el espacio de acción discreto y la observación basada
      en el LOB. CITA OBLIGATORIA en el Related Work del TFG.

[3] Spooner, T., Fearnley, J., Savani, R. & Koukorinis, A. (2018).
    "Market Making via Reinforcement Learning". AAMAS 2018, pp. 434-442.
    → Discretización del action space en niveles del LOB (pattern que
      adoptamos para el {0,1,2}² de las 9 acciones).

[4] Hull, J. (2018). "Options, Futures, and Other Derivatives" (10th
    ed.). Pearson, Cap. 2-3.
    → Cálculo mark-to-market estándar del PnL:
        PnL_t = cash_t + position_t · mid_t

[5] O'Hara, M. (1995). "Market Microstructure Theory". Blackwell.
    → Coste de cruzar el spread en liquidación = |q| · (spread/2).
      No se usa en update_reward de la versión inicial; reservado
      como alternativa si la penalización cuadrática no funciona.

═══════════════════════════════════════════════════════════════════════════
ADVERTENCIA — REWARD SHAPING NO PAPER-BACKED
═══════════════════════════════════════════════════════════════════════════
Dos términos del reward son HEURÍSTICOS (no están en ningún paper):

    · quoting_bonus     → bonus pequeño (+0.001) si el agente cotiza
                          ambos lados. Evita que colapse a HOLD.
    · adverse_selection → penalización cuando el mid se mueve y
                          tenemos posición. Inspirado en Glosten-Milgrom
                          (1985) y Kyle (1985), pero la forma específica
                          es shaping mío.

Estos dos términos deben etiquetarse como "reward shaping" en el TFG
y discutirse en la sección de metodología. Son legítimos en RL aplicado
pero no son ecuaciones derivadas de teoría.
═══════════════════════════════════════════════════════════════════════════

Convención de los decoradores raw_state_to_*: heredada de
abides-jpmc-public/abides-gym/abides_gym/envs/markets_execution_environment_v0.py
"""

import importlib
from typing import Any, Dict, List, Optional

import gym
import numpy as np

import abides_markets.agents.utils as markets_agent_utils
from abides_core import NanosecondTime
from abides_core.generators import ConstantTimeGenerator
from abides_core.utils import str_to_ns

from abides_gym.envs.markets_environment import AbidesGymMarketsEnv


class SpyMarketMakerEnv(AbidesGymMarketsEnv):
    """
    Market Maker environment con observación LOB + portfolio (44 features)
    y 9 acciones cuadriculadas (bid_offset, ask_offset) ∈ {0,1,2}².

    Hereda de AbidesGymMarketsEnv: recibe gratis el Kernel, reset/step,
    el FinancialGymAgent (suscrito a L2 con depth=10), gestión de seed
    y render. Solo implementamos los 5 métodos abstractos raw_state_to_*
    y el _map_action_space_to_ABIDES_SIMULATOR_SPACE.
    """

    # ─────────────────────────────────────────────────────────────────
    # DECORADORES (Decisión 4 — Opción A simple)
    #
    # Aplicamos ignore_buffers_decorator a TODOS los raw_state_to_*.
    # Aplana el buffer histórico y nos entrega un único snapshot plano
    # por método, más fácil de manipular.
    #
    # ─── AMPLIACIÓN FUTURA — Opción B (NO implementada) ─────────────
    # Si el agente no aprende y necesitamos features cross-time
    # (returns, momentum, microprice histórico), seguiríamos el patrón
    # de SubGymMarketsExecutionEnv_v0:
    #
    #     raw_state_pre_process = ignore_buffers_decorator
    #     raw_state_to_state_pre_process = ignore_mkt_data_buffer_decorator
    #
    # y decoraríamos raw_state_to_state con el segundo (mantiene la
    # historia de wakeups intacta). Implica también ampliar la
    # observación a 45-47 features.
    # ─────────────────────────────────────────────────────────────────
    raw_state_pre_process = markets_agent_utils.ignore_buffers_decorator

    def __init__(
        self,
        # ── Parámetros heredados de ABIDES ─────────────────────────
        background_config: str = "rmsc04",
        mkt_close: str = "16:00:00",
        # Decisión 5 — wakeup cada 1s para Market Making rápido.
        timestep_duration: str = "1s",
        starting_cash: int = 1_000_000,
        order_fixed_size: int = 10,
        state_history_length: int = 2,
        market_data_buffer_length: int = 5,
        # 5 minutos antes del primer wakeup → da tiempo al libro a
        # formarse con los background agents.
        first_interval: str = "00:05:00",
        # ── Hiperparámetros nuevos del Market Maker ────────────────
        # Tope lógico de posición. Sirve para normalizar la feature
        # de portfolio y como referencia en los términos del reward.
        max_inventory: int = 100,
        # γ (gamma) — aversión al riesgo de inventario.
        # Avellaneda-Stoikov (2008). Rango usual: 0.001 - 0.1.
        # 0.01 = punto medio razonable; lo ajustamos tras el
        # primer entrenamiento de 100K steps.
        inv_penalty_coef: float = 0.01,
        # REWARD SHAPING (heurístico, no paper-backed):
        # Penaliza tener posición cuando el mid se mueve.
        adverse_selection_coef: float = 0.1,
        # REWARD SHAPING (heurístico, no paper-backed):
        # Bonus por cotizar ambos lados. Evita colapso a HOLD.
        quoting_bonus: float = 0.001,
        debug_mode: bool = False,
        background_config_extra_kvargs: Dict[str, Any] = {},
    ) -> None:
        # ── Carga de la config de fondo (background agents) ────────
        self.background_config: Any = importlib.import_module(
            "abides_markets.configs.{}".format(background_config), package=None
        )

        # ── Conversión de tiempos a nanosegundos ──────────────────
        self.mkt_close: NanosecondTime = str_to_ns(mkt_close)
        self.timestep_duration: NanosecondTime = str_to_ns(timestep_duration)
        self.first_interval: NanosecondTime = str_to_ns(first_interval)

        # ── Guarda hiperparámetros heredados ──────────────────────
        self.starting_cash: int = starting_cash
        self.order_fixed_size: int = order_fixed_size
        self.state_history_length: int = state_history_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.debug_mode: bool = debug_mode

        # ── Guarda hiperparámetros del Market Maker ───────────────
        self.max_inventory: int = max_inventory
        self.inv_penalty_coef: float = inv_penalty_coef
        self.adverse_selection_coef: float = adverse_selection_coef
        self.quoting_bonus: float = quoting_bonus

        # ── Variables de estado interno ───────────────────────────
        # Las usan raw_state_to_* y _map_action_space_to_*. Se
        # actualizan paso a paso por los métodos correspondientes
        # (se rellenarán al implementar esos métodos).
        self.previous_mid_price: Optional[float] = None
        self.has_active_bid: bool = False
        self.has_active_ask: bool = False
        self.current_step: int = 0
        # Cache del LOB del último step. Lo necesita
        # _map_action_space_to_ABIDES_SIMULATOR_SPACE para saber el
        # precio real de cada nivel del libro al colocar las órdenes.
        self._last_bids: List = []
        self._last_asks: List = []

        # ── Validaciones (mismo patrón que execution_v0) ──────────
        assert background_config in [
            "rmsc03",
            "rmsc04",
            "smc_01",
        ], "background_config debe ser rmsc03, rmsc04 o smc_01"

        assert (self.first_interval <= str_to_ns("16:00:00")) & (
            self.first_interval >= str_to_ns("00:00:00")
        ), "first_interval fuera de rango"

        assert (self.mkt_close <= str_to_ns("16:00:00")) & (
            self.mkt_close >= str_to_ns("09:30:00")
        ), "mkt_close fuera de horario de mercado"

        assert (self.timestep_duration <= str_to_ns("06:30:00")) & (
            self.timestep_duration >= str_to_ns("00:00:00")
        ), "timestep_duration fuera de rango"

        assert (
            isinstance(self.starting_cash, int) and self.starting_cash >= 0
        ), "starting_cash debe ser entero >= 0"

        assert (
            isinstance(self.order_fixed_size, int) and self.order_fixed_size >= 0
        ), "order_fixed_size debe ser entero >= 0"

        assert isinstance(self.max_inventory, int) and self.max_inventory > 0, (
            "max_inventory debe ser entero > 0"
        )

        assert 0.0 <= self.inv_penalty_coef <= 1.0, (
            "inv_penalty_coef debe estar en [0, 1]"
        )

        # ── Llamada al padre — orquesta el Kernel de ABIDES ───────
        # Patrón idéntico al de SubGymMarketsExecutionEnv_v0:
        # mkt_close se inyecta como end_time dentro del tuple
        # background_config_pair (lo lee build_config).
        background_config_args = {"end_time": self.mkt_close}
        background_config_args.update(background_config_extra_kvargs)
        super().__init__(
            background_config_pair=(
                self.background_config.build_config,
                background_config_args,
            ),
            wakeup_interval_generator=ConstantTimeGenerator(
                step_duration=self.timestep_duration
            ),
            starting_cash=self.starting_cash,
            state_buffer_length=self.state_history_length,
            market_data_buffer_length=self.market_data_buffer_length,
            first_interval=self.first_interval,
        )

        # ── Action Space ──────────────────────────────────────────
        # 9 acciones cuadriculadas (bid_offset, ask_offset) ∈ {0,1,2}²
        # Justificación: Avellaneda-Stoikov 2008 (formulación continua)
        # + Spooner et al. 2018 / Ganesh-JPMC 2019 (discretización RL).
        self.num_actions: int = 9
        self.action_space: gym.Space = gym.spaces.Discrete(self.num_actions)

        # ── Observation Space ────────────────────────────────────
        # 44 features: 40 LOB (10 niveles × 2 lados × 2 atributos)
        #              + 4 portfolio.
        # Box(-inf, +inf) para evitar problemas de clipping en step().
        self.num_state_features: int = 44
        self.observation_space: gym.Space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_state_features,),
            dtype=np.float32,
        )

    # ═══════════════════════════════════════════════════════════════
    # MÉTODOS ABSTRACTOS — pendientes de implementar en commits
    # siguientes (uno por commit, según la estructura del branch):
    #
    #   1. _map_action_space_to_ABIDES_SIMULATOR_SPACE(action)
    #   2. raw_state_to_state(raw_state)
    #   3. raw_state_to_reward(raw_state)
    #   4. raw_state_to_update_reward(raw_state)   ← terminal A-S
    #   5. raw_state_to_done(raw_state)
    #   6. raw_state_to_info(raw_state)
    #
    # Helpers privados:
    #   - _get_bid_price_at_level(level)
    #   - _get_ask_price_at_level(level)
    # ═══════════════════════════════════════════════════════════════
