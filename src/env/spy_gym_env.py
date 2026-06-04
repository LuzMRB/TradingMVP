import importlib
from typing import Any, Dict, List

import gym
import numpy as np

import abides_markets.agents.utils as markets_agent_utils
from abides_core.utils import str_to_ns
from abides_core.generators import ConstantTimeGenerator
from abides_gym.envs.markets_environment import AbidesGymMarketsEnv

LOB_LEVELS = 10


class SpyGymEnv(AbidesGymMarketsEnv):
    """
    Entorno Gym para trading direccional con ABIDES.

    Observación (44 features):
        - 10 niveles bid: (precio_bps_desde_mid, log1p_volumen) → 20
        - 10 niveles ask: (precio_bps_desde_mid, log1p_volumen) → 20
        - 4 portfolio: (holdings_norm, cash_norm, pnl_norm, time_progress)

    Acciones:
        0 → MKT BUY  order_fixed_size acciones
        1 → HOLD
        2 → MKT SELL order_fixed_size acciones

    Recompensa: PnL mark-to-market incremental por step
    """

    raw_state_pre_process = markets_agent_utils.ignore_buffers_decorator
    raw_state_to_state_pre_process = markets_agent_utils.ignore_buffers_decorator

    def __init__(
        self,
        background_config: str = "rmsc04",
        mkt_close: str = "10:00:00",
        timestep_duration: str = "60s",
        starting_cash: int = 1_000_000,
        order_fixed_size: int = 10,
        max_inventory: int = 100,
        inv_penalty_coef: float = 0.1,
        state_history_length: int = 2,
        market_data_buffer_length: int = 5,
        first_interval: str = "00:05:00",
        background_config_extra_kvargs: Dict = {},
    ) -> None:
        self.background_config = importlib.import_module(
            "abides_markets.configs.{}".format(background_config), package=None
        )
        self.mkt_close = str_to_ns(mkt_close)
        self.timestep_duration = str_to_ns(timestep_duration)
        self.starting_cash = starting_cash
        self.order_fixed_size = order_fixed_size
        self.max_inventory = max_inventory
        self.inv_penalty_coef = inv_penalty_coef
        self.state_history_length = state_history_length
        self.market_data_buffer_length = market_data_buffer_length
        self.first_interval = str_to_ns(first_interval)

        self.price_norm = 100_000.0
        self.previous_marked_to_market = float(self.starting_cash)
        self.current_step = 0
        self.current_mid_price = float(self.price_norm)

        total_time_ns = self.mkt_close - self.first_interval
        self.total_steps = max(int(total_time_ns / self.timestep_duration), 1)

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

        self.num_actions = 3
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.num_state_features = LOB_LEVELS * 4 + 4  # 44
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_state_features,),
            dtype=np.float32,
        )

    def reset(self):
        self.previous_marked_to_market = float(self.starting_cash)
        self.current_step = 0
        self.current_mid_price = float(self.price_norm)
        return super().reset()

    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(
        self, action: int
    ) -> List[Dict[str, Any]]:
        holdings = getattr(self, '_last_holdings', 0)
        if action == 0 and holdings < self.max_inventory:
            return [{"type": "MKT", "direction": "BUY", "size": self.order_fixed_size}]
        elif action == 2 and holdings > 0:
            return [{"type": "MKT", "direction": "SELL", "size": self.order_fixed_size}]
        return []

    @raw_state_to_state_pre_process
    def raw_state_to_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        bids = raw_state["parsed_mkt_data"]["bids"]
        asks = raw_state["parsed_mkt_data"]["asks"]
        last_transaction = raw_state["parsed_mkt_data"]["last_transaction"]
        holdings = raw_state["internal_data"]["holdings"]
        cash = raw_state["internal_data"]["cash"]

        mid_price = markets_agent_utils.get_mid_price(bids, asks, last_transaction)
        if mid_price <= 0:
            mid_price = float(last_transaction) if last_transaction > 0 else self.price_norm
        self.current_mid_price = float(mid_price)
        self._last_holdings = int(holdings)

        self.current_step += 1
        time_progress = min(self.current_step / self.total_steps, 1.0)

        features = []
        for i in range(LOB_LEVELS):
            price, vol = markets_agent_utils.get_val(bids, i)
            bps = (price - mid_price) / mid_price * 10_000 if mid_price > 0 else 0.0
            features.extend([float(bps), float(np.log1p(vol))])

        for i in range(LOB_LEVELS):
            price, vol = markets_agent_utils.get_val(asks, i)
            bps = (price - mid_price) / mid_price * 10_000 if mid_price > 0 else 0.0
            features.extend([float(bps), float(np.log1p(vol))])

        unrealized_pnl = (cash + holdings * mid_price - self.starting_cash) / self.starting_cash
        features.extend([
            holdings / self.max_inventory,
            cash / self.starting_cash,
            float(unrealized_pnl),
            time_progress,
        ])

        return np.array(features, dtype=np.float32)

    @raw_state_pre_process
    def raw_state_to_reward(self, raw_state: Dict[str, Any]) -> float:
        holdings = raw_state["internal_data"]["holdings"]
        cash = raw_state["internal_data"]["cash"]
        last_transaction = raw_state["parsed_mkt_data"]["last_transaction"]

        m2m = cash + holdings * last_transaction
        pnl = (m2m - self.previous_marked_to_market) / self.starting_cash
        inv_norm = min(abs(holdings) / self.max_inventory, 1.0)
        inv_penalty = self.inv_penalty_coef * inv_norm ** 2
        self.previous_marked_to_market = m2m
        return float(pnl - inv_penalty)

    @raw_state_pre_process
    def raw_state_to_done(self, raw_state: Dict[str, Any]) -> bool:
        return False

    @raw_state_pre_process
    def raw_state_to_update_reward(self, raw_state: Dict[str, Any]) -> float:
        return 0.0

    @raw_state_pre_process
    def raw_state_to_info(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "holdings":  raw_state["internal_data"]["holdings"],
            "cash":      raw_state["internal_data"]["cash"],
            "mid_price": self.current_mid_price,
        }
