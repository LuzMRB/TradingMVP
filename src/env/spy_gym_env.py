import importlib
from typing import Any, Dict, List

import gym
import numpy as np

import abides_markets.agents.utils as markets_agent_utils
from abides_core.utils import str_to_ns
from abides_core.generators import ConstantTimeGenerator
from abides_gym.envs.markets_environment import AbidesGymMarketsEnv


class SpyGymEnv(AbidesGymMarketsEnv):
    """
    Entorno Gym para trading con ABIDES.

    Observación (8 features flat):
        [bid_p, bid_v, ask_p, ask_v, holdings, cash, mid_price, spread]
        — todas normalizadas

    Acciones:
        0 → MKT BUY  order_fixed_size acciones
        1 → HOLD
        2 → MKT SELL order_fixed_size acciones

    Recompensa: PnL mark-to-market incremental por step
    """

    raw_state_pre_process = markets_agent_utils.ignore_buffers_decorator
    raw_state_to_state_pre_process = (
        markets_agent_utils.ignore_mkt_data_buffer_decorator
    )

    def __init__(
        self,
        background_config: str = "rmsc04",
        mkt_close: str = "10:00:00",
        timestep_duration: str = "60s",
        starting_cash: int = 1_000_000,
        order_fixed_size: int = 10,
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
        self.state_history_length = state_history_length
        self.market_data_buffer_length = market_data_buffer_length
        self.first_interval = str_to_ns(first_interval)

        # Ancla de normalización: r_bar de rmsc04 en centavos ($1000)
        self.price_norm = 100_000.0

        # Para el cálculo de recompensa incremental
        self.previous_marked_to_market = float(self.starting_cash)

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

        self.num_state_features = 8
        self.observation_space = gym.spaces.Box(
            low=-np.finfo(np.float32).max,
            high=np.finfo(np.float32).max,
            shape=(self.num_state_features,),
            dtype=np.float32,
        )

    def reset(self):
        self.previous_marked_to_market = float(self.starting_cash)
        return super().reset()

    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(
        self, action: int
    ) -> List[Dict[str, Any]]:
        if action == 0:
            return [{"type": "MKT", "direction": "BUY", "size": self.order_fixed_size}]
        elif action == 1:
            return []
        else:
            return [{"type": "MKT", "direction": "SELL", "size": self.order_fixed_size}]

    @raw_state_to_state_pre_process
    def raw_state_to_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        bids = raw_state["parsed_mkt_data"]["bids"]
        asks = raw_state["parsed_mkt_data"]["asks"]
        last_transactions = raw_state["parsed_mkt_data"]["last_transaction"]

        mid_prices = [
            markets_agent_utils.get_mid_price(b, a, lt)
            for b, a, lt in zip(bids, asks, last_transactions)
        ]
        mid_price = mid_prices[-1]

        bid_price, bid_vol = markets_agent_utils.get_val(bids[-1], 0)
        ask_price, ask_vol = markets_agent_utils.get_val(asks[-1], 0)
        spread = ask_price - bid_price

        holdings = raw_state["internal_data"]["holdings"][-1]
        cash = raw_state["internal_data"]["cash"][-1]

        p = self.price_norm
        return np.array(
            [
                bid_price / p,
                bid_vol / 100.0,
                ask_price / p,
                ask_vol / 100.0,
                holdings / 100.0,
                cash / self.starting_cash,
                mid_price / p,
                spread / p,
            ],
            dtype=np.float32,
        )

    @raw_state_pre_process
    def raw_state_to_reward(self, raw_state: Dict[str, Any]) -> float:
        holdings = raw_state["internal_data"]["holdings"]
        cash = raw_state["internal_data"]["cash"]
        last_transaction = raw_state["parsed_mkt_data"]["last_transaction"]

        m2m = cash + holdings * last_transaction
        reward = (m2m - self.previous_marked_to_market) / max(self.order_fixed_size, 1)
        self.previous_marked_to_market = m2m
        return float(reward)

    @raw_state_pre_process
    def raw_state_to_done(self, raw_state: Dict[str, Any]) -> bool:
        holdings = raw_state["internal_data"]["holdings"]
        cash = raw_state["internal_data"]["cash"]
        last_transaction = raw_state["parsed_mkt_data"]["last_transaction"]
        m2m = cash + holdings * last_transaction
        return bool(m2m < 0.5 * self.starting_cash)

    @raw_state_pre_process
    def raw_state_to_update_reward(self, raw_state: Dict[str, Any]) -> float:
        return 0.0

    @raw_state_pre_process
    def raw_state_to_info(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "holdings": raw_state["internal_data"]["holdings"],
            "cash": raw_state["internal_data"]["cash"],
        }
