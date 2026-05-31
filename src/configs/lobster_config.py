"""
lobster_config.py — Configuración de simulación ABIDES con LOBSTERReplayAgent.

Uso:
    from src.configs.lobster_config import build_config
    config = build_config()
    end_state = abides.run(config)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_markets.agents import ExchangeAgent
from abides_markets.oracles import Oracle
from abides_markets.utils import generate_latency_model

from src.agents.lobster_replay_agent import LOBSTERReplayAgent


class LOBSTEROracle(Oracle):
    """Oracle mínimo para replay: devuelve el mid-price de apertura del orderbook."""

    def __init__(self, open_price: int):
        self.open_price = open_price

    def get_daily_open_price(self, symbol: str, mkt_open: NanosecondTime, cents: bool = True) -> int:
        return self.open_price


def build_config(
    symbol: str = "GOOG",
    date: str = "2012-06-21",
    filepath: str = "data/LOBSTER_SampleFile_GOOG_2012-06-21_10",
    num_levels: int = 10,
    seed: int = 42,
    stdout_log_level: str = "INFO",
):
    date_ns = int(pd.to_datetime(date).to_datetime64())
    mkt_open = date_ns + str_to_ns("09:30:00")
    mkt_close = date_ns + str_to_ns("16:00:00")

    # Mid-price de apertura desde la primera fila del orderbook
    ob_file = f"{filepath}/{symbol}_{date}_34200000_57600000_orderbook_{num_levels}.csv"
    first_row = pd.read_csv(ob_file, header=None).iloc[0]
    open_price = (int(first_row[0]) + int(first_row[2])) // 2

    oracle = LOBSTEROracle(open_price=open_price)

    agents = [
        ExchangeAgent(
            id=0,
            name="EXCHANGE",
            type="ExchangeAgent",
            mkt_open=mkt_open,
            mkt_close=mkt_close,
            symbols=[symbol],
            book_logging=True,
            book_log_depth=num_levels,
            log_orders=False,
            pipeline_delay=0,
            computation_delay=0,
            stream_history=500,
            random_state=np.random.RandomState(seed=seed),
        ),
        LOBSTERReplayAgent(
            id=1,
            symbol=symbol,
            date=date,
            filepath=filepath,
            num_levels=num_levels,
            log_orders=False,
            random_state=np.random.RandomState(seed=0),
        ),
    ]

    return {
        "seed": seed,
        "start_time": mkt_open,
        "stop_time": mkt_close + str_to_ns("1s"),
        "agents": agents,
        "agent_latency_model": generate_latency_model(len(agents), latency_type="no_latency"),
        "default_computation_delay": 50,
        "custom_properties": {"oracle": oracle},
        "random_state_kernel": np.random.RandomState(seed=1),
        "stdout_log_level": stdout_log_level,
    }
