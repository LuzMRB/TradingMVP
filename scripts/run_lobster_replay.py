"""
Script para ejecutar una simulación ABIDES con el LOBSTERReplayAgent.

Uso:
    python scripts/run_lobster_replay.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from abides_core import abides
from abides_core.utils import str_to_ns
from abides_markets.agents import ExchangeAgent
from abides_markets.utils import generate_latency_model

from src.agents.lobster_replay_agent import LOBSTERReplayAgent


SYMBOL    = "GOOG"
DATE      = "2012-06-21"
FILEPATH  = "data/LOBSTER_SampleFile_GOOG_2012-06-21_10"
NUM_LEVELS = 10

# Medianoche del día en ns desde época Unix
DATE_NS   = int(pd.to_datetime(DATE).to_datetime64())
MKT_OPEN  = DATE_NS + str_to_ns("09:30:00")
MKT_CLOSE = DATE_NS + str_to_ns("16:00:00")

agents = []

agents.append(
    ExchangeAgent(
        id=0,
        name="EXCHANGE",
        type="ExchangeAgent",
        mkt_open=MKT_OPEN,
        mkt_close=MKT_CLOSE,
        symbols=[SYMBOL],
        book_logging=True,
        book_log_depth=NUM_LEVELS,
        log_orders=False,
        pipeline_delay=0,
        computation_delay=0,
        stream_history=500,
        random_state=np.random.RandomState(seed=42),
    )
)

agents.append(
    LOBSTERReplayAgent(
        id=1,
        symbol=SYMBOL,
        date=DATE,
        filepath=FILEPATH,
        num_levels=NUM_LEVELS,
        log_orders=False,
        random_state=np.random.RandomState(seed=0),
    )
)

latency_model = generate_latency_model(len(agents))

config = {
    "seed": 42,
    "start_time": DATE_NS,
    "stop_time": MKT_CLOSE + str_to_ns("1s"),
    "agents": agents,
    "agent_latency_model": latency_model,
    "default_computation_delay": 50,
    "custom_properties": {},
    "random_state_kernel": np.random.RandomState(seed=1),
    "stdout_log_level": "INFO",
}

print("Iniciando simulación LOBSTER replay...")
end_state = abides.run(config)
print("Simulación completada.")

# Mostrar el estado final del libro de órdenes
ob = end_state["agents"][0].order_books[SYMBOL]
print(f"\nBest bid: {ob.get_l1_bid_data()}")
print(f"Best ask: {ob.get_l1_ask_data()}")
