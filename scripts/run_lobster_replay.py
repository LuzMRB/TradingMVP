"""
Script para ejecutar una simulación ABIDES con el LOBSTERReplayAgent.

Uso:
    python scripts/run_lobster_replay.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from abides_core import abides
from src.configs.lobster_config import build_config

config = build_config()

print("Iniciando simulación LOBSTER replay...")
end_state = abides.run(config)
print("Simulación completada.")

ob = end_state["agents"][0].order_books["GOOG"]
print(f"\nBest bid: {ob.get_l1_bid_data()}")
print(f"Best ask: {ob.get_l1_ask_data()}")
