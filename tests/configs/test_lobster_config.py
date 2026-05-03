import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.agents.buy_and_hold_agent import BuyAndHoldAgent
from src.configs.lobster_config import build_config


def _mock_orderbook():
    # Columnas: ask_price, ask_size, bid_price, bid_size (un nivel)
    # open_price = (57500 + 57400) // 2 = 57450
    return pd.DataFrame([[57500, 100, 57400, 100]])


def _mock_messages():
    # Un solo mensaje LOBSTER con las columnas esperadas
    return pd.DataFrame({
        "time":      [34200.006],
        "type":      [1],
        "order_id":  [100],
        "size":      [50],
        "price":     [57500],
        "direction": [1],
    })


def _make_side_effect():
    # Call order in build_config:
    # 1. pd.read_csv(ob_file) — open_price calculation
    # 2. pd.read_csv(msg_file) — inside LOBSTERReplayAgent.__init__
    # 3. pd.read_csv(ob_file)  — inside LOBSTERReplayAgent.__init__ (if os.path.exists)
    return [_mock_orderbook(), _mock_messages(), _mock_orderbook()]


def test_build_config_has_three_agents():
    with patch("pandas.read_csv", side_effect=_make_side_effect()), \
         patch("os.path.exists", return_value=True), \
         patch("src.configs.lobster_config.generate_latency_model", return_value=MagicMock()):
        config = build_config()
    assert len(config["agents"]) == 3


def test_build_config_buy_and_hold_is_id_2():
    with patch("pandas.read_csv", side_effect=_make_side_effect()), \
         patch("os.path.exists", return_value=True), \
         patch("src.configs.lobster_config.generate_latency_model", return_value=MagicMock()):
        config = build_config()
    bah = [a for a in config["agents"] if isinstance(a, BuyAndHoldAgent)]
    assert len(bah) == 1
    assert bah[0].id == 2


def test_build_config_starting_cash_passed_to_buy_and_hold():
    with patch("pandas.read_csv", side_effect=_make_side_effect()), \
         patch("os.path.exists", return_value=True), \
         patch("src.configs.lobster_config.generate_latency_model", return_value=MagicMock()):
        config = build_config(starting_cash=5_000_000)
    bah = [a for a in config["agents"] if isinstance(a, BuyAndHoldAgent)][0]
    assert bah.starting_cash == 5_000_000


def test_build_config_default_starting_cash_is_100k_usd():
    with patch("pandas.read_csv", side_effect=_make_side_effect()), \
         patch("os.path.exists", return_value=True), \
         patch("src.configs.lobster_config.generate_latency_model", return_value=MagicMock()):
        config = build_config()
    bah = [a for a in config["agents"] if isinstance(a, BuyAndHoldAgent)][0]
    assert bah.starting_cash == 10_000_000  # = $100,000 USD (ABIDES usa centavos)
