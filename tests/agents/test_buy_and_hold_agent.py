import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from abides_core.utils import str_to_ns
from abides_markets.orders import Side

from src.agents.buy_and_hold_agent import BuyAndHoldAgent


def make_agent(starting_cash=10_000_000, open_price=57500):
    return BuyAndHoldAgent(
        id=2,
        symbol="GOOG",
        date="2012-06-21",
        open_price=open_price,
        starting_cash=starting_cash,
        random_state=np.random.RandomState(42),
    )


def test_wakeup_buys_correct_shares():
    agent = make_agent(starting_cash=10_000_000, open_price=57500)
    agent.place_market_order = MagicMock()
    with patch("abides_markets.agents.TradingAgent.wakeup"):
        agent.wakeup(0)
    # floor(10_000_000 / 57500) = 173
    agent.place_market_order.assert_called_once_with("GOOG", 173, Side.BID)


def test_wakeup_only_buys_once():
    agent = make_agent()
    agent.place_market_order = MagicMock()
    with patch("abides_markets.agents.TradingAgent.wakeup"):
        agent.wakeup(0)
        agent.wakeup(1_000)
    assert agent.place_market_order.call_count == 1


def test_no_buy_when_insufficient_cash():
    agent = make_agent(starting_cash=100, open_price=57500)
    agent.place_market_order = MagicMock()
    with patch("abides_markets.agents.TradingAgent.wakeup"):
        agent.wakeup(0)
    agent.place_market_order.assert_not_called()


def test_kernel_starting_schedules_wakeup_at_mkt_open():
    agent = make_agent()
    agent.set_wakeup = MagicMock()
    midnight_ns = int(pd.to_datetime("2012-06-21").to_datetime64())
    expected_mkt_open = midnight_ns + str_to_ns("09:30:00")
    with patch("abides_markets.agents.TradingAgent.kernel_starting"):
        agent.kernel_starting(0)
    agent.set_wakeup.assert_called_once_with(expected_mkt_open)


def test_get_wake_frequency_returns_large_value():
    agent = make_agent()
    assert agent.get_wake_frequency() == int(1e18)
