import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, call

from abides_core.utils import str_to_ns
from abides_markets.orders import Side, LimitOrder

from src.agents.lobster_replay_agent import LOBSTERReplayAgent


MIDNIGHT_NS = int(pd.to_datetime("2012-06-21").to_datetime64())

# Tiempos en segundos desde medianoche (formato real LOBSTER: decimal con precisión µs)
# 34200.006 = 9:30:00.006 AM
MOCK_MESSAGES = pd.DataFrame({
    "time":      [34200.006, 34200.015, 34200.027, 34200.030, 34200.040],
    "type":      [1,         2,         3,         4,         1        ],
    "order_id":  [100,       100,       200,       300,       400      ],
    "size":      [50,        10,        100,       75,        200      ],
    "price":     [57500,     0,         57400,     57500,     57450    ],
    "direction": [1,         1,         -1,        -1,        1        ],
})

# 1 nivel de orderbook: ask_price, ask_size, bid_price, bid_size
MOCK_ORDERBOOK_ROW = [57500, 100, 57400, 100]


def make_agent(with_orderbook=True, num_levels=1, extra_messages=None):
    messages = MOCK_MESSAGES.copy()
    if extra_messages is not None:
        messages = pd.concat([messages, extra_messages], ignore_index=True)

    ob_df = pd.DataFrame([MOCK_ORDERBOOK_ROW])

    with patch("pandas.read_csv") as mock_csv, \
         patch("os.path.exists", return_value=with_orderbook):
        mock_csv.side_effect = [messages, ob_df] if with_orderbook else [messages]
        agent = LOBSTERReplayAgent(
            id=1,
            symbol="GOOG",
            date="2012-06-21",
            filepath="fake/path",
            num_levels=num_levels,
            random_state=np.random.RandomState(42),
        )
    return agent


# ── Inicialización ────────────────────────────────────────────────────────────

def test_init_converts_timestamps_to_nanoseconds():
    agent = make_agent()
    expected_ns = int(MOCK_MESSAGES["time"].iloc[0] * 1e9)
    assert agent.messages["time_ns"].iloc[0] == expected_ns


def test_init_filters_halt_events():
    halt_row = pd.DataFrame({
        "time": [34200050], "type": [7], "order_id": [999],
        "size": [0], "price": [0], "direction": [0],
    })
    agent = make_agent(extra_messages=halt_row)
    assert 7 not in agent.messages["type"].values


def test_init_loads_orderbook_when_file_exists():
    agent = make_agent(with_orderbook=True)
    assert agent.initial_orderbook == MOCK_ORDERBOOK_ROW


def test_init_no_orderbook_when_file_missing():
    agent = make_agent(with_orderbook=False)
    assert agent.initial_orderbook is None


# ── Wake frequency ────────────────────────────────────────────────────────────

def test_get_wake_frequency_suppresses_default_wakeups():
    agent = make_agent()
    assert agent.get_wake_frequency() == int(1e18)


# ── kernel_starting ───────────────────────────────────────────────────────────

def test_kernel_starting_schedules_first_event_wakeup():
    agent = make_agent()
    agent.set_wakeup = MagicMock()
    first_event_ns = int(MOCK_MESSAGES["time"].iloc[0] * 1e9)
    expected_wakeup = MIDNIGHT_NS + first_event_ns

    with patch("abides_markets.agents.TradingAgent.kernel_starting"):
        agent.kernel_starting(0)

    agent.set_wakeup.assert_called_once_with(expected_wakeup)


# ── wakeup: inicialización del orderbook ─────────────────────────────────────

def test_wakeup_initializes_orderbook_on_first_call():
    agent = make_agent(with_orderbook=True, num_levels=1)
    agent.place_limit_order = MagicMock()
    agent.set_wakeup = MagicMock()

    with patch("abides_markets.agents.TradingAgent.wakeup"):
        agent.wakeup(0)

    # 1 nivel → 1 ask + 1 bid = 2 llamadas de orderbook + 1 del primer evento (type 1)
    ob_calls = [
        call("GOOG", 100, Side.ASK, 57500),
        call("GOOG", 100, Side.BID, 57400),
    ]
    agent.place_limit_order.assert_any_call("GOOG", 100, Side.ASK, 57500)
    agent.place_limit_order.assert_any_call("GOOG", 100, Side.BID, 57400)


def test_wakeup_initializes_orderbook_only_once():
    agent = make_agent(with_orderbook=True, num_levels=1)
    agent.place_limit_order = MagicMock()
    agent.set_wakeup = MagicMock()

    with patch("abides_markets.agents.TradingAgent.wakeup"):
        agent.wakeup(0)
        # 1 nivel → 2 llamadas de orderbook (ASK + BID) + 1 del evento type=1 = 3
        assert agent.place_limit_order.call_count == 3

        agent.wakeup(1)
        # Segundo wakeup: orderbook ya inicializado; evento type=2 no llama place_limit_order
        assert agent.place_limit_order.call_count == 3


# ── wakeup: procesado de eventos ─────────────────────────────────────────────

def test_wakeup_type1_places_limit_order():
    agent = make_agent(with_orderbook=False)
    agent.place_limit_order = MagicMock()
    agent.set_wakeup = MagicMock()
    # Primer evento: type=1, order_id=100, size=50, price=57500, direction=1 (BID)

    with patch("abides_markets.agents.TradingAgent.wakeup"):
        agent.wakeup(0)

    agent.place_limit_order.assert_called_once_with("GOOG", 50, Side.BID, 57500, order_id=100)


def test_wakeup_type2_partial_cancel_active_order():
    agent = make_agent(with_orderbook=False)
    agent.place_limit_order = MagicMock()
    agent.set_wakeup = MagicMock()
    agent.partial_cancel_order = MagicMock()

    mock_order = MagicMock()
    agent.active_orders[100] = mock_order
    # Evento idx=1: type=2, order_id=100, size=10
    agent.current_idx = 1

    with patch("abides_markets.agents.TradingAgent.wakeup"):
        agent.wakeup(0)

    agent.partial_cancel_order.assert_called_once_with(mock_order, 10)


def test_wakeup_type3_cancels_active_order():
    agent = make_agent(with_orderbook=False)
    agent.place_limit_order = MagicMock()
    agent.set_wakeup = MagicMock()
    agent.cancel_order = MagicMock()

    mock_order = MagicMock()
    agent.active_orders[200] = mock_order
    # Evento idx=2: type=3, order_id=200
    agent.current_idx = 2

    with patch("abides_markets.agents.TradingAgent.wakeup"):
        agent.wakeup(0)

    agent.cancel_order.assert_called_once_with(mock_order)


def test_wakeup_type4_ignored():
    agent = make_agent(with_orderbook=False)
    agent.place_limit_order = MagicMock()
    agent.cancel_order = MagicMock()
    agent.set_wakeup = MagicMock()
    # Evento idx=3: type=4
    agent.current_idx = 3

    with patch("abides_markets.agents.TradingAgent.wakeup"):
        agent.wakeup(0)

    agent.place_limit_order.assert_not_called()
    agent.cancel_order.assert_not_called()


def test_wakeup_schedules_next_event():
    agent = make_agent(with_orderbook=False)
    agent.place_limit_order = MagicMock()
    agent.set_wakeup = MagicMock()

    with patch("abides_markets.agents.TradingAgent.wakeup"):
        agent.wakeup(0)

    # Después del primer evento (idx=0), debe programar el wakeup del idx=1
    next_event_ns = int(MOCK_MESSAGES["time"].iloc[1] * 1e9)
    expected_next = MIDNIGHT_NS + next_event_ns
    agent.set_wakeup.assert_called_once_with(expected_next)


def test_wakeup_stops_after_last_event():
    agent = make_agent(with_orderbook=False)
    agent.place_limit_order = MagicMock()
    agent.set_wakeup = MagicMock()
    agent.current_idx = len(agent.messages) - 1  # último evento

    with patch("abides_markets.agents.TradingAgent.wakeup"):
        agent.wakeup(0)

    agent.set_wakeup.assert_not_called()


# ── Callbacks de órdenes ─────────────────────────────────────────────────────

def test_order_accepted_stores_in_active_orders():
    agent = make_agent()
    mock_order = MagicMock()
    mock_order.order_id = 42

    with patch("abides_markets.agents.TradingAgent.order_accepted"):
        agent.order_accepted(mock_order)

    assert agent.active_orders[42] is mock_order


def test_order_cancelled_removes_from_active_orders():
    agent = make_agent()
    mock_order = MagicMock()
    mock_order.order_id = 42
    agent.active_orders[42] = mock_order

    with patch("abides_markets.agents.TradingAgent.order_cancelled"):
        agent.order_cancelled(mock_order)

    assert 42 not in agent.active_orders


def test_order_executed_removes_from_active_orders():
    agent = make_agent()
    mock_order = MagicMock()
    mock_order.order_id = 42
    agent.active_orders[42] = mock_order

    with patch("abides_markets.agents.TradingAgent.order_executed"):
        agent.order_executed(mock_order)

    assert 42 not in agent.active_orders


def test_cancel_unknown_order_does_not_raise():
    agent = make_agent()
    mock_order = MagicMock()
    mock_order.order_id = 999  # no está en active_orders

    with patch("abides_markets.agents.TradingAgent.order_cancelled"):
        agent.order_cancelled(mock_order)  # no debe lanzar excepción
