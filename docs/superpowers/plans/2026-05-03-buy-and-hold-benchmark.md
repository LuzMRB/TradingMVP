# Buy & Hold Benchmark Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `BuyAndHoldAgent` to TradingMVP as a B&H benchmark that buys maximum shares at market open and holds to close, with P&L tracked by ABIDES' existing `kernel_stopping` infrastructure.

**Architecture:** `BuyAndHoldAgent` inherits from `TradingAgent` (same as `LOBSTERReplayAgent`). It overrides `get_wake_frequency()` → `int(1e18)` to suppress default wakeups, schedules a single wakeup at `mkt_open` in `kernel_starting`, and issues one `place_market_order` in `wakeup`. `lobster_config.py` gains a `starting_cash` parameter and includes the agent as `id=2`.

**Tech Stack:** Python 3, ABIDES (`abides-markets`, `abides-core`), NumPy, Pandas, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/agents/buy_and_hold_agent.py` | Create | `BuyAndHoldAgent` class |
| `src/configs/lobster_config.py` | Modify | Add `starting_cash` param + `BuyAndHoldAgent` as `id=2` |
| `tests/__init__.py` | Create | Test package marker |
| `tests/agents/__init__.py` | Create | Test package marker |
| `tests/agents/test_buy_and_hold_agent.py` | Create | Unit tests for `BuyAndHoldAgent` |
| `tests/configs/__init__.py` | Create | Test package marker |
| `tests/configs/test_lobster_config.py` | Create | Tests for `build_config` changes |

---

### Task 1: `BuyAndHoldAgent` class

**Files:**
- Create: `src/agents/buy_and_hold_agent.py`
- Create: `tests/__init__.py`
- Create: `tests/agents/__init__.py`
- Create: `tests/agents/test_buy_and_hold_agent.py`

- [ ] **Step 1: Create test package stubs**

Create two empty files:
- `tests/__init__.py` (empty)
- `tests/agents/__init__.py` (empty)

- [ ] **Step 2: Write failing tests**

Create `tests/agents/test_buy_and_hold_agent.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

```
pytest tests/agents/test_buy_and_hold_agent.py -v
```

Expected: `ImportError` — `No module named 'src.agents.buy_and_hold_agent'`

- [ ] **Step 4: Implement `BuyAndHoldAgent`**

Create `src/agents/buy_and_hold_agent.py`:

```python
import numpy as np
import pandas as pd
from typing import Optional
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_markets.agents import TradingAgent
from abides_markets.orders import Side


class BuyAndHoldAgent(TradingAgent):

    def __init__(
        self,
        id: int,
        symbol: str,
        date: str,          # formato "2012-06-21"
        open_price: int,    # precio de apertura en centavos
        starting_cash: int, # capital inicial en centavos
        random_state: Optional[np.random.RandomState] = None,
    ):
        super().__init__(id, starting_cash=starting_cash, random_state=random_state)
        self.symbol = symbol
        self._open_price = open_price
        self._midnight_ns = int(pd.to_datetime(date).to_datetime64())
        self._bought = False

    def get_wake_frequency(self) -> NanosecondTime:
        return int(1e18)

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        self.set_wakeup(self._midnight_ns + str_to_ns("09:30:00"))

    def wakeup(self, current_time: NanosecondTime) -> None:
        super().wakeup(current_time)
        if self._bought:
            return
        shares = self.starting_cash // self._open_price
        if shares > 0:
            self.place_market_order(self.symbol, shares, Side.BID)
        self._bought = True
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/agents/test_buy_and_hold_agent.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agents/buy_and_hold_agent.py tests/__init__.py tests/agents/__init__.py tests/agents/test_buy_and_hold_agent.py
git commit -m "feat: add BuyAndHoldAgent benchmark"
```

---

### Task 2: Update `lobster_config.py`

**Files:**
- Modify: `src/configs/lobster_config.py`
- Create: `tests/configs/__init__.py`
- Create: `tests/configs/test_lobster_config.py`

- [ ] **Step 1: Create test package stub**

Create empty file: `tests/configs/__init__.py`

- [ ] **Step 2: Write failing tests**

Create `tests/configs/test_lobster_config.py`:

```python
import pandas as pd
import pytest
from unittest.mock import patch

from src.agents.buy_and_hold_agent import BuyAndHoldAgent
from src.configs.lobster_config import build_config


def _mock_csv():
    # Columnas: ask_price, ask_size, bid_price, bid_size (un nivel)
    # open_price = (57500 + 57400) // 2 = 57450
    return pd.DataFrame([[57500, 100, 57400, 100]])


def test_build_config_has_three_agents():
    with patch("pandas.read_csv", return_value=_mock_csv()):
        config = build_config()
    assert len(config["agents"]) == 3


def test_build_config_buy_and_hold_is_id_2():
    with patch("pandas.read_csv", return_value=_mock_csv()):
        config = build_config()
    bah = [a for a in config["agents"] if isinstance(a, BuyAndHoldAgent)]
    assert len(bah) == 1
    assert bah[0].id == 2


def test_build_config_starting_cash_passed_to_buy_and_hold():
    with patch("pandas.read_csv", return_value=_mock_csv()):
        config = build_config(starting_cash=5_000_000)
    bah = [a for a in config["agents"] if isinstance(a, BuyAndHoldAgent)][0]
    assert bah.starting_cash == 5_000_000


def test_build_config_default_starting_cash_is_100k_usd():
    with patch("pandas.read_csv", return_value=_mock_csv()):
        config = build_config()
    bah = [a for a in config["agents"] if isinstance(a, BuyAndHoldAgent)][0]
    assert bah.starting_cash == 10_000_000  # $100k in cents
```

- [ ] **Step 3: Run tests to verify they fail**

```
pytest tests/configs/test_lobster_config.py -v
```

Expected: FAIL — `build_config()` got unexpected keyword argument `starting_cash` and agents list has 2 elements.

- [ ] **Step 4: Update `src/configs/lobster_config.py`**

**4a.** Add the import after the existing `LOBSTERReplayAgent` import (line 23):

```python
from src.agents.buy_and_hold_agent import BuyAndHoldAgent
```

**4b.** Add `starting_cash` parameter to `build_config` signature (line 35–42):

```python
def build_config(
    symbol: str = "GOOG",
    date: str = "2012-06-21",
    filepath: str = "data/LOBSTER_SampleFile_GOOG_2012-06-21_10",
    num_levels: int = 10,
    seed: int = 42,
    stdout_log_level: str = "INFO",
    starting_cash: int = 10_000_000,
):
```

**4c.** Append `BuyAndHoldAgent` to the `agents` list, after `LOBSTERReplayAgent` (after line 79):

```python
        BuyAndHoldAgent(
            id=2,
            symbol=symbol,
            date=date,
            open_price=open_price,
            starting_cash=starting_cash,
            random_state=np.random.RandomState(seed=2),
        ),
```

(`generate_latency_model(len(agents), ...)` already uses `len(agents)` dynamically — no change needed there.)

- [ ] **Step 5: Run Task 2 tests to verify they pass**

```
pytest tests/configs/test_lobster_config.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 6: Run full test suite**

```
pytest tests/ -v
```

Expected: 9 tests PASS (5 from Task 1 + 4 from Task 2).

- [ ] **Step 7: Commit**

```bash
git add src/configs/lobster_config.py tests/configs/__init__.py tests/configs/test_lobster_config.py
git commit -m "feat: add BuyAndHoldAgent to lobster_config with starting_cash param"
```
