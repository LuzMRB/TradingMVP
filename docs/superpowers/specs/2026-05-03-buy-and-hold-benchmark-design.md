# Buy & Hold Benchmark Agent — Design Spec

**Date:** 2026-05-03  
**Status:** Approved  

---

## Goal

Add a `BuyAndHoldAgent` to the TradingMVP project that serves as a simple benchmark for comparing against the RL trading model. The agent buys as many shares as possible at market open and holds until market close, relying on ABIDES' existing P&L infrastructure for final metrics.

---

## Architecture

### New file: `src/agents/buy_and_hold_agent.py`

```
BuyAndHoldAgent(TradingAgent)
  └─ kernel_starting(start_time)  → requests wakeup at mkt_open
  └─ wakeup(current_time)         → place_market_order(BUY, shares)
  └─ kernel_stopping()            → inherited; prints final P&L
```

**Lifecycle:**

1. `kernel_starting()` — schedules a single wakeup at `self.mkt_open`.
2. `wakeup()` — fires once at market open. Reads `open_price` from `self.last_trade[self.symbol]` (published by the exchange via `LOBSTEROracle` at startup). Computes `shares = floor(self.starting_cash / open_price)` and calls `place_market_order(symbol, shares, Side.BID)`. Does nothing after this.
3. `kernel_stopping()` — inherited from `TradingAgent`. Automatically computes `gain = mark_to_market(holdings) - starting_cash` (in cents) and logs it to console and `kernel.mean_result_by_agent_type`.

**Constructor parameters:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `id` | int | required | ABIDES agent ID |
| `symbol` | str | required | Ticker to buy (e.g. `"GOOG"`) |
| `starting_cash` | int | required | Capital in cents (same as RL agent) |
| `random_state` | np.RandomState | None | For ABIDES compatibility |

---

## Integration: `src/configs/lobster_config.py`

- `build_config()` gains a new parameter: `starting_cash: int = 10_000_000` (= $100k in cents).
- `BuyAndHoldAgent` is added as agent `id=2` in the agents list.
- Both the future RL agent and `BuyAndHoldAgent` receive the same `starting_cash` so P&L comparisons are on equal footing.

---

## P&L Output

No additional metrics code is needed. At simulation end, the inherited `kernel_stopping()` outputs:

- `gain` in cents (total dollar P&L)
- Result stored in `kernel.mean_result_by_agent_type["BuyAndHoldAgent"]`

This is sufficient for a baseline benchmark comparison.

---

## Out of Scope

- Intraday equity curve / time-series tracking
- Sharpe ratio, max drawdown, volatility
- Multi-day or multi-symbol B&H
- Transaction cost modeling

These can be added later once the RL agent is operational and a richer comparison is needed.

---

## Files Changed

| File | Change |
|------|--------|
| `src/agents/buy_and_hold_agent.py` | **New** — `BuyAndHoldAgent` class |
| `src/configs/lobster_config.py` | **Modified** — add `starting_cash` param, add agent `id=2` |
