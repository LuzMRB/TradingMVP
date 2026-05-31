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
        shares = self.holdings["CASH"] // self._open_price
        if shares > 0:
            self.place_market_order(self.symbol, shares, Side.BID)
        self._bought = True
