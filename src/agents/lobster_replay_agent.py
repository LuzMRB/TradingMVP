import os
import pandas as pd
import numpy as np
from typing import Optional
from abides_core import NanosecondTime
from abides_markets.agents import TradingAgent
from abides_markets.orders import Side, LimitOrder


# Agente que reproduce eventos de LOBSTER, basado en TradingAgent de ABIDES.
# Lee un archivo CSV con mensajes de LOBSTER y los procesa para simular el comportamiento del mercado.

class LOBSTERReplayAgent(TradingAgent):

    def __init__(
        self,
        id: int,
        symbol: str,
        date: str,  # formato "2012-06-21"
        starting_cash: int = 0,
        filepath: str = "data/LOBSTER_SampleFile_GOOG_2012-06-21_10",
        num_levels: int = 10,
        log_orders: bool = False,
        random_state: Optional[np.random.RandomState] = None,
    ):
        super().__init__(id, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)

        self.symbol = symbol

        # Construir nombre del fichero
        msg_file = f"{filepath}/{symbol}_{date}_34200000_57600000_message_{num_levels}.csv"

        # Cargar mensajes LOBSTER
        self.messages = pd.read_csv(msg_file, header=None, names=[
            "time", "type", "order_id", "size", "price", "direction"
        ])

        # Convertir tiempo a nanosegundos
        self.messages["time_ns"] = (self.messages["time"] * 1e9).astype(int)

        # Ignorar tipo 7 (trading halt)
        self.messages = self.messages[self.messages["type"] != 7].reset_index(drop=True)

        # Cargar estado inicial del libro si existe el fichero orderbook
        ob_file = f"{filepath}/{symbol}_{date}_34200000_57600000_orderbook_{num_levels}.csv"
        if os.path.exists(ob_file):
            self.initial_orderbook = pd.read_csv(ob_file, header=None).iloc[0].tolist()
        else:
            self.initial_orderbook = None
        self.num_levels = num_levels

        # Índice del evento actual
        self.current_idx = 0

        # Flag para pre-poblar el libro en el primer wakeup
        self._orderbook_initialized = False

        # Diccionario para trackear órdenes activas: order_id -> LimitOrder
        self.active_orders = {}

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        # start_time es medianoche del día en ns desde época Unix.
        # Los timestamps de LOBSTER son ns desde medianoche, así que sumamos ambos.
        self._start_time = start_time
        first_event_time = self.messages.iloc[0]["time_ns"]
        self.set_wakeup(self._start_time + first_event_time)

    def wakeup(self, current_time: NanosecondTime) -> None:
        super().wakeup(current_time)

        # Pre-poblar el libro con el estado inicial en el primer wakeup
        if not self._orderbook_initialized and self.initial_orderbook is not None:
            self._orderbook_initialized = True
            for i in range(self.num_levels):
                ask_price = int(self.initial_orderbook[i * 4])
                ask_size  = int(self.initial_orderbook[i * 4 + 1])
                bid_price = int(self.initial_orderbook[i * 4 + 2])
                bid_size  = int(self.initial_orderbook[i * 4 + 3])
                if ask_price > 0 and ask_size > 0:
                    self.place_limit_order(self.symbol, ask_size, Side.ASK, ask_price)
                if bid_price > 0 and bid_size > 0:
                    self.place_limit_order(self.symbol, bid_size, Side.BID, bid_price)

        # Si hemos procesado todos los eventos, terminar
        if self.current_idx >= len(self.messages):
            return

        # Coger el evento actual
        event = self.messages.iloc[self.current_idx]

        # Procesar la orden
        self._process_event(event)

        # Avanzar al siguiente evento
        self.current_idx += 1

        # Programar el siguiente wakeup si quedan eventos
        if self.current_idx < len(self.messages):
            next_time = self.messages.iloc[self.current_idx]["time_ns"]
            self.set_wakeup(self._start_time + next_time)

    def _process_event(self, event) -> None:
        order_type = event["type"]
        size = int(event["size"])
        price = int(event["price"])
        order_id = int(event["order_id"])
        direction = Side.BID if event["direction"] == 1 else Side.ASK

        # Tipo 1: Nueva orden límite
        if order_type == 1:
            self.place_limit_order(
                self.symbol, size, direction, price, order_id=order_id
            )

        # Tipo 2: Cancelación parcial
        elif order_type == 2:
            if order_id in self.active_orders:
                self.partial_cancel_order(self.active_orders[order_id], size)

        # Tipo 3: Cancelación total
        elif order_type == 3:
            if order_id in self.active_orders:
                self.cancel_order(self.active_orders[order_id])

        # Tipos 4 y 5: Ejecuciones - las gestiona el exchange automáticamente
        elif order_type in [4, 5]:
            pass

    def order_accepted(self, order: LimitOrder) -> None:
        super().order_accepted(order)
        # Guardar la orden para poder cancelarla después
        self.active_orders[order.order_id] = order

    def order_cancelled(self, order: LimitOrder) -> None:
        super().order_cancelled(order)
        self.active_orders.pop(order.order_id, None)

    def order_executed(self, order: LimitOrder) -> None:
        super().order_executed(order)
        self.active_orders.pop(order.order_id, None)