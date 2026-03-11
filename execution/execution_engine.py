from .order_manager import OrderManager
from .order import OrderSide


class ExecutionEngine:

    def __init__(self, broker, risk_manager):

        self.broker = broker
        self.risk_manager = risk_manager
        self.order_manager = OrderManager()

    def execute_signal(self, instrument, signal, price):

        if signal == "buy":
            side = OrderSide.BUY
        elif signal == "sell":
            side = OrderSide.SELL
        else:
            return

        quantity = self.risk_manager.calculate_position_size(price)

        order = self.order_manager.create_order(
            instrument=instrument,
            side=side,
            quantity=quantity,
            price=price
        )

        self.send_order(order)

    def send_order(self, order):

        result = self.broker.place_order(
            instrument=order.instrument,
            side=order.side.value,
            quantity=order.quantity,
            price=order.price
        )

        if result:
            self.order_manager.mark_filled(order)