from typing import List
from .order import Order, OrderStatus


class OrderManager:

    def __init__(self):
        self.orders: List[Order] = []

    def create_order(self, instrument, side, quantity, price=None):

        order = Order(
            instrument=instrument,
            side=side,
            quantity=quantity,
            price=price
        )

        self.orders.append(order)

        return order

    def mark_filled(self, order: Order):
        order.status = OrderStatus.FILLED

    def get_active_orders(self):
        return [o for o in self.orders if o.status != OrderStatus.FILLED]