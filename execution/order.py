from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    CREATED = "created"
    SENT = "sent"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class Order:
    instrument: str
    side: OrderSide
    quantity: int
    price: float | None = None

    status: OrderStatus = OrderStatus.CREATED