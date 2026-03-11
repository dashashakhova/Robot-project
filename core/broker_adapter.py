class BrokerAdapter:

    def __init__(self, client, account_id):
        self.client = client
        self.account_id = account_id

    def place_order(self, instrument, side, quantity, price=None):

        print(f"EXECUTING ORDER: {side} {quantity} {instrument}")

        # Пока просто имитация исполнения
        return True