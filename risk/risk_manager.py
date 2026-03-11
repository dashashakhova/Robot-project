class RiskManager:

    def __init__(self, capital=100000):
        self.capital = capital
        self.risk_per_trade = 0.01

    def calculate_position_size(self, price):

        risk_amount = self.capital * self.risk_per_trade

        quantity = int(risk_amount / price)

        if quantity < 1:
            quantity = 1

        return quantity