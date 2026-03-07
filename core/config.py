"""
Конфигурационный файл проекта
"""
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
import json

# Базовая директория проекта
BASE_DIR = Path(__file__).parent.parent

# Директории
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# Создаем директории
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, CONFIG_DIR]:
    dir_path.mkdir(exist_ok=True)

# Файлы
ENV_FILE = BASE_DIR / ".env"
SANDBOX_ACCOUNT_FILE = BASE_DIR / ".sandbox_account"
STRATEGY_CONFIG_FILE = CONFIG_DIR / "strategy_config.json"

# Тестовые FIGI для разных инструментов
TEST_FIGIS = {
    "SBER": "BBG004730N88",  # Сбербанк
    "VTBR": "BBG004730ZJ9",  # ВТБ
    "GAZP": "BBG004730RP0",  # Газпром
    "LKOH": "BBG004731032",  # Лукойл
    "YNDX": "BBG006L8G4H1",  # Яндекс
}

# Настройки по умолчанию
DEFAULT_CONFIG = {
    "trading": {
        "initial_capital": 1000000,
        "max_position_size_pct": 0.25,  # 25% от капитала на одну позицию
        "max_portfolio_risk_pct": 0.10,  # 10% максимальный риск портфеля
        "commission": 0.001,  # 0.1%
        "slippage": 0.0005,  # 0.05% проскальзывание
    },
    "risk": {
        "max_risk_per_trade": 0.02,  # 2% риска на сделку
        "max_daily_loss_pct": 0.05,  # 5% максимальный дневной убыток
        "max_consecutive_losses": 3,  # Макс. последовательных убытков
        "min_risk_reward": 1.5,  # Минимальное соотношение риск/прибыль
    },
    "strategy": {
        "forecast_hours": [1, 2, 4],  # Прогнозы на разные горизонты
        "min_confidence": 0.6,  # Минимальная уверенность для входа
        "train_days": 60,  # Дней для обучения
        "retrain_hours": 24,  # Переобучение каждые N часов
    },
    "monitoring": {
        "check_interval_minutes": 5,
        "signal_cooldown_minutes": 30,
        "min_trades_for_stats": 20,
    }
}


class Config:
    """Класс для управления конфигурацией"""

    def __init__(self):
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию из файла или создает дефолтную"""
        if STRATEGY_CONFIG_FILE.exists():
            with open(STRATEGY_CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            self.save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()

    def save_config(self, config: Dict[str, Any] = None):
        """Сохраняет конфигурацию"""
        if config is None:
            config = self.config
        with open(STRATEGY_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

    def get(self, *keys, default=None):
        """Получает значение по цепочке ключей"""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    def update(self, *keys, value):
        """Обновляет значение по цепочке ключей"""
        d = self.config
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value
        self.save_config()


def get_sandbox_account_id():
    """Читает сохраненный ID счета из файла"""
    if SANDBOX_ACCOUNT_FILE.exists():
        with open(SANDBOX_ACCOUNT_FILE, 'r') as f:
            return f.read().strip()
    return None


def get_current_time():
    """Возвращает текущее время в UTC"""
    return datetime.now(timezone.utc)


# Глобальный экземпляр конфигурации
config = Config()