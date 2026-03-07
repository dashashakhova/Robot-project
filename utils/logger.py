"""
Модуль настройки логирования
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logging(name: str = 'trading_robot', log_level: str = 'INFO'):
    """
    Настраивает логирование

    Args:
        name: имя логгера
        log_level: уровень логирования

    Returns:
        Настроенный логгер
    """
    # Создаем директорию для логов
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Имя файла с датой
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"

    # Настройка форматирования
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Хендлер для файла (с ротацией)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    # Хендлер для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Настройка корневого логгера
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Удаляем существующие хендлеры
    root_logger.handlers = []

    # Добавляем новые хендлеры
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Создаем и возвращаем логгер для модуля
    logger = logging.getLogger(name)
    logger.info(f"📝 Логирование настроено. Файл: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Возвращает логгер для модуля

    Args:
        name: имя модуля

    Returns:
        Логгер
    """
    return logging.getLogger(name)