"""
Модуль для отправки уведомлений и алертов
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class AlertManager:
    """
    Менеджер уведомлений и алертов
    """

    def __init__(self, strategy_name: str):
        """
        Args:
            strategy_name: название стратегии
        """
        self.strategy_name = strategy_name

        # Настройки Telegram
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        # Настройки Email
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_user = os.getenv('EMAIL_USER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.email_recipient = os.getenv('EMAIL_RECIPIENT')

        # История отправленных уведомлений (чтобы не спамить)
        self.sent_alerts = {}  # {alert_type: last_sent_time}
        self.min_interval = {
            'signal': timedelta(minutes=15),
            'warning': timedelta(minutes=5),
            'critical': timedelta(minutes=1),
            'daily': timedelta(hours=1),
            'info': timedelta(minutes=30)
        }

        # Файл для логирования алертов
        self.alerts_file = Path(f"alerts_{strategy_name}.json")
        self.load_history()

    def send_telegram(self, message: str, parse_mode: str = 'Markdown'):
        """
        Отправляет сообщение в Telegram

        Args:
            message: текст сообщения
            parse_mode: режим разметки (Markdown или HTML)
        """
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram не настроен. Пропускаем отправку.")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            response = requests.post(url, json=data, timeout=10)

            if response.status_code == 200:
                logger.info(f"✅ Уведомление отправлено в Telegram")
                return True
            else:
                logger.error(f"❌ Ошибка Telegram: {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Ошибка отправки в Telegram: {e}")
            return False

    def send_email(self, subject: str, body: str, html: bool = False):
        """
        Отправляет email

        Args:
            subject: тема письма
            body: текст письма
            html: использовать HTML разметку
        """
        if not all([self.smtp_server, self.email_user, self.email_password, self.email_recipient]):
            logger.warning("Email не настроен. Пропускаем отправку.")
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = self.email_recipient
            msg['Subject'] = subject

            if html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.send_message(msg)
            server.quit()

            logger.info(f"✅ Email отправлен: {subject}")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка отправки email: {e}")
            return False

    def send_signal_alert(self, signal: Dict):
        """
        Отправляет уведомление о торговом сигнале

        Args:
            signal: информация о сигнале
        """
        # Проверяем интервал
        if not self._can_send('signal'):
            return

        # Формируем сообщение
        emoji = '🟢' if signal['action'] == 'BUY' else '🔴' if signal['action'] == 'SELL' else '⚪'

        message = f"""
{emoji} *Торговый сигнал - {self.strategy_name}*

📊 *Действие:* {signal['action']}
💰 *Цена:* {signal.get('current_price', 0):.2f}
📈 *Уверенность:* {signal.get('confidence', 0):.1%}

📋 *Причины:*
"""
        for reason in signal.get('reasons', []):
            message += f"  • {reason}\n"

        if 'agreement' in signal:
            message += f"\n🤝 *Согласие стратегий:* {signal['agreement']:.1%}"

        if 'market_context' in signal:
            mc = signal['market_context']
            message += f"\n\n📊 *Рыночный контекст:*"
            message += f"\n  • Режим: {mc.get('regime', 'unknown')}"
            message += f"\n  • Волатильность: {mc.get('volatility', {}).get('current_atr_pct', 0):.2f}%"

        # Отправляем
        self.send_telegram(message)

        # Логируем
        self._log_alert('signal', signal)

    def send_trade_alert(self, trade: Dict, is_open: bool = True):
        """
        Отправляет уведомление о сделке

        Args:
            trade: информация о сделке
            is_open: True если открытие, False если закрытие
        """
        # Проверяем интервал
        alert_type = 'signal'  # используем тот же интервал

        if not self._can_send(alert_type):
            return

        if is_open:
            emoji = '🟢'
            action = 'ОТКРЫТИЕ'
        else:
            emoji = '🔴'
            action = 'ЗАКРЫТИЕ'

        pnl_emoji = '✅' if trade.get('pnl', 0) > 0 else '❌' if trade.get('pnl', 0) < 0 else '⚪'

        message = f"""
{emoji} *{action} ПОЗИЦИИ - {self.strategy_name}*

📊 *Инструмент:* {trade['instrument']}
📈 *Направление:* {trade['direction']}
💰 *Цена:* {trade.get('entry_price' if is_open else 'exit_price', 0):.2f}
"""

        if not is_open:
            message += f"""
{pnl_emoji} *Результат:* 
  • PnL: {trade.get('pnl', 0):+,.0f}
  • PnL %: {trade.get('pnl_pct', 0):+.2%}
  • Причина: {trade.get('reason', 'unknown')}
"""

        # Отправляем
        self.send_telegram(message)

        # Логируем
        self._log_alert('trade', trade)

    def send_warning(self, warning_type: str, message: str, data: Dict = None):
        """
        Отправляет предупреждение

        Args:
            warning_type: тип предупреждения
            message: текст сообщения
            data: дополнительные данные
        """
        # Проверяем интервал
        if not self._can_send('warning'):
            return

        full_message = f"""
⚠️ *ПРЕДУПРЕЖДЕНИЕ - {self.strategy_name}*
*Тип:* {warning_type}

{message}
"""

        if data:
            full_message += "\n*Детали:*\n"
            for key, value in data.items():
                full_message += f"  • {key}: {value}\n"

        # Отправляем
        self.send_telegram(full_message)

        # Логируем
        self._log_alert('warning', {'type': warning_type, 'message': message, 'data': data})

    def send_critical_alert(self, alert_type: str, message: str, data: Dict = None):
        """
        Отправляет критическое оповещение (все каналы)

        Args:
            alert_type: тип оповещения
            message: текст сообщения
            data: дополнительные данные
        """
        # Проверяем интервал
        if not self._can_send('critical'):
            return

        # Telegram
        tg_message = f"""
🚨 *КРИТИЧЕСКОЕ ОПОВЕЩЕНИЕ - {self.strategy_name}*
*Тип:* {alert_type}

❗️ {message}
"""

        if data:
            tg_message += "\n*Детали:*\n"
            for key, value in data.items():
                tg_message += f"  • {key}: {value}\n"

        self.send_telegram(tg_message)

        # Email
        email_subject = f"🚨 КРИТИЧЕСКОЕ ОПОВЕЩЕНИЕ - {self.strategy_name} - {alert_type}"
        email_body = f"""
<html>
<body>
<h2>🚨 Критическое оповещение - {self.strategy_name}</h2>
<p><b>Тип:</b> {alert_type}</p>
<p><b>Время:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p><b>Сообщение:</b> {message}</p>
"""
        if data:
            email_body += "<h3>Детали:</h3><ul>"
            for key, value in data.items():
                email_body += f"<li><b>{key}:</b> {value}</li>"
            email_body += "</ul>"

        email_body += """
</body>
</html>
"""

        self.send_email(email_subject, email_body, html=True)

        # Логируем
        self._log_alert('critical', {'type': alert_type, 'message': message, 'data': data})

    def send_daily_report(self, stats: Dict):
        """
        Отправляет ежедневный отчет

        Args:
            stats: статистика за день
        """
        # Проверяем интервал
        if not self._can_send('daily'):
            return

        message = f"""
📊 *ЕЖЕДНЕВНЫЙ ОТЧЕТ - {self.strategy_name}*
📅 {datetime.now().strftime('%Y-%m-%d')}

📈 *Результаты дня:*
  • PnL: {stats.get('daily_pnl', 0):+,.0f}
  • Сделок: {stats.get('daily_trades', 0)}
  • Винрейт: {stats.get('daily_win_rate', 0):.1%}

📊 *Общая статистика:*
  • Капитал: {stats.get('current_capital', 0):,.0f}
  • Общая доходность: {stats.get('total_return', 0):+.2%}
  • Винрейт: {stats.get('win_rate', 0):.1%}
  • Profit Factor: {stats.get('profit_factor', 0):.2f}
  • Max DD: {stats.get('max_drawdown', 0):.2%}

🎯 *На сегодня:*
  • Открытых позиций: {stats.get('open_positions', 0)}
  • Сигналов получено: {stats.get('signals_today', 0)}
"""

        # Отправляем
        self.send_telegram(message)

        # Логируем
        self._log_alert('daily_report', stats)

    def send_test_message(self):
        """Отправляет тестовое сообщение для проверки настроек"""
        message = f"""
🧪 *ТЕСТОВОЕ СООБЩЕНИЕ - {self.strategy_name}*

Система уведомлений работает корректно.
Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        success_tg = self.send_telegram(message)

        if success_tg:
            logger.info("✅ Тестовое сообщение отправлено")
        else:
            logger.error("❌ Ошибка отправки тестового сообщения")

    def _can_send(self, alert_type: str) -> bool:
        """
        Проверяет, можно ли отправить уведомление (не чаще min_interval)

        Args:
            alert_type: тип уведомления

        Returns:
            True если можно отправить
        """
        if alert_type not in self.min_interval:
            return True

        last_sent = self.sent_alerts.get(alert_type)
        if last_sent is None:
            return True

        time_diff = datetime.now() - last_sent
        return time_diff >= self.min_interval[alert_type]

    def _log_alert(self, alert_type: str, data: Any):
        """
        Логирует отправленное уведомление

        Args:
            alert_type: тип уведомления
            data: данные уведомления
        """
        self.sent_alerts[alert_type] = datetime.now()

        # Сохраняем в историю
        self.alerts_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'data': data
        })

        # Ограничиваем размер истории
        if len(self.alerts_history) > 1000:
            self.alerts_history = self.alerts_history[-1000:]

        self.save_history()

    def load_history(self):
        """Загружает историю уведомлений из файла"""
        self.alerts_history = []

        if not self.alerts_file.exists():
            return

        try:
            with open(self.alerts_file, 'r') as f:
                data = json.load(f)

            self.alerts_history = data.get('history', [])

            # Восстанавливаем last_sent
            for alert in self.alerts_history[-10:]:  # только последние 10
                alert_type = alert.get('type')
                timestamp = datetime.fromisoformat(alert['timestamp'])
                self.sent_alerts[alert_type] = max(
                    self.sent_alerts.get(alert_type, datetime.min),
                    timestamp
                )

        except Exception as e:
            logger.error(f"Ошибка загрузки истории уведомлений: {e}")

    def save_history(self):
        """Сохраняет историю уведомлений в файл"""
        try:
            with open(self.alerts_file, 'w') as f:
                json.dump({
                    'strategy': self.strategy_name,
                    'history': self.alerts_history[-500:]  # последние 500
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Ошибка сохранения истории уведомлений: {e}")