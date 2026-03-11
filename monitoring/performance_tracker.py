"""
Модуль для отслеживания производительности торговой системы в реальном времени
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceTracker:
    """
    Отслеживает производительность торговой системы в реальном времени
    """

    def __init__(self, initial_capital: float, strategy_name: str):
        """
        Args:
            initial_capital: начальный капитал
            strategy_name: название стратегии
        """
        self.initial_capital = initial_capital
        self.strategy_name = strategy_name

        # История
        self.equity_curve = deque(maxlen=10000)  # [(timestamp, equity)]
        self.trades = deque(maxlen=1000)
        self.daily_stats = defaultdict(list)

        # Текущее состояние
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0

        # Метрики
        self.metrics = {
            'total_return': 0.0,
            'daily_return': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'total_trades': 0,
            'avg_holding_time': 0.0
        }

        # Ежедневная статистика
        self.today_pnl = 0.0
        self.today_trades = 0
        self.today_date = datetime.now().date()

        # История метрик для графиков
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))

        # Файл для сохранения
        self.history_file = Path(f"performance_{strategy_name}.json")
        self.load_history()

    def update_equity(self, current_equity: float, timestamp: Optional[datetime] = None):
        """
        Обновляет кривую капитала

        Args:
            current_equity: текущая стоимость портфеля
            timestamp: временная метка
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.equity_curve.append((timestamp, current_equity))
        self.current_capital = current_equity

        # Обновляем пик
        if current_equity > self.peak_capital:
            self.peak_capital = current_equity

        # Рассчитываем просадку
        self.current_drawdown = (self.peak_capital - current_equity) / self.peak_capital

        # Обновляем метрики
        self._update_metrics()

    def add_trade(self, trade: Dict):
        """
        Добавляет сделку в историю

        Args:
            trade: информация о сделке
        """
        trade = self._normalize_trade(trade)
        if trade is None:
            return

        self.trades.append(trade)

        # Обновляем дневную статистику
        trade_date = trade['exit_time'].date()
        if trade_date == self.today_date:
            self.today_pnl += trade['pnl']
            self.today_trades += 1
        else:
            # Сохраняем статистику за предыдущий день
            if self.today_trades > 0:
                self.daily_stats[self.today_date] = {
                    'pnl': self.today_pnl,
                    'trades': self.today_trades,
                    'win_rate': self._calculate_daily_win_rate(self.today_date)
                }

            # Начинаем новый день
            self.today_date = trade_date
            self.today_pnl = trade['pnl']
            self.today_trades = 1

        # Обновляем метрики
        self._update_metrics()

    def _update_metrics(self):
        """Обновляет все метрики"""
        if len(self.trades) == 0:
            return

        trades_df = pd.DataFrame(list(self.trades))

        # Основные метрики
        self.metrics['total_trades'] = len(trades_df)
        self.metrics['total_return'] = (self.current_capital - self.initial_capital) / self.initial_capital

        # Прибыльные и убыточные сделки
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] <= 0]

        self.metrics['win_rate'] = len(winning) / len(trades_df) if len(trades_df) > 0 else 0
        self.metrics['avg_win'] = winning['pnl_pct'].mean() if len(winning) > 0 else 0
        self.metrics['avg_loss'] = losing['pnl_pct'].mean() if len(losing) > 0 else 0

        # Profit factor
        total_profit = winning['pnl'].sum() if len(winning) > 0 else 0
        total_loss = abs(losing['pnl'].sum()) if len(losing) > 0 else 0
        self.metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')

        # Коэффициент Шарпа
        if len(self.equity_curve) >= 2:
            returns = pd.Series([v for _, v in self.equity_curve]).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                self.metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252 * 24)

        # Максимальная просадка
        if len(self.equity_curve) > 0:
            equity_values = [v for _, v in self.equity_curve]
            peak = equity_values[0]
            max_dd = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            self.metrics['max_drawdown'] = max_dd

        # Среднее время удержания
        if 'holding_period' in trades_df.columns:
            self.metrics['avg_holding_time'] = trades_df['holding_period'].mean()

        # Сохраняем историю метрик
        timestamp = datetime.now()
        for key, value in self.metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append((timestamp, value))

    def _calculate_daily_win_rate(self, date: datetime.date) -> float:
        """Рассчитывает винрейт за конкретный день"""
        day_trades = [t for t in self.trades if t['exit_time'].date() == date]
        if not day_trades:
            return 0.0
        wins = sum(1 for t in day_trades if t.get('pnl', 0) > 0)
        return wins / len(day_trades)

    @staticmethod
    def _normalize_trade(trade: Dict) -> Optional[Dict]:
        """Нормализует формат сделки под единый контракт."""
        if not isinstance(trade, dict):
            return None

        entry_time = trade.get('entry_time')
        exit_time = trade.get('exit_time')
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time)

        if not isinstance(entry_time, datetime) or not isinstance(exit_time, datetime):
            return None

        trade_type = trade.get('type')
        direction = trade.get('direction')
        if direction is None:
            if trade_type == 'BUY':
                direction = 'LONG'
            elif trade_type == 'SELL':
                direction = 'SHORT'
            else:
                direction = 'UNKNOWN'

        pnl = trade.get('pnl')
        if pnl is None:
            pnl_pct = trade.get('pnl_pct', trade.get('profit', 0))
            # Если абсолютный PnL не передан, сохраняем 0, но процент оставляем
            pnl = 0
        else:
            pnl_pct = trade.get('pnl_pct', trade.get('profit', 0))

        try:
            pnl = float(pnl)
        except (TypeError, ValueError):
            pnl = 0.0

        try:
            pnl_pct = float(pnl_pct)
        except (TypeError, ValueError):
            pnl_pct = 0.0

        return {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'instrument': trade.get('instrument', 'UNKNOWN'),
            'direction': direction,
            'entry_price': float(trade.get('entry_price', 0) or 0),
            'exit_price': float(trade.get('exit_price', 0) or 0),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': trade.get('reason', 'unknown'),
            'holding_period': trade.get('holding_period')
        }

    def get_daily_stats(self, days: int = 30) -> pd.DataFrame:
        """
        Возвращает дневную статистику за последние N дней

        Args:
            days: количество дней

        Returns:
            DataFrame с дневной статистикой
        """
        cutoff = datetime.now().date() - timedelta(days=days)
        recent_days = {d: s for d, s in self.daily_stats.items() if d >= cutoff}

        if not recent_days:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(recent_days, orient='index')
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        return df

    def get_weekly_stats(self) -> pd.DataFrame:
        """
        Возвращает недельную статистику

        Returns:
            DataFrame с недельной статистикой
        """
        daily = self.get_daily_stats(days=90)
        if daily.empty:
            return pd.DataFrame()

        weekly = daily.resample('W').agg({
            'pnl': 'sum',
            'trades': 'sum',
            'win_rate': 'mean'
        })

        return weekly

    def get_monthly_stats(self) -> pd.DataFrame:
        """
        Возвращает месячную статистику

        Returns:
            DataFrame с месячной статистикой
        """
        daily = self.get_daily_stats(days=365)
        if daily.empty:
            return pd.DataFrame()

        monthly = daily.resample('M').agg({
            'pnl': 'sum',
            'trades': 'sum',
            'win_rate': 'mean'
        })

        return monthly

    def check_health(self) -> Dict:
        """
        Проверяет здоровье торговой системы

        Returns:
            Словарь с результатами проверки
        """
        warnings = []
        is_critical = False

        # Проверка просадки
        if self.current_drawdown > 0.1:
            warnings.append(f"High drawdown: {self.current_drawdown:.1%}")
            if self.current_drawdown > 0.15:
                is_critical = True

        # Проверка винрейта
        if self.metrics['win_rate'] < 0.4 and self.metrics['total_trades'] > 20:
            warnings.append(f"Low win rate: {self.metrics['win_rate']:.1%}")
            if self.metrics['win_rate'] < 0.3:
                is_critical = True

        # Проверка profit factor
        if self.metrics['profit_factor'] < 1.0 and self.metrics['total_trades'] > 20:
            warnings.append(f"Profit factor below 1.0: {self.metrics['profit_factor']:.2f}")
            is_critical = True

        # Проверка дневного убытка
        if abs(self.today_pnl) > self.initial_capital * 0.03:
            warnings.append(f"Large daily loss: {self.today_pnl:,.0f}")
            if abs(self.today_pnl) > self.initial_capital * 0.05:
                is_critical = True

        # Проверка количества сделок
        if self.metrics['total_trades'] == 0:
            warnings.append("No trades yet")
        elif self.today_trades == 0 and datetime.now().hour > 14:  # После 14:00
            warnings.append("No trades today")

        return {
            'status': 'critical' if is_critical else 'warning' if warnings else 'healthy',
            'warnings': warnings,
            'is_critical': is_critical,
            'metrics': self.metrics,
            'current_drawdown': self.current_drawdown,
            'daily_pnl': self.today_pnl,
            'daily_trades': self.today_trades
        }

    def plot_performance(self, save_path: Optional[str] = None):
        """
        Строит графики производительности

        Args:
            save_path: путь для сохранения
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Производительность {self.strategy_name}', fontsize=16)

        # 1. Кривая капитала
        ax1 = axes[0, 0]
        if self.equity_curve:
            times = [t for t, _ in self.equity_curve]
            equity = [v for _, v in self.equity_curve]
            ax1.plot(times, equity, color='blue', linewidth=1)
            ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
            ax1.set_title('Кривая капитала')
            ax1.set_xlabel('Дата')
            ax1.set_ylabel('Капитал')
            ax1.grid(True, alpha=0.3)

        # 2. Накопленная прибыль
        ax2 = axes[0, 1]
        if self.trades:
            cumulative_pnl = np.cumsum([t['pnl'] for t in self.trades])
            ax2.plot(range(len(cumulative_pnl)), cumulative_pnl, color='green')
            ax2.set_title('Накопленная прибыль')
            ax2.set_xlabel('Номер сделки')
            ax2.set_ylabel('Прибыль')
            ax2.grid(True, alpha=0.3)

        # 3. Винрейт по дням
        ax3 = axes[1, 0]
        daily = self.get_daily_stats(days=30)
        if not daily.empty:
            ax3.bar(daily.index, daily['win_rate'], color='purple', alpha=0.7)
            ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
            ax3.set_title('Винрейт по дням')
            ax3.set_xlabel('Дата')
            ax3.set_ylabel('Винрейт')
            ax3.grid(True, alpha=0.3)

        # 4. Распределение прибыли
        ax4 = axes[1, 1]
        if self.trades:
            pnls = [t['pnl_pct'] * 100 for t in self.trades]
            ax4.hist(pnls, bins=30, color='orange', alpha=0.7, edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--')
            ax4.set_title('Распределение прибыли по сделкам')
            ax4.set_xlabel('Прибыль %')
            ax4.set_ylabel('Частота')
            ax4.grid(True, alpha=0.3)

        # 5. Месячная доходность
        ax5 = axes[2, 0]
        monthly = self.get_monthly_stats()
        if not monthly.empty:
            monthly_returns = monthly['pnl'] / self.initial_capital * 100
            colors = ['green' if r > 0 else 'red' for r in monthly_returns]
            ax5.bar(range(len(monthly_returns)), monthly_returns.values, color=colors, alpha=0.7)
            ax5.set_title('Месячная доходность')
            ax5.set_xlabel('Месяц')
            ax5.set_ylabel('Доходность %')
            ax5.set_xticks(range(len(monthly_returns)))
            ax5.set_xticklabels([idx.strftime('%Y-%m') for idx in monthly.index], rotation=45)
            ax5.grid(True, alpha=0.3)

        # 6. Метрики
        ax6 = axes[2, 1]
        ax6.axis('off')
        metrics_text = f"""
        Текущие метрики:

        Доходность: {self.metrics['total_return']:.2%}
        Винрейт: {self.metrics['win_rate']:.1%}
        Profit Factor: {self.metrics['profit_factor']:.2f}
        Sharpe: {self.metrics['sharpe_ratio']:.2f}
        Max DD: {self.metrics['max_drawdown']:.1%}
        Текущая DD: {self.current_drawdown:.1%}

        Сделок всего: {self.metrics['total_trades']}
        Сделок сегодня: {self.today_trades}
        PnL сегодня: {self.today_pnl:+,.0f}

        Средний вин: {self.metrics['avg_win']:.2%}
        Средний лосс: {self.metrics['avg_loss']:.2%}
        """
        ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def save_history(self):
        """Сохраняет историю в файл"""
        history = {
            'strategy': self.strategy_name,
            'initial_capital': self.initial_capital,
            'last_update': datetime.now().isoformat(),
            'metrics': self.metrics,
            'equity_curve': [
                {'timestamp': t.isoformat(), 'equity': v}
                for t, v in self.equity_curve
            ],
            'trades': [
                {
                    'entry_time': t.get('entry_time').isoformat() if t.get('entry_time') else None,
                    'exit_time': t.get('exit_time').isoformat() if t.get('exit_time') else None,
                    'instrument': t.get('instrument', 'UNKNOWN'),
                    'direction': t.get('direction', 'UNKNOWN'),
                    'entry_price': t.get('entry_price', 0),
                    'exit_price': t.get('exit_price', 0),
                    'pnl': t.get('pnl', 0),
                    'pnl_pct': t.get('pnl_pct', 0),
                    'reason': t.get('reason', 'unknown'),
                    'holding_period': t.get('holding_period')
                }
                for t in self.trades
            ],
            'daily_stats': {
                str(k): v for k, v in self.daily_stats.items()
            }
        }

        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def load_history(self):
        """Загружает историю из файла"""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)

            # Загружаем метрики
            self.metrics.update(history.get('metrics', {}))

            # Загружаем кривую капитала
            for item in history.get('equity_curve', []):
                self.equity_curve.append((
                    datetime.fromisoformat(item['timestamp']),
                    item['equity']
                ))

            # Загружаем сделки
            for t in history.get('trades', []):
                normalized = self._normalize_trade(t)
                if normalized:
                    self.trades.append(normalized)

            # Загружаем дневную статистику
            for date_str, stats in history.get('daily_stats', {}).items():
                self.daily_stats[datetime.fromisoformat(date_str).date()] = stats

        except Exception as e:
            print(f"⚠️ Ошибка загрузки истории: {e}")

    def get_report(self) -> str:
        """
        Возвращает текстовый отчет о производительности

        Returns:
            Отформатированный отчет
        """
        report = []
        report.append(f"\n📊 Отчет о производительности - {self.strategy_name}")
        report.append("=" * 60)
        report.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("")

        # Основные метрики
        report.append("📈 Основные метрики:")
        report.append(f"  Начальный капитал: {self.initial_capital:,.0f}")
        report.append(f"  Текущий капитал: {self.current_capital:,.0f}")
        report.append(f"  Общая доходность: {self.metrics['total_return']:+.2%}")
        report.append(f"  Пик капитала: {self.peak_capital:,.0f}")
        report.append(f"  Текущая просадка: {self.current_drawdown:.2%}")
        report.append(f"  Макс. просадка: {self.metrics['max_drawdown']:.2%}")
        report.append("")

        # Статистика по сделкам
        report.append("📊 Статистика по сделкам:")
        report.append(f"  Всего сделок: {self.metrics['total_trades']}")
        report.append(f"  Винрейт: {self.metrics['win_rate']:.1%}")
        report.append(f"  Profit Factor: {self.metrics['profit_factor']:.2f}")
        report.append(f"  Коэф. Шарпа: {self.metrics['sharpe_ratio']:.2f}")
        report.append(f"  Средний вин: {self.metrics['avg_win']:.2%}")
        report.append(f"  Средний лосс: {self.metrics['avg_loss']:.2%}")
        report.append(f"  Среднее время удержания: {self.metrics['avg_holding_time']:.1f} ч")
        report.append("")

        # Дневная статистика
        report.append("📅 Сегодня:")
        report.append(f"  PnL: {self.today_pnl:+,.0f}")
        report.append(f"  Сделок: {self.today_trades}")

        # Оценка здоровья
        health = self.check_health()
        if health['status'] == 'healthy':
            report.append("\n🟢 Система работает нормально")
        elif health['status'] == 'warning':
            report.append("\n🟡 Есть предупреждения:")
            for w in health['warnings']:
                report.append(f"  • {w}")
        else:
            report.append("\n🔴 Критическое состояние:")
            for w in health['warnings']:
                report.append(f"  • {w}")

        return "\n".join(report)
