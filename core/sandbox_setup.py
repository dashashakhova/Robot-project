"""
Настройка и тестирование песочницы T-Invest API
"""
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from t_tech.invest import Client
from t_tech.invest import CandleInterval
from t_tech.invest import MoneyValue
from t_tech.invest import AccountType

# Определяем корневую директорию проекта
ROOT_DIR = Path(__file__).parent.parent
ENV_PATH = ROOT_DIR / '.env'

# Загружаем .env файл из корня проекта
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    print(f"✅ Загружен .env файл из {ENV_PATH}")
else:
    print(f"❌ Файл .env не найден по пути: {ENV_PATH}")
    print("Создайте файл .env в корне проекта с содержимым: INVEST_TOKEN=ваш_токен_песочницы")
    sys.exit(1)

TOKEN = os.getenv('INVEST_TOKEN')

if not TOKEN:
    print("❌ Токен не найден в файле .env")
    print("Убедитесь, что файл .env содержит строку: INVEST_TOKEN=ваш_токен_песочницы")
    sys.exit(1)


def get_sandbox_accounts():
    """Получает все счета в песочнице"""
    print("\n📊 Проверка счетов в песочнице...")

    try:
        with Client(TOKEN) as client:
            # Для песочницы используем sandbox-клиент
            sandbox_client = client.sandbox

            # Получаем все счета
            accounts_response = client.users.get_accounts()

            # Фильтруем только счета песочницы
            sandbox_accounts = [
                acc for acc in accounts_response.accounts
                if acc.type == AccountType.ACCOUNT_TYPE_SANDBOX
            ]

            if sandbox_accounts:
                print(f"  ✅ Найдено счетов в песочнице: {len(sandbox_accounts)}")
                for account in sandbox_accounts:
                    print(f"    • ID: {account.id}")
                    print(f"      Название: {account.name}")
                    print(f"      Статус: {account.status}")
                return sandbox_accounts
            else:
                print("  📭 Счетов в песочнице нет")
                return []

    except Exception as e:
        print(f"  ❌ Ошибка получения счетов: {e}")
        return []


def create_sandbox_account():
    """Создает новый счет в песочнице"""
    print("\n🆕 Создание нового счета в песочнице...")

    try:
        with Client(TOKEN) as client:
            # Уникальное имя для счета
            account_name = f"robot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Открываем счет в песочнице
            response = client.sandbox.open_sandbox_account(name=account_name)

            account_id = response.account_id
            print(f"  ✅ Счет создан успешно!")
            print(f"  🆔 ID счета: {account_id}")
            print(f"  📝 Название: {account_name}")

            return account_id

    except Exception as e:
        print(f"  ❌ Ошибка создания счета: {e}")
        return None


def fund_sandbox_account(account_id, amount=1000000):
    """Пополняет счет в песочнице тестовыми средствами"""
    print(f"\n💰 Пополнение счета {account_id[:8]}... на {amount} RUB")

    try:
        with Client(TOKEN) as client:
            # Пополняем счет
            client.sandbox.sandbox_pay_in(
                account_id=account_id,
                amount=MoneyValue(currency="rub", units=amount, nano=0)
            )
            print(f"  ✅ Счет пополнен на {amount} RUB")

            # Небольшая пауза для обновления баланса
            time.sleep(1)

            # Проверяем баланс
            portfolio = client.operations.get_portfolio(account_id=account_id)

            # Извлекаем баланс
            if portfolio.total_amount_currencies:
                balance = portfolio.total_amount_currencies.units + portfolio.total_amount_currencies.nano / 1e9
                currency = portfolio.total_amount_currencies.currency
                print(f"  💵 Текущий баланс: {balance:,.2f} {currency}")
            else:
                print(f"  ⚠️ Не удалось получить баланс")

    except Exception as e:
        print(f"  ❌ Ошибка пополнения счета: {e}")


def clear_sandbox_accounts():
    """Очищает все счета в песочнице"""
    print("\n🧹 Очистка счетов в песочнице...")

    accounts = get_sandbox_accounts()

    if not accounts:
        print("  📭 Нет счетов для очистки")
        return

    try:
        with Client(TOKEN) as client:
            for account in accounts:
                try:
                    client.sandbox.close_sandbox_account(account_id=account.id)
                    print(f"  ✅ Счет {account.id[:8]}... удален")
                    time.sleep(0.5)  # Небольшая пауза между удалениями
                except Exception as e:
                    print(f"  ❌ Ошибка удаления счета {account.id[:8]}...: {e}")
    except Exception as e:
        print(f"  ❌ Ошибка при очистке: {e}")


def test_market_data(account_id):
    """Тестирует получение рыночных данных"""
    print("\n📈 Тестирование получения рыночных данных...")

    # Тестовые инструменты
    test_instruments = [
        {"name": "Сбербанк", "figi": "BBG004730N88"},
        {"name": "ВТБ", "figi": "BBG004730ZJ9"},
        {"name": "Газпром", "figi": "BBG004730RP0"},
    ]

    try:
        with Client(TOKEN) as client:
            for instrument in test_instruments:
                print(f"\n  {instrument['name']} ({instrument['figi']}):")

                # Период для запроса (последний час)
                now = datetime.utcnow()
                from_time = now - timedelta(hours=1)

                # Получаем свечи
                candles = client.get_all_candles(
                    figi=instrument['figi'],
                    from_=from_time,
                    to=now,
                    interval=CandleInterval.CANDLE_INTERVAL_1_MIN
                )

                # Считаем количество полученных свечей
                candle_count = 0
                last_candle = None

                for candle in candles:
                    candle_count += 1
                    last_candle = candle

                if candle_count > 0:
                    print(f"    ✅ Получено свечей за последний час: {candle_count}")

                    if last_candle:
                        # Извлекаем цену закрытия
                        close_price = last_candle.close.units + last_candle.close.nano / 1e9
                        print(f"    💰 Последняя цена закрытия: {close_price:.2f} RUB")
                        print(f"    📊 Объем последней свечи: {last_candle.volume}")
                else:
                    print(f"    ⚠️ Нет данных за последний час")

                # Небольшая пауза между запросами
                time.sleep(0.5)

    except Exception as e:
        print(f"  ❌ Ошибка получения данных: {e}")


def save_account_id(account_id):
    """Сохраняет ID счета в файл"""
    account_file = Path('.sandbox_account')
    with open(account_file, 'w') as f:
        f.write(account_id)
    print(f"  💾 ID счета сохранен в {account_file}")


def load_account_id():
    """Загружает ID счета из файла"""
    account_file = Path('.sandbox_account')
    if account_file.exists():
        with open(account_file, 'r') as f:
            return f.read().strip()
    return None


def main():
    """Главная функция"""
    print("=" * 60)
    print("🔧 НАСТРОЙКА ПЕСОЧНИЦЫ T-INVEST API")
    print("=" * 60)

    print(f"\n🔑 Токен: {TOKEN[:15]}...")

    # Проверяем существующие счета
    accounts = get_sandbox_accounts()

    # Определяем, какой счет использовать
    account_id = None

    if accounts:
        print(f"\n💡 Найдено {len(accounts)} счетов в песочнице")

        # Проверяем, есть ли сохраненный ID
        saved_id = load_account_id()

        if saved_id and any(acc.id == saved_id for acc in accounts):
            print(f"  ✅ Используем сохраненный счет: {saved_id[:8]}...")
            account_id = saved_id
        else:
            print("\nВыберите действие:")
            print("1. Использовать существующий счет")
            print("2. Создать новый счет (старые будут удалены)")
            print("3. Очистить все счета и выйти")

            choice = input("Ваш выбор (1/2/3): ").strip()

            if choice == '1':
                # Показываем список счетов
                print("\nДоступные счета:")
                for i, acc in enumerate(accounts, 1):
                    print(f"  {i}. {acc.id} ({acc.name})")

                try:
                    idx = int(input("Выберите номер счета: ")) - 1
                    if 0 <= idx < len(accounts):
                        account_id = accounts[idx].id
                        print(f"  ✅ Выбран счет: {account_id[:8]}...")
                    else:
                        print("  ❌ Неверный номер")
                        return
                except ValueError:
                    print("  ❌ Неверный ввод")
                    return

            elif choice == '2':
                # Очищаем старые счета и создаем новый
                clear_sandbox_accounts()
                account_id = create_sandbox_account()

            elif choice == '3':
                clear_sandbox_accounts()
                print("\n✅ Очистка завершена")
                return
            else:
                print("❌ Неверный выбор")
                return
    else:
        # Счетов нет, создаем новый
        print("\n💡 Счетов в песочнице не найдено")
        account_id = create_sandbox_account()

    if account_id:
        # Пополняем счет
        fund_sandbox_account(account_id)

        # Сохраняем ID счета
        save_account_id(account_id)

        # Тестируем получение данных
        test_market_data(account_id)

        print("\n" + "=" * 60)
        print("✅ ПЕСОЧНИЦА УСПЕШНО НАСТРОЕНА!")
        print("=" * 60)
        print(f"\n🏦 ID счета: {account_id}")
        print("💵 Баланс: 1,000,000 RUB")
        print("\nТеперь вы можете запускать торгового робота в режиме песочницы:")
        print("  python main.py --instrument SBER --mode sandbox")
    else:
        print("\n❌ Не удалось настроить песочницу")


if __name__ == "__main__":
    main()