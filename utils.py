"""
Вспомогательные функции
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import logging
from pathlib import Path
from config import HISTORICAL_FILE, FORECAST_FILE, REPORTS_DIR, POPULATION_THRESHOLDS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data():
    """Загрузка исторических и прогнозных данных"""
    try:
        if not HISTORICAL_FILE.exists():
            logger.error(f"Файл не найден: {HISTORICAL_FILE}")
            return pd.DataFrame(), pd.DataFrame()

        historical = pd.read_csv(HISTORICAL_FILE)
        forecast = pd.read_csv(FORECAST_FILE) if FORECAST_FILE.exists() else pd.DataFrame()

        historical['year'] = historical['year'].astype(int)
        historical['population'] = historical['population'].astype(float)

        # Фильтруем только крупные города (исключаем пгт, поселки)
        historical = filter_major_cities(historical)

        logger.info(f"Загружено {len(historical)} исторических записей, {len(forecast)} прогнозов")
        return historical, forecast
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame(), pd.DataFrame()


def filter_major_cities(df):
    """
    Фильтрация: оставляем только крупные города (население > 100k)
    """
    # Оставляем только города с населением > 100000
    df = df[df['population'] >= 50000].copy()

    # Исключаем явно поселки
    exclude_patterns = ['пгт', 'рп', 'поселок', 'село', 'деревня', 'п ', 'д ']
    for pattern in exclude_patterns:
        df = df[~df['city'].str.contains(pattern, na=False, case=False)]

    return df


def classify_city(population):
    """Классификация города по численности"""
    T = POPULATION_THRESHOLDS

    if population >= T['million']:
        return 'город-миллионник'
    elif population >= T['large']:
        return 'крупный город'
    elif population >= T['medium']:
        return 'средний город'
    elif population >= T['small']:
        return 'большой город'
    elif population >= T['town']:
        return 'малый город'
    else:
        return 'малый город'


def calculate_cagr(start_pop, end_pop, years):
    """Расчет среднегодового темпа роста (CAGR)"""
    if start_pop <= 0 or years <= 0:
        return 0
    return (end_pop / start_pop) ** (1/years) - 1


def save_report(data, filename, format='json'):
    """Сохранение отчета"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = REPORTS_DIR / f"{filename}_{timestamp}.{format}"

    if format == 'json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    elif format == 'csv' and isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False, encoding='utf-8-sig')
    elif format == 'html' and hasattr(data, 'write_html'):
        data.write_html(filepath)

    logger.info(f"Отчет сохранен: {filepath}")
    return filepath


def get_region_from_city(city_name):
    """Определение региона по названию города"""
    regions = {
        'Москва': 'Москва',
        'Санкт-Петербург': 'Санкт-Петербург',
        'Новосибирск': 'Новосибирская область',
        'Екатеринбург': 'Свердловская область',
        'Казань': 'Республика Татарстан',
        'Нижний Новгород': 'Нижегородская область',
        'Красноярск': 'Красноярский край',
        'Челябинск': 'Челябинская область',
        'Самара': 'Самарская область',
        'Уфа': 'Республика Башкортостан',
        'Ростов-на-Дону': 'Ростовская область',
        'Краснодар': 'Краснодарский край',
        'Омск': 'Омская область',
        'Воронеж': 'Воронежская область',
        'Пермь': 'Пермский край',
        'Волгоград': 'Волгоградская область',
        'Саратов': 'Саратовская область',
        'Тюмень': 'Тюменская область',
        'Тольятти': 'Самарская область',
        'Ижевск': 'Удмуртская Республика',
        'Барнаул': 'Алтайский край',
        'Ульяновск': 'Ульяновская область',
        'Иркутск': 'Иркутская область',
        'Хабаровск': 'Хабаровский край',
        'Ярославль': 'Ярославская область',
        'Владивосток': 'Приморский край',
        'Махачкала': 'Республика Дагестан',
        'Томск': 'Томская область',
        'Оренбург': 'Оренбургская область',
        'Кемерово': 'Кемеровская область',
        'Новокузнецк': 'Кемеровская область',
        'Рязань': 'Рязанская область',
        'Астрахань': 'Астраханская область',
        'Набережные Челны': 'Республика Татарстан',
        'Пенза': 'Пензенская область',
        'Липецк': 'Липецкая область',
        'Киров': 'Кировская область',
        'Тула': 'Тульская область',
        'Чебоксары': 'Чувашская Республика',
        'Калининград': 'Калининградская область',
        'Курск': 'Курская область',
        'Ставрополь': 'Ставропольский край',
        'Сочи': 'Краснодарский край',
        'Белгород': 'Белгородская область',
        'Нижний Тагил': 'Свердловская область',
        'Владимир': 'Владимирская область',
        'Архангельск': 'Архангельская область',
        'Сургут': 'Ханты-Мансийский АО',
        'Чита': 'Забайкальский край',
        'Калуга': 'Калужская область',
        'Смоленск': 'Смоленская область',
        'Волжский': 'Волгоградская область',
        'Курган': 'Курганская область',
        'Череповец': 'Вологодская область',
        'Орёл': 'Орловская область',
        'Вологда': 'Вологодская область',
        'Владикавказ': 'Северная Осетия-Алания',
        'Мурманск': 'Мурманская область',
        'Саранск': 'Республика Мордовия',
        'Якутск': 'Республика Саха (Якутия)',
        'Тамбов': 'Тамбовская область',
        'Грозный': 'Чеченская Республика',
        'Стерлитамак': 'Республика Башкортостан',
        'Кострома': 'Костромская область',
        'Петрозаводск': 'Республика Карелия',
        'Нижневартовск': 'Ханты-Мансийский АО',
        'Новороссийск': 'Краснодарский край',
        'Йошкар-Ола': 'Республика Марий Эл',
        'Таганрог': 'Ростовская область',
        'Комсомольск-на-Амуре': 'Хабаровский край',
    }
    return regions.get(city_name, city_name[:20])