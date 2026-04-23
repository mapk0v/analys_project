"""
ЗАДАНИЕ 1: Мониторинг численности населения и демографических показателей
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import load_data, classify_city, calculate_cagr, get_region_from_city
from config import DEMOGRAPHIC_RATES
import logging

logger = logging.getLogger(__name__)


class PopulationMonitor:
    """Мониторинг населения по муниципальным образованиям"""

    def __init__(self):
        self.historical, self.forecast = load_data()

    def get_municipality_population(self, year=None):
        """Получение численности населения по городам"""
        if year is None:
            year = self.historical['year'].max()

        data = self.historical[self.historical['year'] == year].copy()
        data = data.sort_values('population', ascending=False)

        # Список крупных городов (исключаем поселки)
        major_cities = [
            'Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург', 'Казань',
            'Нижний Новгород', 'Красноярск', 'Челябинск', 'Самара', 'Уфа',
            'Ростов-на-Дону', 'Краснодар', 'Омск', 'Воронеж', 'Пермь',
            'Волгоград', 'Саратов', 'Тюмень', 'Тольятти', 'Ижевск',
            'Барнаул', 'Ульяновск', 'Иркутск', 'Хабаровск', 'Ярославль',
            'Владивосток', 'Махачкала', 'Томск', 'Оренбург', 'Кемерово',
            'Новокузнецк', 'Рязань'
        ]

        result = []
        for _, row in data.iterrows():
            if row['city'] in major_cities or (row['population'] >= 500000 and 'пгт' not in row['city']):
                result.append({
                    'name': row['city'],
                    'population': int(row['population']),
                    'type': classify_city(row['population']),
                    'region': get_region_from_city(row['city'])
                })

        return result

    def get_all_regions(self):
        """Получение всех регионов"""
        last_year = self.historical['year'].max()
        data = self.historical[self.historical['year'] == last_year].copy()
        data['region'] = data['city'].apply(get_region_from_city)

        regional_stats = data.groupby('region').agg({
            'population': 'sum'
        }).reset_index()
        regional_stats = regional_stats.sort_values('population', ascending=False)

        return regional_stats.to_dict('records')

    def get_population_dynamics(self, city_name):
        """Динамика изменения населения для конкретного города"""
        city_data = self.historical[
            self.historical['city'] == city_name
        ].sort_values('year')

        if city_data.empty or len(city_data) < 2:
            return None

        populations = city_data['population'].values
        years = city_data['year'].values

        start_pop = populations[0]
        end_pop = populations[-1]
        years_span = len(populations) - 1

        absolute_change = end_pop - start_pop
        relative_change = (absolute_change / start_pop) * 100 if start_pop > 0 else 0
        cagr = calculate_cagr(start_pop, end_pop, years_span) * 100

        return {
            'city': city_name,
            'start_year': years[0],
            'end_year': years[-1],
            'start_population': int(start_pop),
            'end_population': int(end_pop),
            'absolute_change': int(absolute_change),
            'relative_change': round(relative_change, 2),
            'cagr': round(cagr, 2)
        }

    def get_all_population_dynamics(self):
        """Динамика для всех крупных городов"""
        major_cities = [c['name'] for c in self.get_municipality_population()]
        results = []

        for city in major_cities:
            dynamics = self.get_population_dynamics(city)
            if dynamics:
                results.append(dynamics)

        return sorted(results, key=lambda x: x['relative_change'], reverse=True)

    def get_demographic_indicators(self, city_name):
        """Демографические показатели для города"""
        dynamics = self.get_population_dynamics(city_name)
        if not dynamics:
            return None

        population = dynamics['end_population']

        if dynamics['cagr'] > 0.5:
            birth_rate = 11.5
            death_rate = 10.5
        elif dynamics['cagr'] < -0.5:
            birth_rate = 8.5
            death_rate = 13.5
        else:
            birth_rate = 10.0
            death_rate = 12.0

        natural_increase = birth_rate - death_rate
        births = int(population * birth_rate / 1000)
        deaths = int(population * death_rate / 1000)

        return {
            'city': city_name,
            'population': population,
            'birth_rate': round(birth_rate, 1),
            'death_rate': round(death_rate, 1),
            'natural_increase': round(natural_increase, 1),
            'births': births,
            'deaths': deaths,
            'natural_change': births - deaths
        }

    def get_all_demographic_indicators(self):
        """Демографические показатели для всех крупных городов"""
        major_cities = [c['name'] for c in self.get_municipality_population()]
        results = []

        for city in major_cities:
            demo = self.get_demographic_indicators(city)
            if demo:
                results.append(demo)

        return results

    def get_top_growing_declining(self, n=10, min_population=500000):
        """Топ растущих и убывающих городов"""
        dynamics = self.get_all_population_dynamics()

        growing = [d for d in dynamics if d['relative_change'] > 0 and d['start_population'] >= min_population]
        declining = [d for d in dynamics if d['relative_change'] < 0 and d['start_population'] >= min_population]

        return {
            'growing': sorted(growing, key=lambda x: x['relative_change'], reverse=True)[:n],
            'declining': sorted(declining, key=lambda x: x['relative_change'])[:n]
        }

    def create_regional_heatmap(self):
        """Создание тепловой карты"""
        last_year = self.historical['year'].max()
        data = self.historical[self.historical['year'] == last_year].copy()
        data['region'] = data['city'].apply(get_region_from_city)

        regional_data = data.groupby('region').agg({
            'population': 'sum'
        }).reset_index()
        regional_data = regional_data.sort_values('population', ascending=False)

        fig = go.Figure(data=go.Bar(
            x=regional_data['region'][:15],
            y=regional_data['population'][:15],
            marker_color='crimson',
            text=regional_data['population'][:15].apply(lambda x: f'{x/1e6:.1f}M'),
            textposition='auto'
        ))

        fig.update_layout(
            title='Топ-15 регионов по численности населения',
            xaxis_title='Регион',
            yaxis_title='Население (чел.)',
            height=500,
            xaxis_tickangle=-45
        )

        return fig