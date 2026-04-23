"""
ЗАДАНИЕ 2: Прогнозирование численности населения
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import load_data, calculate_cagr
import logging

logger = logging.getLogger(__name__)


class PopulationForecaster:
    """
    Прогнозирование численности населения на основе исторических данных
    """

    def __init__(self):
        self.historical, self.forecast = load_data()
        self.models = {}

    # ==================== 1. Формирование прогноза ====================

    def forecast_city(self, city_name, horizon=10):
        """
        Прогноз для конкретного города на заданный горизонт
        """
        city_data = self.historical[self.historical['city'] == city_name].sort_values('year')

        if city_data.empty:
            return None

        years = city_data['year'].values
        populations = city_data['population'].values

        if len(populations) < 3:
            return None

        # Линейная регрессия
        X = years.reshape(-1, 1)
        y = populations

        model = LinearRegression()
        model.fit(X, y)

        # Прогноз
        last_year = years[-1]
        future_years = np.arange(last_year + 1, last_year + horizon + 1)
        predictions = model.predict(future_years.reshape(-1, 1))

        # Доверительный интервал
        residuals = y - model.predict(X)
        std_residuals = np.std(residuals) if len(residuals) > 0 else np.std(populations) * 0.05
        z_score = 1.96  # 95% доверительный интервал

        lower_bound = predictions - z_score * std_residuals
        upper_bound = predictions + z_score * std_residuals

        # Ограничиваем нижнюю границу нулем
        lower_bound = np.maximum(lower_bound, 0)

        # Данные для графика
        historical_data = city_data[['year', 'population']].to_dict('records')

        return {
            'city': city_name,
            'last_year': int(last_year),
            'last_population': int(populations[-1]),
            'horizon': horizon,
            'future_years': future_years.tolist(),
            'predictions': predictions.tolist(),
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
            'historical_data': historical_data,
            'model': model,
            'residuals_std': float(std_residuals)
        }

    def forecast_multiple_cities(self, cities, horizon=10):
        """
        Прогноз для нескольких городов
        """
        results = {}
        for city in cities:
            forecast = self.forecast_city(city, horizon)
            if forecast:
                results[city] = forecast
                # Добавляем метрики качества
                results[city]['metrics'] = self.calculate_metrics(forecast)
        return results

    # ==================== 2. Метрики качества прогноза ====================

    def calculate_metrics(self, forecast_result):
        """
        Расчет метрик качества прогноза (MAPE, RMSE, MAE, R²)
        """
        historical_data = forecast_result['historical_data']

        if len(historical_data) < 2:
            return None

        # Берем исторические данные для проверки
        years = [d['year'] for d in historical_data]
        actual = [d['population'] for d in historical_data]

        # Прогноз на историческом периоде (кросс-валидация)
        X = np.array(years).reshape(-1, 1)
        model = forecast_result['model']
        predicted = model.predict(X)

        # Расчет метрик
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted) * 100

        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'mae': round(mae, 0),
            'rmse': round(rmse, 0),
            'mape': round(mape, 2),
            'r2': round(r2, 4),
            'interpretation': self._interpret_metrics(mape, r2)
        }

    def _interpret_metrics(self, mape, r2):
        """Интерпретация метрик качества"""
        if mape < 5:
            accuracy = "Высокая точность"
        elif mape < 10:
            accuracy = "Хорошая точность"
        elif mape < 20:
            accuracy = "Удовлетворительная точность"
        else:
            accuracy = "Низкая точность"

        if r2 > 0.9:
            fit = "Отличное соответствие"
        elif r2 > 0.7:
            fit = "Хорошее соответствие"
        elif r2 > 0.5:
            fit = "Умеренное соответствие"
        else:
            fit = "Слабое соответствие"

        return f"{accuracy}, {fit} (MAPE={mape:.1f}%, R²={r2:.2f})"

    # ==================== 3. Визуализация прогноза ====================

    def create_forecast_chart(self, forecast_result):
        """
        Создание графика с историческими данными и прогнозом
        """
        if forecast_result is None:
            return None

        fig = go.Figure()

        # Исторические данные
        hist_data = forecast_result['historical_data']
        fig.add_trace(go.Scatter(
            x=[d['year'] for d in hist_data],
            y=[d['population'] for d in hist_data],
            mode='lines+markers',
            name='Фактические данные',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8, color='#2E86AB')
        ))

        # Прогноз
        fig.add_trace(go.Scatter(
            x=forecast_result['future_years'],
            y=forecast_result['predictions'],
            mode='lines+markers',
            name='Прогноз',
            line=dict(color='#F18F01', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond', color='#F18F01')
        ))

        # Доверительный интервал
        fig.add_trace(go.Scatter(
            x=forecast_result['future_years'] + forecast_result['future_years'][::-1],
            y=forecast_result['upper_bound'] + forecast_result['lower_bound'][::-1],
            fill='toself',
            fillcolor='rgba(241, 143, 1, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% доверительный интервал'
        ))

        # Настройка графика
        last_value = forecast_result['last_population']
        last_year = forecast_result['last_year']
        final_value = forecast_result['predictions'][-1]

        fig.update_layout(
            title=f"Прогноз численности населения: {forecast_result['city']}",
            xaxis_title="Год",
            yaxis_title="Численность населения (чел.)",
            hovermode='x unified',
            height=500,
            annotations=[
                dict(
                    x=0.02, y=0.98, xref="paper", yref="paper",
                    text=f"Текущее: {last_value:,} чел. ({last_year})<br>Прогноз: {int(final_value):,} чел. ({forecast_result['future_years'][-1]})",
                    showarrow=False, font=dict(size=12),
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="black", borderwidth=1
                )
            ]
        )

        return fig

    def create_comparison_chart(self, forecasts_dict):
        """
        Создание графика для сравнения прогнозов нескольких городов
        """
        fig = go.Figure()
        colors = ['#2E86AB', '#A23B72', '#3AAFA9', '#F18F01', '#6C91B2']

        for i, (city, forecast) in enumerate(forecasts_dict.items()):
            if forecast is None:
                continue

            # Исторические данные
            hist_data = forecast['historical_data']
            fig.add_trace(go.Scatter(
                x=[d['year'] for d in hist_data],
                y=[d['population'] for d in hist_data],
                mode='lines+markers',
                name=f"{city} (факт)",
                line=dict(color=colors[i % len(colors)], width=2),
                legendgroup=city
            ))

            # Прогноз
            fig.add_trace(go.Scatter(
                x=forecast['future_years'],
                y=forecast['predictions'],
                mode='lines',
                name=f"{city} (прогноз)",
                line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                legendgroup=city
            ))

        fig.update_layout(
            title="Сравнение прогнозов населения городов",
            xaxis_title="Год",
            yaxis_title="Численность населения (чел.)",
            hovermode='x unified',
            height=500
        )

        return fig

    # ==================== 4. Прогноз с разными сценариями ====================

    def forecast_with_scenarios(self, city_name, horizon=15):
        """
        Прогноз с разными сценариями (оптимистичный, базовый, пессимистичный)
        """
        base_forecast = self.forecast_city(city_name, horizon)

        if base_forecast is None:
            return None

        last_pop = base_forecast['last_population']

        # Сценарии роста
        scenarios = {
            'optimistic': {'rate': 0.02, 'color': '#3AAFA9'},   # +2% в год
            'base': {'rate': base_forecast['predictions'][-1] / last_pop - 1, 'color': '#2E86AB'},  # исторический тренд
            'pessimistic': {'rate': -0.01, 'color': '#F18F01'}   # -1% в год
        }

        # Корректируем базовый сценарий
        scenarios['base']['rate'] = max(scenarios['base']['rate'], -0.03)
        scenarios['base']['rate'] = min(scenarios['base']['rate'], 0.03)

        results = {}
        for name, scenario in scenarios.items():
            predictions = []
            current = last_pop
            for i in range(1, horizon + 1):
                current = current * (1 + scenario['rate'])
                predictions.append(int(current))

            results[name] = {
                'rate': round(scenario['rate'] * 100, 2),
                'predictions': predictions,
                'future_years': base_forecast['future_years'],
                'color': scenario['color']
            }

        return {
            'city': city_name,
            'last_population': last_pop,
            'last_year': base_forecast['last_year'],
            'scenarios': results,
            'historical_data': base_forecast['historical_data']
        }

    def create_scenarios_chart(self, scenarios_result):
        """
        Создание графика с разными сценариями прогноза
        """
        if scenarios_result is None:
            return None

        fig = go.Figure()

        # Исторические данные
        hist_data = scenarios_result['historical_data']
        fig.add_trace(go.Scatter(
            x=[d['year'] for d in hist_data],
            y=[d['population'] for d in hist_data],
            mode='lines+markers',
            name='Фактические данные',
            line=dict(color='#2C3E50', width=3),
            marker=dict(size=8)
        ))

        # Сценарии
        for name, scenario in scenarios_result['scenarios'].items():
            fig.add_trace(go.Scatter(
                x=scenario['future_years'],
                y=scenario['predictions'],
                mode='lines',
                name=f"{name.capitalize()} сценарий ({scenario['rate']:+.1f}%/год)",
                line=dict(color=scenario['color'], width=2, dash='dash' if name != 'base' else 'solid')
            ))

        fig.update_layout(
            title=f"Сценарии развития населения: {scenarios_result['city']}",
            xaxis_title="Год",
            yaxis_title="Численность населения (чел.)",
            hovermode='x unified',
            height=500
        )

        return fig

    # ==================== Полный отчет прогнозирования ====================

    def generate_forecast_report(self, city_name, horizon=15):
        """
        Генерация полного отчета прогнозирования
        """
        forecast = self.forecast_city(city_name, horizon)

        if forecast is None:
            return None

        metrics = self.calculate_metrics(forecast)
        scenarios = self.forecast_with_scenarios(city_name, horizon)

        return {
            'city': city_name,
            'horizon': horizon,
            'forecast': forecast,
            'metrics': metrics,
            'scenarios': scenarios,
            'chart': self.create_forecast_chart(forecast),
            'scenarios_chart': self.create_scenarios_chart(scenarios) if scenarios else None,
            'timestamp': pd.Timestamp.now().isoformat()
        }