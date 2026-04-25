"""
API для аналитической системы демографии РФ
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# Импортируем существующие классы аналитики
from task1_monitoring import PopulationMonitor
from task2_forecasting import PopulationForecaster
from task3_ai_analytics import AIAnalytics

from typing import List, Optional
from pydantic import BaseModel

import json
import numpy as np


class AIReportResponse(BaseModel):
    city: str
    region: str
    generated_at: str
    section_31_summary: str
    section_32_trends_and_factors: dict
    section_33_forecast: dict
    section_34_recommendations: List[dict]
    section_35_conclusion: str

class ForecastResponse(BaseModel):
    city: str
    last_year: int
    last_population: int
    horizon: int
    future_years: List[int]
    predictions: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    metrics: Optional[dict] = None

class MultiForecastRequest(BaseModel):
    cities: List[str]
    horizon: int = 10

# Создаём экземпляры классов (они загрузят данные при старте)
monitor = PopulationMonitor()
forecaster = PopulationForecaster()
ai_analytics = AIAnalytics()

class CityInfo(BaseModel):
    name: str
    population: int
    type: str
    region: str

class DynamicsResponse(BaseModel):
    city: str
    start_year: int
    end_year: int
    start_population: int
    end_population: int
    absolute_change: int
    relative_change: float
    cagr: float

class TopCitiesResponse(BaseModel):
    growing: List[DynamicsResponse]
    declining: List[DynamicsResponse]

# Создаём приложение FastAPI
app = FastAPI(
    title="Демографическая аналитика РФ",
    description="API для мониторинга, прогнозирования и AI-аналитики населения",
    version="1.0.0"
)

# 2.2. Добавляем CORS (пока разрешим всё для тестирования)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # при разработке можно звёздочку, потом ограничим
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2.3. Тестовый эндпоинт
@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "API работает"}

# 2.4. Корневой эндпоинт
@app.get("/")
async def root():
    return {
        "service": "Demographic Analytics API",
        "endpoints": "/ping, /docs, /redoc",
        "status": "running"
    }


# ==================== Эндпоинты для мониторинга ====================

@app.get("/api/regions", response_model=List[str])
async def get_regions():
    """Получить список всех регионов"""
    all_cities = monitor.get_municipality_population()
    regions = sorted(set(city['region'] for city in all_cities))
    return regions


@app.get("/api/cities", response_model=List[CityInfo])
async def get_cities(
        region: Optional[str] = None,
        city_type: Optional[str] = None,
        min_population: Optional[int] = None
):
    """
    Получить список городов с возможностью фильтрации.
    - region: название региона (например, "Москва", "Московская область")
    - city_type: тип города ("город-миллионник", "крупный город", ...)
    - min_population: минимальная численность населения
    """
    cities = monitor.get_municipality_population()

    if region:
        cities = [c for c in cities if c['region'] == region]
    if city_type:
        cities = [c for c in cities if c['type'] == city_type]
    if min_population:
        cities = [c for c in cities if c['population'] >= min_population]

    return cities


@app.get("/api/cities/search")
async def search_cities(q: str, limit: int = 10):
    """Поиск городов по названию (по подстроке)"""
    cities = monitor.get_municipality_population()
    q_lower = q.lower()
    matches = [c for c in cities if q_lower in c['name'].lower()]
    return matches[:limit]


@app.get("/api/dynamics/{city_name}", response_model=DynamicsResponse)
async def get_city_dynamics(city_name: str):
    """Получить динамику населения города за весь доступный период"""
    dynamics = monitor.get_population_dynamics(city_name)
    if dynamics is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Город '{city_name}' не найден")
    return dynamics


@app.get("/api/top_changes", response_model=TopCitiesResponse)
async def get_top_changes(
        n: int = 10,
        min_population: int = 500000
):
    """
    Получить топ городов по росту и снижению населения.
    - n: количество городов в каждом списке
    - min_population: минимальная численность населения для включения
    """
    result = monitor.get_top_growing_declining(n=n, min_population=min_population)
    return result


@app.get("/api/regional_population")
async def get_regional_population():
    """
    Получить численность населения по регионам (для тепловой карты).
    Возвращает список объектов: {region: str, population: int}
    """
    # Используем существующий метод get_all_regions из monitor
    regions = monitor.get_all_regions()
    # Метод get_all_regions возвращает список словарей вида {'region': ..., 'population': ...}
    # Если такого метода нет в классе PopulationMonitor, реализуем вручную.
    return regions


# ==================== Эндпоинты для прогнозирования ====================

@app.get("/api/forecast/{city_name}", response_model=ForecastResponse)
async def get_forecast(
        city_name: str,
        horizon: int = 10,
        include_metrics: bool = True
):
    """
    Получить прогноз численности населения для города.
    - horizon: горизонт прогноза (от 1 до 15 лет, по умолчанию 10)
    - include_metrics: включить метрики качества прогноза
    """
    if horizon < 1 or horizon > 15:
        raise HTTPException(status_code=400, detail="Горизонт прогноза должен быть от 1 до 15 лет")

    forecast = forecaster.forecast_city(city_name, horizon=horizon)
    if forecast is None:
        raise HTTPException(status_code=404,
                            detail=f"Город '{city_name}' не найден или недостаточно данных для прогноза")

    # Преобразуем numpy типы в стандартные Python
    response = {
        "city": forecast['city'],
        "last_year": forecast['last_year'],
        "last_population": forecast['last_population'],
        "horizon": horizon,
        "future_years": [int(y) for y in forecast['future_years']],
        "predictions": [float(p) for p in forecast['predictions']],
        "lower_bound": [float(lb) for lb in forecast['lower_bound']],
        "upper_bound": [float(ub) for ub in forecast['upper_bound']],
    }

    if include_metrics:
        metrics = forecaster.calculate_metrics(forecast)
        if metrics:
            response["metrics"] = metrics

    return response


@app.post("/api/forecast/compare")
async def compare_forecasts(request: MultiForecastRequest):
    """
    Сравнить прогнозы для нескольких городов.
    Принимает JSON: {"cities": ["Москва", "СПб"], "horizon": 10}
    """
    results = {}
    for city in request.cities:
        forecast = forecaster.forecast_city(city, horizon=request.horizon)
        if forecast:
            results[city] = {
                "future_years": [int(y) for y in forecast['future_years']],
                "predictions": [float(p) for p in forecast['predictions']],
                "last_population": forecast['last_population']
            }
        else:
            results[city] = {"error": "Недостаточно данных"}
    return {"horizon": request.horizon, "results": results}


@app.get("/api/forecast/scenarios/{city_name}")
async def get_forecast_scenarios(city_name: str, horizon: int = 15):
    """
    Получить прогноз с тремя сценариями (оптимистичный, базовый, пессимистичный).
    """
    scenarios = forecaster.forecast_with_scenarios(city_name, horizon=horizon)
    if scenarios is None:
        raise HTTPException(status_code=404, detail=f"Город '{city_name}' не найден")

    # Преобразуем данные для удобства фронтенда
    return {
        "city": scenarios['city'],
        "last_population": scenarios['last_population'],
        "last_year": scenarios['last_year'],
        "scenarios": {
            name: {
                "rate": sc['rate'],
                "future_years": [int(y) for y in sc['future_years']],
                "predictions": sc['predictions']
            }
            for name, sc in scenarios['scenarios'].items()
        }
    }


# ==================== Эндпоинты для AI-аналитики ====================

@app.get("/api/ai/report/{city_name}")
async def get_ai_report(city_name: str):
    """
    Получить полный аналитический отчёт по городу в формате JSON.
    Если город не найден, возвращает 404.
    """
    report = ai_analytics.generate_full_report(city_name, forecast_horizon=10)

    # Проверяем, есть ли данные по городу (если резюме содержит фразу "не найдены")
    if "не найдены" in report.get('section_31_summary', ''):
        raise HTTPException(status_code=404, detail=f"Город '{city_name}' не найден в данных")

    # Преобразуем numpy типы в стандартные Python
    from numpy import integer, floating, ndarray
    import json

    def convert_to_serializable(obj):
        """Рекурсивно преобразует numpy типы в стандартные Python"""
        if isinstance(obj, (integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool)):
            # Для объектов, не являющихся примитивами
            return convert_to_serializable(obj.__dict__)
        return obj

    # Применяем конвертацию
    report = convert_to_serializable(report)

    return report


@app.get("/api/ai/report/{city_name}/markdown")
async def get_ai_report_markdown(city_name: str):
    """
    Скачать аналитический отчёт в формате Markdown.
    """
    md_content = ai_analytics.generate_markdown_report(city_name, forecast_horizon=10)

    if md_content is None or "не найдены" in md_content:
        raise HTTPException(status_code=404, detail=f"Город '{city_name}' не найден в данных")

    # Используем только латиницу в имени файла, чтобы избежать проблем с кодировкой
    # Транслитерация простейшая (можно заменить на любую)
    translit_map = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
        'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'
    }

    safe_name = ''.join(translit_map.get(c.lower(), c.lower()) for c in city_name)
    safe_name = safe_name.replace(' ', '_')

    return PlainTextResponse(
        content=md_content,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename={safe_name}_analytics_report.md"}
    )


@app.get("/api/ai/summary/{city_name}")
async def get_ai_summary(city_name: str):
    """Получить только краткое резюме (пункт 3.1)"""
    summary = ai_analytics.generate_summary(city_name)
    if summary is None or "не найдены" in summary:
        raise HTTPException(status_code=404, detail=f"Город '{city_name}' не найден")
    return {"city": city_name, "summary": summary}


@app.get("/api/ai/recommendations/{city_name}")
async def get_ai_recommendations(city_name: str):
    """Получить только рекомендации (пункт 3.4)"""
    recommendations = ai_analytics.generate_recommendations(city_name)
    if not recommendations:
        raise HTTPException(status_code=404, detail=f"Город '{city_name}' не найден или нет рекомендаций")
    return {"city": city_name, "recommendations": recommendations}

# ==================== Дополнительные эндпоинты ====================

@app.get("/api/available_years")
async def get_available_years():
    """Получить диапазон лет, за которые есть исторические данные"""
    years = sorted(monitor.historical['year'].unique())
    return {"min_year": int(min(years)), "max_year": int(max(years)), "all_years": [int(y) for y in years]}

