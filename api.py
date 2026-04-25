"""
API для аналитической системы демографии РФ
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Импортируем существующие классы аналитики
from task1_monitoring import PopulationMonitor
from task2_forecasting import PopulationForecaster
from task3_ai_analytics import AIAnalytics

from typing import List, Optional
from pydantic import BaseModel

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