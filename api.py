"""
API для аналитической системы демографии РФ
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Импортируем существующие классы аналитики
from task1_monitoring import PopulationMonitor
from task2_forecasting import PopulationForecaster
from task3_ai_analytics import AIAnalytics

# Создаём экземпляры классов (они загрузят данные при старте)
monitor = PopulationMonitor()
forecaster = PopulationForecaster()
ai_analytics = AIAnalytics()

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