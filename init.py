"""
Аналитическая система демографии РФ
"""

from .task1_monitoring import PopulationMonitor
from .task2_forecasting import PopulationForecaster
from .task3_ai_analytics import AIAnalytics

__all__ = ['PopulationMonitor', 'PopulationForecaster', 'AIAnalytics']