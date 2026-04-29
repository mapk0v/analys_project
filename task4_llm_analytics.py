"""
Модуль для генерации демографических отчётов с помощью GigaChat API
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from gigachat import GigaChat
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла (если есть)
from pathlib import Path
env_path = Path(__file__).parent / "key.env"
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)


class LLMAnalytics:
    """Анализатор с использованием GigaChat для генерации отчётов"""

    def __init__(self):
        """Инициализация клиента GigaChat"""
        self.api_key = os.getenv("GIGACHAT_API_KEY")
        if not self.api_key:
            logger.warning("GIGACHAT_API_KEY не задан в переменных окружения. "
                           "Используйте .env файл или установите переменную.")

        self.model = os.getenv("GIGACHAT_MODEL", "GigaChat:latest")
        self.llm = None

        if self.api_key:
            try:
                self.llm = GigaChat(
                    credentials=self.api_key,
                    verify_ssl_certs=False,   # Для локальной разработки (можно True при наличии сертификатов)
                    model=self.model,
                    timeout=30,
                )
                logger.info("GigaChat клиент успешно инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации GigaChat: {e}")
                self.llm = None
        else:
            logger.error("Нет API-ключа, GigaChat недоступен")

    def _get_city_region(self, city_name: str) -> str:
        """Получение региона города из существующих данных (заглушка – можно расширить)"""
        # Этот метод можно реализовать через monitor.get_municipality_population(),
        # но для простоты вернём пустую строку. В реальном проекте лучше передавать регион извне.
        return ""

    def _get_cagr_from_forecast(self, forecast: Dict) -> float:
        """Возвращает прогнозный CAGR в процентах (например, 0.48 вместо 0.0048)"""
        if forecast.get('metrics') and 'cagr' in forecast['metrics']:
            return float(forecast['metrics']['cagr']) * 100  # в процентах

        predictions = forecast.get('predictions', [])
        if len(predictions) < 2:
            return 0.0

        first = predictions[0]
        last = predictions[-1]
        years = len(predictions)
        if first and last and years > 1:
            cagr = (last / first) ** (1.0 / (years - 1)) - 1.0
            return cagr * 100  # в процентах
        return 0.0

    def _get_mape_from_forecast(self, forecast: Dict) -> Optional[float]:
        """Возвращает MAPE из метрик или None"""
        if forecast.get('metrics') and 'mape' in forecast['metrics']:
            return float(forecast['metrics']['mape'])
        return None

    def _build_analytics_prompt(self, city_name: str, dynamics: Dict, forecast: Dict) -> str:
        """Создаёт промпт для LLM с данными по городу"""
        abs_change = dynamics.get('absolute_change', 0)
        trend = "рост" if abs_change >= 0 else "снижение"

        predictions = forecast.get('predictions', [])
        forecast_5y = predictions[4] if len(predictions) >= 5 else None
        forecast_10y = predictions[9] if len(predictions) >= 10 else None
        mape = self._get_mape_from_forecast(forecast)
        forecast_cagr = self._get_cagr_from_forecast(forecast)

        # Безопасное получение cagr из метрик (если есть)
        metrics_cagr = forecast.get('metrics', {}).get('cagr', 0) if forecast.get('metrics') else 0

        prompt = f"""
    Ты — эксперт-демограф. Напиши аналитический отчёт по городу {city_name} в строго определённом JSON-формате.

    **Данные для анализа:**
    - Период: {dynamics.get('start_year', 'Н/Д')}–{dynamics.get('end_year', 'Н/Д')}
    - Численность на начало периода: {dynamics.get('start_population', 'Н/Д'):,} чел.
    - Численность на конец периода: {dynamics.get('end_population', 'Н/Д'):,} чел.
    - Абсолютное изменение: {abs_change:+,} чел. ({trend})
    - Относительное изменение: {dynamics.get('relative_change', 0):+.2f}%
    - Среднегодовой темп роста (CAGR) за исторический период: {dynamics.get('cagr', 0):+.2f}%
    - Прогноз на 5 лет: {forecast_5y if forecast_5y else 'нет данных':,} чел.
    - Прогноз на 10 лет: {forecast_10y if forecast_10y else 'нет данных':,} чел.
    - Прогнозный CAGR на 10 лет: {forecast_cagr:+.4f}%
    - Качество прогноза (MAPE): {mape if mape else 'нет данных'}%

    **Структура ответа (JSON):**
    {{
        "city": "{city_name}",
        "region": "название региона",
        "generated_at": "{datetime.now().isoformat()}",
        "section_31_summary": "Краткое резюме с переносами строк (как в примере). Используй **жирный** для чисел и выводов.",
        "section_32_trends_and_factors": {{
            "trends": ["тенденция 1", "тенденция 2"],
            "factors": [
                {{"name": "Название фактора", "description": "Описание"}}
            ]
        }},
        "section_33_forecast": {{
            "current_population": {dynamics.get('end_population', 0)},
            "forecast_5y": {forecast_5y if forecast_5y else 0},
            "forecast_10y": {forecast_10y if forecast_10y else 0},
            "cagr": {metrics_cagr}
        }},
        "section_34_recommendations": [
            {{
                "title": "Название рекомендации",
                "priority": "medium",
                "category": "КАТЕГОРИЯ",
                "description": "Описание"
            }}
        ],
        "section_35_conclusion": "Итоговое заключение (2-3 абзаца, **выдели** выводы)"
    }}

    **Пример форматирования для section_31_summary:**
    **Москва** демонстрирует рост численности населения.

    За период 2015-2021 население изменилось с **12,000,000** до **12,500,000** человек.

    **Абсолютное изменение:** +500,000 чел.
    **Относительное изменение:** +4.2%
    **Среднегодовой темп роста (CAGR):** +0.68%

    **Важно:**
    - Все числа должны быть числами (без кавычек).
    - Приоритет рекомендаций: critical, high, medium, low.
    - Используй **жирный** в текстовых полях.
    - Ответ должен содержать ТОЛЬКО JSON.
    - В "section_31_summary" обязательно используй переносы строк (\\n) для читаемости.
    """
        return prompt

    def _parse_llm_response(self, llm_text: str, city_name: str, dynamics: Dict, forecast: Dict) -> Dict[str, Any]:
        """Парсит JSON-ответ LLM и возвращает структурированный отчёт, заменяя cagr на реальное значение"""
        try:
            json_start = llm_text.find('{')
            json_end = llm_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_text[json_start:json_end]
                report = json.loads(json_str)
                # Заполняем обязательные поля, если их нет
                report.setdefault('city', city_name)
                report.setdefault('region', self._get_city_region(city_name))
                report.setdefault('generated_at', datetime.now().isoformat())
                report.setdefault('section_31_summary', "Нет данных для резюме")
                report.setdefault('section_32_trends_and_factors', {"trends": [], "factors": []})

                # Базовый прогноз из наших данных (гарантированно правильный)
                predictions = forecast.get('predictions', [])
                default_forecast = {
                    "current_population": dynamics.get('end_population', 0),
                    "forecast_5y": predictions[4] if len(predictions) >= 5 else 0,
                    "forecast_10y": predictions[9] if len(predictions) >= 10 else 0,
                    "cagr": self._get_cagr_from_forecast(forecast)  # вычисляем реальный CAGR
                }
                # Если в ответе LLM нет section_33_forecast, добавляем default
                if 'section_33_forecast' not in report:
                    report['section_33_forecast'] = default_forecast
                else:
                    # Обновляем недостающие ключи, но главное – перезаписываем cagr
                    for key, value in default_forecast.items():
                        if key not in report['section_33_forecast']:
                            report['section_33_forecast'][key] = value
                    # Принудительно заменяем cagr на вычисленное значение
                    report['section_33_forecast']['cagr'] = default_forecast['cagr']

                report.setdefault('section_34_recommendations', [])
                report.setdefault('section_35_conclusion', "Заключение отсутствует")
                return report
        except Exception as e:
            logger.error(f"Ошибка парсинга JSON: {e}")

        return self._get_fallback_report(city_name, dynamics, forecast)

    def _get_fallback_report(self, city_name: str, dynamics: Dict, forecast: Dict) -> Dict[str, Any]:
        """Возвращает отчёт-заглушку на основе переданных данных (без LLM)"""
        abs_change = dynamics.get('absolute_change', 0)
        trend_text = "рост" if abs_change >= 0 else "снижение"
        current_pop = dynamics.get('end_population', 0)

        predictions = forecast.get('predictions', [])
        forecast_5y = predictions[4] if len(predictions) >= 5 else 0
        forecast_10y = predictions[9] if len(predictions) >= 10 else 0
        forecast_cagr = self._get_cagr_from_forecast(forecast)
        mape = self._get_mape_from_forecast(forecast)

        summary = (
            f"**{city_name}** демонстрирует {trend_text} численности населения.\n\n"
            f"За период {dynamics.get('start_year')}-{dynamics.get('end_year')} "
            f"население изменилось с **{dynamics.get('start_population', 0):,}** до **{current_pop:,}** человек.\n\n"
            f"**Абсолютное изменение:** {abs_change:+,} чел.\n"
            f"**Относительное изменение:** {dynamics.get('relative_change', 0):+.2f}%\n"
            f"**Среднегодовой темп роста (исторический):** {dynamics.get('cagr', 0):+.2f}%\n"
            f"**Прогнозный CAGR:** {forecast_cagr:+.4f}%"
        )

        # Тенденции
        if abs_change > 0:
            trends = ["Положительная динамика роста населения"]
            if dynamics.get('cagr', 0) > 0.5:
                trends.append("Умеренный устойчивый рост")
            else:
                trends.append("Слабый положительный рост")
        else:
            trends = ["Отрицательная динамика убыли населения"]
            if dynamics.get('cagr', 0) < -0.5:
                trends.append("Ускоренная убыль населения")
            else:
                trends.append("Незначительное сокращение численности")

        conclusion = (
            f"**Заключение по демографической ситуации в городе {city_name}**\n\n"
            f"Анализ показывает, что город находится в фазе **{trend_text}** численности населения.\n\n"
            f"**Ключевые выводы:**\n"
            f"1. Динамика населения: {dynamics.get('relative_change', 0):+.2f}% за период.\n"
            f"2. Среднегодовой темп роста (исторический): {dynamics.get('cagr', 0):+.2f}%.\n"
            f"3. Прогнозный CAGR на 10 лет: {forecast_cagr:+.4f}%.\n"
            f"4. Прогноз на {dynamics.get('end_year', 0)+10} год: {forecast_10y:,} чел.\n"
            f"5. Качество прогноза: MAPE = {mape if mape else 0:.2f}%.\n\n"
            f"**Приоритетные направления:**\n"
            f"- Регулярный мониторинг демографических показателей\n"
            f"- Адаптация стратегий на основе актуальных данных"
        )

        return {
            "city": city_name,
            "region": self._get_city_region(city_name),
            "generated_at": datetime.now().isoformat(),
            "section_31_summary": summary,
            "section_32_trends_and_factors": {
                "trends": trends,
                "factors": [
                    {"name": "Экономическая активность", "description": "Влияние на миграционный прирост"},
                    {"name": "Рождаемость и смертность", "description": "Естественное движение населения"}
                ]
            },
            "section_33_forecast": {
                "current_population": current_pop,
                "forecast_5y": forecast_5y,
                "forecast_10y": forecast_10y,
                "cagr": forecast_cagr
            },
            "section_34_recommendations": [
                {
                    "title": "Сбалансированное развитие",
                    "priority": "medium",
                    "category": "РАЗВИТИЕ",
                    "description": "Ожидается умеренное изменение численности населения"
                },
                {
                    "title": "Регулярный анализ демографической ситуации",
                    "priority": "medium",
                    "category": "МОНИТОРИНГ",
                    "description": "Для своевременного реагирования на изменения"
                }
            ],
            "section_35_conclusion": conclusion
        }

    def generate_report(self, city_name: str, dynamics: Dict, forecast: Dict, metrics: Dict = None) -> Dict[str, Any]:
        """Генерация отчёта с возможностью передать метрики отдельно"""
        # Если метрики переданы, добавляем их в копию forecast
        if metrics:
            # Создаём копию, чтобы не менять оригинал
            forecast_with_metrics = forecast.copy()
            forecast_with_metrics['metrics'] = metrics
        else:
            forecast_with_metrics = forecast

        if not self.llm:
            logger.warning("GigaChat недоступен, используется fallback-отчёт")
            return self._get_fallback_report(city_name, dynamics, forecast_with_metrics)

        prompt = self._build_analytics_prompt(city_name, dynamics, forecast_with_metrics)
        try:
            response = self.llm.chat(prompt)
            if hasattr(response, 'choices') and response.choices:
                llm_output = response.choices[0].message.content
            elif hasattr(response, 'messages') and response.messages:
                llm_output = response.messages[0].content
            else:
                llm_output = str(response)

            report = self._parse_llm_response(llm_output, city_name, dynamics, forecast)
            logger.info(f"Отчёт для {city_name} успешно сгенерирован через GigaChat")
            return report
        except Exception as e:
            logger.exception(f"Ошибка при вызове GigaChat: {e}")
            return self._get_fallback_report(city_name, dynamics, forecast_with_metrics)
