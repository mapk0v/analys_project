"""
Главный скрипт для запуска аналитической системы
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from task1_monitoring import PopulationMonitor
from task2_forecasting import PopulationForecaster
from task3_ai_analytics import AIAnalytics
from utils import save_report
from config import REPORTS_DIR


def print_separator():
    print("\n" + "-"*50)


def main():
    print("\n" + "="*70)
    print("АНАЛИТИЧЕСКАЯ СИСТЕМА ДЕМОГРАФИИ РОССИИ")
    print("="*70 + "\n")

    monitor = PopulationMonitor()
    forecaster = PopulationForecaster()
    ai = AIAnalytics()

    # ==================== ЗАДАНИЕ 1: МОНИТОРИНГ ====================
    print("="*70)
    print("ЗАДАНИЕ 1: МОНИТОРИНГ НАСЕЛЕНИЯ")
    print("="*70)

    # 1.1 Топ городов
    print("\n1.1 ТОП-20 КРУПНЕЙШИХ ГОРОДОВ РОССИИ:")
    top_cities = monitor.get_municipality_population()
    for i, city in enumerate(top_cities[:20], 1):
        print(f"   {i:2d}. {city['name']}: {city['population']:,} чел. ({city['type']})")

    # 1.2 Динамика
    print("\n1.2 ДИНАМИКА НАСЕЛЕНИЯ ЗА 2015-2024:")
    all_dynamics = monitor.get_all_population_dynamics()
    print("\n   ТОП-10 ПО РОСТУ:")
    for city in all_dynamics[:10]:
        print(f"      {city['city']}: {city['start_population']:,} → {city['end_population']:,} чел. (+{city['relative_change']:.1f}%)")

    print("\n   ТОП-10 ПО СНИЖЕНИЮ:")
    for city in all_dynamics[-10:]:
        print(f"      {city['city']}: {city['start_population']:,} → {city['end_population']:,} чел. ({city['relative_change']:.1f}%)")

    # 1.3 Демографические показатели
    print("\n1.3 ДЕМОГРАФИЧЕСКИЕ ПОКАЗАТЕЛИ (КРУПНЕЙШИЕ ГОРОДА):")
    all_demo = monitor.get_all_demographic_indicators()
    for demo in all_demo[:15]:
        print(f"\n   {demo['city']}:")
        print(f"      Население: {demo['population']:,} чел.")
        print(f"      Рождаемость: {demo['birth_rate']}‰, Смертность: {demo['death_rate']}‰")
        print(f"      Естественный прирост: {demo['natural_increase']:+.1f}‰")

    # 1.4 Топ роста/снижения
    print("\n1.4 МУНИЦИПАЛИТЕТЫ С НАИБОЛЬШИМ РОСТОМ/СНИЖЕНИЕМ:")
    top_changes = monitor.get_top_growing_declining(n=10, min_population=500000)

    print("\n   ТОП-10 РАСТУЩИХ (население > 500k):")
    for city in top_changes['growing']:
        print(f"      {city['city']}: +{city['relative_change']:.1f}%")

    print("\n   ТОП-10 УБЫВАЮЩИХ (население > 500k):")
    for city in top_changes['declining']:
        print(f"      {city['city']}: {city['relative_change']:.1f}%")

    # 1.5 Тепловая карта
    print("\n1.5 ТЕПЛОВАЯ КАРТА ПО РЕГИОНАМ:")
    heatmap = monitor.create_regional_heatmap()
    save_report(heatmap, 'regional_heatmap', 'html')
    print("   Тепловая карта сохранена")

    # ==================== ЗАДАНИЕ 2: ПРОГНОЗИРОВАНИЕ ====================
    print("\n" + "="*70)
    print("ЗАДАНИЕ 2: ПРОГНОЗИРОВАНИЕ ЧИСЛЕННОСТИ НАСЕЛЕНИЯ")
    print("="*70)

    # Берем первые 5 городов для демонстрации
    demo_cities = [c['name'] for c in top_cities[:5]]

    for idx, city in enumerate(demo_cities, 1):
        print(f"\n{'='*50}")
        print(f"ГОРОД: {city.upper()}")
        print(f"{'='*50}")

        forecast = forecaster.forecast_city(city, horizon=15)

        if forecast:
            # 2.1 Прогноз на 5-15 лет
            print("\n2.1 ПРОГНОЗ НА 5-15 ЛЕТ:")
            print(f"   Исходные данные (2023): {forecast['last_population']:,} чел.")
            print(f"   Прогноз на 2028 (5 лет): {int(forecast['predictions'][4]):,} чел.")
            print(f"   Прогноз на 2033 (10 лет): {int(forecast['predictions'][9]):,} чел.")
            print(f"   Прогноз на 2038 (15 лет): {int(forecast['predictions'][14]):,} чел.")

            # 2.2 График
            print("\n2.2 ГРАФИК ПРОГНОЗА:")
            chart = forecaster.create_forecast_chart(forecast)
            if chart:
                chart_path = save_report(chart, f'{city.lower()}_forecast_chart', 'html')
                print(f"   График сохранен: {chart_path.name}")

            # 2.3 Прогнозный интервал
            print("\n2.3 ПРОГНОЗНЫЙ ИНТЕРВАЛ (95% доверительный интервал):")
            if forecast.get('lower_bound') and forecast.get('upper_bound'):
                print(f"   2028: [{int(forecast['lower_bound'][4]):,} - {int(forecast['upper_bound'][4]):,}] чел.")
                print(f"   2033: [{int(forecast['lower_bound'][9]):,} - {int(forecast['upper_bound'][9]):,}] чел.")
                print(f"   2038: [{int(forecast['lower_bound'][14]):,} - {int(forecast['upper_bound'][14]):,}] чел.")
            else:
                std_error = forecast.get('residuals_std', forecast['last_population'] * 0.01)
                print(f"   2028: [{int(forecast['predictions'][4] - 1.96*std_error):,} - {int(forecast['predictions'][4] + 1.96*std_error):,}] чел.")
                print(f"   2033: [{int(forecast['predictions'][9] - 1.96*std_error):,} - {int(forecast['predictions'][9] + 1.96*std_error):,}] чел.")
                print(f"   2038: [{int(forecast['predictions'][14] - 1.96*std_error):,} - {int(forecast['predictions'][14] + 1.96*std_error):,}] чел.")

            # 2.4 Метрики качества
            print("\n2.4 МЕТРИКИ КАЧЕСТВА ПРОГНОЗА:")
            metrics = forecaster.calculate_metrics(forecast)
            if metrics:
                print(f"   MAPE (средняя абсолютная процентная ошибка): {metrics['mape']:.2f}%")
                print(f"   RMSE (среднеквадратичная ошибка): {metrics['rmse']:,} чел.")
                print(f"   MAE (средняя абсолютная ошибка): {metrics['mae']:,} чел.")
                print(f"   R² (коэффициент детерминации): {metrics['r2']:.4f}")
                print(f"\n   Интерпретация: {metrics['interpretation']}")
        else:
            print(f"\n   Недостаточно данных для прогнозирования города {city}")

    # ==================== ЗАДАНИЕ 3: AI-АНАЛИТИКА ====================
    print("\n" + "="*70)
    print("ЗАДАНИЕ 3: AI-АНАЛИТИКА")
    print("="*70)

    for idx, city in enumerate(demo_cities, 1):
        print(f"\n{'='*50}")
        print(f"ГОРОД: {city.upper()}")
        print(f"{'='*50}")

        report = ai.generate_full_report(city, forecast_horizon=10)

        if report:
            # 3.1 Краткое резюме
            print("\n3.1 КРАТКОЕ РЕЗЮМЕ ДИНАМИКИ НАСЕЛЕНИЯ:")
            print(f"   {report['section_31_summary']}")

            # 3.2 Демографические тенденции и факторы
            print("\n3.2 ДЕМОГРАФИЧЕСКИЕ ТЕНДЕНЦИИ И ФАКТОРЫ ВЛИЯНИЯ:")
            print("\n   Выявленные тенденции:")
            for trend in report['section_32_trends_and_factors']['trends']:
                print(f"      - {trend}")

            print("\n   Факторы влияния:")
            for factor in report['section_32_trends_and_factors']['factors']:
                print(f"      - {factor['name']}: {factor['description']}")

            # 3.3 Прогнозная оценка
            print("\n3.3 ПРОГНОЗНАЯ ОЦЕНКА НА 10 ЛЕТ:")
            print(f"   {report['section_33_forecast']['assessment']}")

            # 3.4 Рекомендации
            print("\n3.4 РЕКОМЕНДАЦИИ ПО СОЦИАЛЬНОЙ ПОЛИТИКЕ И ТЕРРИТОРИАЛЬНОМУ ПЛАНИРОВАНИЮ:")
            for rec in report['section_34_recommendations']:
                priority = "КРИТИЧЕСКИЙ" if rec['priority'] == 'critical' else "ВЫСОКИЙ" if rec['priority'] == 'high' else "СРЕДНИЙ"
                print(f"\n   [{priority}] {rec['title']}")
                print(f"   Категория: {rec['category']}")
                print("   Рекомендуемые действия:")
                for action in rec['actions'][:4]:
                    print(f"      - {action}")

            # Сохраняем отчеты
            save_report(report, f'{city.lower()}_analytics_report', 'json')

            # Markdown отчет
            md_report = ai.generate_markdown_report(city, forecast_horizon=10)
            if md_report:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                md_path = REPORTS_DIR / f"{city.lower()}_analytics_report_{timestamp}.md"
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(md_report)
                print(f"\n   Markdown отчет сохранен: {md_path.name}")
        else:
            print(f"\n   Не удалось сгенерировать отчет для {city}")

    print("\n" + "="*70)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print(f"Отчеты в: {REPORTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()