from __future__ import annotations

# Константа: число агентов
N_AGENTS: int = 9

# Константа: список описаний агентов
# Каждый агент имеет: id (0..8), name (короткое имя), description (что делает)
AGENTS = [
    {"id": 0, "name": "Code Agent (Python)", "description": "Пишет и исправляет Python-код (функции, скрипты)."},
    {"id": 1, "name": "SQL Agent", "description": "Пишет SQL-запросы по описанию задачи (SELECT/JOIN/GROUP BY)."},
    {"id": 2, "name": "Data Analysis Agent (Pandas)", "description": "Пишет код на Pandas для анализа таблиц (фильтры, агрегации)."},
    {"id": 3, "name": "Math Formula Solver", "description": "Решает математические задачи с формулами и вычислениями."},
    {"id": 4, "name": "Structured Extraction Agent (JSON)", "description": "Извлекает поля из текста и возвращает JSON по схеме."},
    {"id": 5, "name": "Summarization & Formatting Agent", "description": "Суммаризирует текст в заданном формате (буллеты, TL;DR)."},
    {"id": 6, "name": "Requirements / ТЗ Agent", "description": "Пишет ТЗ и требования по шаблону (FR/NFR/Acceptance Criteria)."},
    {"id": 7, "name": "Rewrite / Style Constraints Agent", "description": "Переписывает текст под ограничения (стиль, длина, формат)."},
    {"id": 8, "name": "Finance / Numeric Computation Agent", "description": "Делает прикладные расчёты (проценты, маржа, простые фин. формулы)."},
]


def agent_name(agent_id: int) -> str:
    """
    Возвращает имя агента по его id.
    Если id неверный — бросает ошибку ValueError.
    """
    if not (0 <= agent_id < N_AGENTS):
        raise ValueError(f"agent_id must be in [0, {N_AGENTS - 1}], got {agent_id}")
    return AGENTS[agent_id]["name"]
