import logging
from docx.shared import Pt, Cm, Mm, Inches
from .models import DocumentFormattingError

logger = logging.getLogger(__name__)


def parse_measurement(value: str) -> object:
    """
    Парсинг размеров из строки (поддерживает mm, cm, pt, in).

    Args:
        value: Значение размера (строка или число).

    Returns:
        Объект размера (Pt, Mm, Cm, Inches).

    Raises:
        DocumentFormattingError: Если формат некорректен.
    """
    try:
        if isinstance(value, (int, float)):
            return Pt(float(value))
        value = str(value).strip().lower()
        if value.endswith('mm'):
            return Mm(float(value[:-2]))
        elif value.endswith('cm'):
            return Cm(float(value[:-2]))
        elif value.endswith('pt'):
            return Pt(float(value[:-2]))
        elif value.endswith('in') or value.endswith('"'):
            return Inches(float(value[:-2] if value.endswith('in') else value[:-1]))
        else:
            return Pt(float(value))
    except (ValueError, TypeError) as e:
        logger.error(f"Ошибка парсинга размера '{value}': {e}")
        raise DocumentFormattingError(f"Некорректный формат размера '{value}': {e}")


def parse_size(size_str: str) -> float:
    """Парсит строку с размером (поддерживает pt, px, mm) и возвращает в pt."""
    size_str = size_str.lower().strip()

    if size_str.endswith('pt'):
        return float(size_str[:-2])
    elif size_str.endswith('px'):
        return float(size_str[:-2]) * 0.75  # Примерное преобразование px в pt
    elif size_str.endswith('mm'):
        return Cm(float(size_str[:-2]) / 10).pt
    else:
        # Если нет суффикса, считаем что это pt
        return float(size_str)
