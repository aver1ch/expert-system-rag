"""
AppendixProcessor - Компонент для управления приложениями.

Отвечает за:
- Поиск и идентификацию приложений в документе
- Нумерацию приложений буквами (А, Б, В...) или числами (1, 2, 3...)
- Форматирование заголовков приложений
- Интеграцию приложений в структуру документа
"""

import logging
from typing import List, Optional, Tuple
from docx.document import Document
from docx.oxml import OxmlElement
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

logger = logging.getLogger(__name__)


class AppendixProcessor:
    """
    Обработчик приложений документа.
    
    Находит разделы документа, которые должны быть обработаны как приложения,
    и применяет соответствующую нумерацию и форматирование.
    """
    
    def __init__(self, config):
        """
        Инициализация процессора приложений.
        
        Args:
            config: DocumentConfig с настройками приложений
                   (enabled, numbering_style)
        """
        self.config = config
        self.logger = logger
        self.logger.debug("AppendixProcessor инициализирован")
    
    def process_appendices(self, document: Document) -> None:
        """
        Обработать приложения в документе.
        
        Находит все приложения (секции после основного содержимого) и применяет
        соответствующую нумерацию и форматирование согласно конфигурации.
        
        Args:
            document: Документ Word для обработки
        """
        if not self.config.structure.document_structure.appendix.enabled:
            self.logger.debug("Приложения отключены в конфигурации")
            return
        
        try:
            appendix_headings = self._find_appendix_headings(document)
            
            if not appendix_headings:
                self.logger.debug("Приложения не найдены в документе")
                return
            
            self.logger.info(f"Найдено приложений: {len(appendix_headings)}")
            
            # Apply numbering to appendices
            self._apply_appendix_numbering(document, appendix_headings)
            
            self.logger.info("Приложения обработаны успешно")
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке приложений: {str(e)}", 
                            exc_info=True)
            raise
    
    def _find_appendix_headings(self, document: Document) -> List[Tuple[int, str]]:
        """
        Найти все приложения (заголовки приложений) в документе.
        
        Ищет paragraphs, которые:
        - Имеют стиль Heading (любой уровень)
        - Содержат слова "Appendix", "Приложение" или похожие
        - Находятся в конце документа (после основного содержимого)
        
        Args:
            document: Документ для поиска
        
        Returns:
            List[Tuple[int, str]]: Список кортежей (индекс_параграфа, текст)
        """
        appendix_headings = []
        appendix_keywords = [
            'appendix', 'приложение', 'annex', 'приложении',
            'приложению', 'приложением', 'appendices', 'appendix',
            'appendix a', 'appendix b', 'приложение а', 'приложение б'
        ]
        
        for idx, paragraph in enumerate(document.paragraphs):
            # Check if paragraph is a heading
            if not paragraph.style.name.startswith('Heading'):
                continue
            
            # Check if contains appendix keywords
            text_lower = paragraph.text.lower().strip()
            
            if any(keyword in text_lower for keyword in appendix_keywords):
                appendix_headings.append((idx, paragraph.text))
                self.logger.debug(f"Найдено приложение: {paragraph.text}")
        
        return appendix_headings
    
    def _apply_appendix_numbering(self, document: Document, 
                                   appendix_headings: List[Tuple[int, str]]) -> None:
        """
        Применить нумерацию к приложениям.
        
        Обновляет текст приложений согласно выбранному стилю нумерации
        (буквы или числа).
        
        Args:
            document: Документ для обновления
            appendix_headings: Список приложений (индекс, текст)
        """
        numbering_style = self.config.structure.document_structure.appendix.numbering_style
        
        for app_number, (idx, original_text) in enumerate(appendix_headings):
            paragraph = document.paragraphs[idx]
            
            if numbering_style == "numbers":
                # Numeric numbering: 1, 2, 3...
                new_text = f"Приложение {app_number + 1}"
            else:
                # Letter numbering: А, Б, В... (русские буквы)
                letter = self._get_appendix_letter(app_number)
                new_text = f"Приложение {letter}"
            
            # Extract original content after "Appendix"/"Приложение" if present
            if ':' in original_text:
                description = original_text.split(':', 1)[1].strip()
                new_text = f"{new_text}: {description}"
            
            paragraph.text = new_text
            self.logger.debug(f"Приложение {app_number + 1} обновлено: {new_text}")
    
    def _get_appendix_letter(self, index: int) -> str:
        """
        Получить букву для приложения по индексу.
        
        Supports:
        - Russian: А, Б, В, ..., Я (кириллица)
        - English: A, B, C, ..., Z, AA, AB... (при необходимости)
        
        Args:
            index: Индекс приложения (0-based)
        
        Returns:
            str: Буква для приложения
        """
        # Russian letters (А-Я)
        russian_letters = 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
        
        if index < len(russian_letters):
            return russian_letters[index]
        
        # For index >= 33 (количество букв в русском алфавите), 
        # используем двойные буквы (АА, АБ, и т.д.)
        first_letter_idx = (index - len(russian_letters)) // len(russian_letters)
        second_letter_idx = (index - len(russian_letters)) % len(russian_letters)
        
        if first_letter_idx < len(russian_letters):
            return russian_letters[first_letter_idx] + russian_letters[second_letter_idx]
        
        # Fallback для очень больших индексов
        return str(index + 1)
