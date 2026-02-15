import logging
from docx import Document

from .models import DocumentConfig, DocumentFormattingError
from .processors import (
    StyleProcessor,
    HeaderFooterProcessor,
    TitleProcessor,
    MarginsProcessor,
    SectionProcessor,
    TOCProcessor,
    PrefaceProcessor,
    AppendixProcessor,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DocumentProcessingPipeline:
    """
    Орхестратор обработки документа.
    
    Координирует последовательность применения всех процессоров к документу.
    """

    def __init__(self, doc: Document, config: DocumentConfig):
        """
        Инициализация pipeline.

        Args:
            doc: Объект документа.
            config: Типизированная конфигурация.
        """
        self.doc = doc
        self.config = config
        self.logger = logger

    def execute(self, add_title_page: bool = True) -> None:
        """
        Выполняет полный pipeline обработки документа.

        Args:
            add_title_page: Добавлять ли титульный лист.

        Raises:
            DocumentFormattingError: Если обработка не удалась.
        """
        try:
            self.logger.info("Начало выполнения pipeline обработки документа")

            # Фаза 1: Применение стилей и полей
            self._apply_styles_and_margins()

            # Фаза 2: Построение структуры (титульный лист)
            if add_title_page:
                self._apply_title_page()

            # Фаза 3: Повторное применение настроек к новому документу
            self._apply_settings_after_structure()

            self.logger.info("Pipeline обработки успешно завершен")
        except Exception as e:
            self.logger.error(f"Ошибка выполнения pipeline: {e}")
            raise DocumentFormattingError(f"Ошибка выполнения pipeline: {e}")

    def _apply_styles_and_margins(self) -> None:
        """Применяет стили и поля к документу."""
        self.logger.info("Этап 1: Применение стилей и полей")

        # Применяем стили
        style_processor = StyleProcessor(self.doc, self.config)
        style_processor.apply()

        # Применяем поля
        margins_processor = MarginsProcessor(self.doc, self.config)
        margins_processor.apply()

    def _apply_title_page(self) -> None:
        """Добавляет титульный лист к документу."""
        self.logger.info("Этап 2: Добавление титульного листа")

        # Сохраняем текущий документ
        temp_path = "temp_with_styles.docx"
        self.doc.save(temp_path)

        # Добавляем титульный лист
        title_processor = TitleProcessor(self.config)
        output_with_title = "doc_with_title.docx"
        title_processor.apply(temp_path, output_with_title)

        # Загружаем документ с титулом
        self.doc = Document(output_with_title)

    def _apply_settings_after_structure(self) -> None:
        """Повторно применяет настройки после изменения структуры документа."""
        self.logger.info("Этап 3: Повторное применение настроек")

        # Применяем поля и колонтитулы к новому документу
        margins_processor = MarginsProcessor(self.doc, self.config)
        margins_processor.apply()

        header_footer_processor = HeaderFooterProcessor(self.doc, self.config)
        header_footer_processor.apply()
        
        # Фаза 2: Многоуровневая нумерация разделов
        self.logger.info("Этап 4: Применение многоуровневой нумерации разделов (Фаза 2)")
        section_processor = SectionProcessor(self.config)
        section_processor.apply_section_numbering(self.doc)
        
        # Фаза 2: Автоматическое построение оглавления
        self.logger.info("Этап 5: Построение оглавления (Фаза 2)")
        toc_processor = TOCProcessor(self.config)
        toc_processor.create_toc(self.doc)
        
        # Фаза 2: Добавление предисловия
        self.logger.info("Этап 6: Добавление предисловия (Фаза 2)")
        preface_processor = PrefaceProcessor(self.config)
        preface_processor.add_preface(self.doc)
        
        # Фаза 2: Обработка приложений
        self.logger.info("Этап 7: Обработка приложений (Фаза 2)")
        appendix_processor = AppendixProcessor(self.config)
        appendix_processor.process_appendices(self.doc)

    def get_document(self) -> Document:
        """Возвращает обработанный документ."""
        return self.doc
