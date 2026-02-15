from docx import Document
import logging
from typing import Optional

from .parsers import ConfigParser
from .models import DocumentConfig, DocumentFormattingError
from .pipeline import DocumentProcessingPipeline

# Настройка локального логгера
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


class DocumentEditor:
    """Главный интерфейс для редактирования документов."""

    def __init__(self, doc_path: str):
        """
        Инициализация редактора документа.

        Args:
            doc_path: Путь к файлу документа.

        Raises:
            DocumentFormattingError: Если документ не удалось загрузить.
            TypeError: Если doc_path не строка.
        """
        self.logger = logger
        
        if not isinstance(doc_path, str):
            self.logger.error(f"doc_path must be a string, got {type(doc_path)}")
            raise TypeError("doc_path must be a string")

        try:
            self.doc = Document(doc_path)
            self.doc_path = doc_path
            self.config: Optional[DocumentConfig] = None
            self.pipeline: Optional[DocumentProcessingPipeline] = None
            self.logger.info(f"Документ загружен: {doc_path}")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки документа {doc_path}: {e}")
            raise DocumentFormattingError(f"Не удалось загрузить документ: {e}")

    def load_config(self, config_path: str) -> None:
        """
        Загрузка конфигурации из YAML файла.

        Args:
            config_path: Путь к файлу конфигурации.

        Raises:
            DocumentFormattingError: Если конфигурация не загружена корректно.
            TypeError: Если config_path не строка.
        """
        if not isinstance(config_path, str):
            self.logger.error(f"config_path must be a string, got {type(config_path)}")
            raise TypeError("config_path must be a string")

        try:
            self.config = ConfigParser.from_file(config_path)
            self.logger.info(f"Конфигурация загружена: {config_path}")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки конфигурации: {e}")
            raise DocumentFormattingError(f"Ошибка загрузки конфигурации: {e}")

    def apply_config(self) -> None:
        """
        Применение конфигурации к документу.

        Raises:
            ValueError: Если конфигурация не загружена.
            DocumentFormattingError: Если применение конфигурации не удалось.
        """
        if not self.config:
            self.logger.error("Конфигурация не загружена")
            raise ValueError("Конфигурация не загружена")

        try:
            self.logger.info("Начало применения конфигурации к документу")
            
            # Создаем и выполняем pipeline
            self.pipeline = DocumentProcessingPipeline(self.doc, self.config)
            self.pipeline.execute(add_title_page=True)
            
            # Получаем обработанный документ
            self.doc = self.pipeline.get_document()
            
            self.logger.info("Конфигурация успешно применена к документу")
        except Exception as e:
            self.logger.error(f"Ошибка применения конфигурации: {e}")
            raise DocumentFormattingError(f"Ошибка применения конфигурации: {e}")

    def save(self, output_path: str) -> None:
        """
        Сохранение документа.

        Args:
            output_path: Путь для сохранения файла.

        Raises:
            DocumentFormattingError: Если сохранение не удалось.
            TypeError: Если output_path не строка.
        """
        if not isinstance(output_path, str):
            self.logger.error(f"output_path must be a string, got {type(output_path)}")
            raise TypeError("output_path must be a string")

        try:
            self.doc.save(output_path)
            self.logger.info(f"Документ сохранен: {output_path}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения документа {output_path}: {e}")
            raise DocumentFormattingError(f"Ошибка сохранения документа: {e}")

    def get_document(self) -> Document:
        """
        Получение объекта обработанного документа.

        Returns:
            Document: Обработанный документ.
        """
        return self.doc
