import logging
from docx import Document

from ..models import DocumentConfig, ProcessorError
from ..utils import parse_measurement

logger = logging.getLogger(__name__)


class MarginsProcessor:
    """Обработчик полей документа."""

    def __init__(self, doc: Document, config: DocumentConfig):
        """
        Инициализация процессора полей.

        Args:
            doc: Объект документа.
            config: Конфигурация (типизированный объект).
        """
        self.doc = doc
        self.config = config
        self.logger = logger

    def apply(self) -> None:
        """Применяет настройки полей документа."""
        try:
            self.logger.info("Начало применения полей документа")
            margins_config = self.config.general.margins

            for i, section in enumerate(self.doc.sections):
                section.left_margin = parse_measurement(margins_config.left)
                section.right_margin = parse_measurement(margins_config.right)
                section.top_margin = parse_measurement(margins_config.top)
                section.bottom_margin = parse_measurement(margins_config.bottom)

                self.logger.debug(
                    f"Поля установлены для секции {i + 1}: "
                    f"левое={margins_config.left}, правое={margins_config.right}, "
                    f"верхнее={margins_config.top}, нижнее={margins_config.bottom}"
                )

            self.logger.info(f"Поля успешно применены к {len(self.doc.sections)} секциям")
        except Exception as e:
            self.logger.error(f"Ошибка применения полей: {e}")
            raise ProcessorError(f"Ошибка применения полей: {e}")
