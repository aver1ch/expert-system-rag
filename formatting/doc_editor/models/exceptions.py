class DocumentFormattingError(Exception):
    """Базовое исключение для ошибок форматирования документа."""
    pass


class ConfigValidationError(DocumentFormattingError):
    """Ошибка валидации конфигурации."""
    pass


class ConfigParsingError(DocumentFormattingError):
    """Ошибка парсинга конфигурации."""
    pass


class ProcessorError(DocumentFormattingError):
    """Ошибка обработки документа."""
    pass
