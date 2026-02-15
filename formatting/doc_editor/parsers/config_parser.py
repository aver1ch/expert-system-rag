import yaml
import logging
from typing import Dict, Any
from pathlib import Path

from ..models import DocumentConfig, ConfigValidationError, ConfigParsingError

logger = logging.getLogger(__name__)


class ConfigParser:
    """Парсер конфигурации документа из YAML."""

    @staticmethod
    def from_file(config_path: str) -> DocumentConfig:
        """
        Загружает конфигурацию из YAML файла.

        Args:
            config_path: Путь к файлу конфигурации.

        Returns:
            Объект конфигурации.

        Raises:
            ConfigParsingError: Если файл не найден или некорректен.
            ConfigValidationError: Если валидация не пройдена.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            logger.info(f"YAML конфигурация загружена: {config_path}")
        except FileNotFoundError:
            logger.error(f"Файл конфигурации не найден: {config_path}")
            raise ConfigParsingError(f"Файл конфигурации не найден: {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Ошибка парсинга YAML: {e}")
            raise ConfigParsingError(f"Некорректный формат YAML: {e}")
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            raise ConfigParsingError(f"Ошибка загрузки конфигурации: {e}")

        return ConfigParser.from_dict(data)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> DocumentConfig:
        """
        Создаёт объект конфигурации из словаря.

        Args:
            data: Словарь с конфигурацией.

        Returns:
            Объект конфигурации.

        Raises:
            ConfigValidationError: Если валидация не пройдена.
        """
        try:
            ConfigParser._validate(data)
            config = DocumentConfig.from_dict(data)
            logger.info("Конфигурация успешно распарсена и валидирована")
            return config
        except ConfigValidationError:
            raise
        except Exception as e:
            logger.error(f"Ошибка обработки конфигурации: {e}")
            raise ConfigValidationError(f"Ошибка обработки конфигурации: {e}")

    @staticmethod
    def _validate(data: Dict[str, Any]) -> None:
        """
        Валидирует обязательные поля конфигурации.

        Args:
            data: Словарь с конфигурацией.

        Raises:
            ConfigValidationError: Если отсутствуют обязательные поля.
        """
        required_paths = [
            ['document', 'general', 'margins'],
            ['document', 'general', 'fonts'],
            ['document', 'general', 'spacing'],
            ['document', 'structure'],
        ]

        for path in required_paths:
            current = data
            path_str = '.'.join(path)
            
            for key in path:
                if not isinstance(current, dict) or key not in current:
                    logger.error(f"Отсутствует обязательное поле: {path_str}")
                    raise ConfigValidationError(f"Отсутствует обязательное поле: {path_str}")
                current = current[key]

        logger.debug("Валидация конфигурации пройдена успешно")
