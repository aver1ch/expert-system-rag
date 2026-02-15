# Система редактирования и форматирования DOCX документов

Микросервис для автоматизированной обработки и форматирования документов Microsoft Word (DOCX) с поддержкой конфигурируемых стилей, титульных листов, колонтитулов и полей страницы.

## Быстрый старт

### 1. Установка зависимостей

```bash
python -m venv .venv
source .venv/bin/activate  # на Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Локальное тестирование

```bash
python local.py
```

Результат сохранится в `output.docx`.

### 3. Запуск Flask микросервиса

```bash
python main.py
```

Сервис будет доступен на `http://localhost:5000`.

---

## Использование

### Локальный API (Python)

```python
from doc_editor.editor import DocumentEditor

# Загрузить документ и конфигурацию
editor = DocumentEditor('input.docx')
editor.load_config('config.yaml')

# Применить все настройки
editor.apply_config()

# Сохранить результат
editor.save('output.docx')
```

### REST API

#### Endpoint: `POST /api/process_document`

**Request:**
```json
{
  "document_url": "https://example.com/document.docx",
  "config": {
    "document": {
      "general": {
        "margins": {
          "left": "20mm",
          "right": "10mm",
          "top": "20mm",
          "bottom": "20mm"
        },
        "fonts": {
          "main": {
            "family": "Arial",
            "size": "12pt"
          }
        },
        "spacing": {
          "line": 1.5
        }
      },
      "structure": {
        "title_page": {
          "enabled": true,
          "template_path": "doc_editor/templates/title_page_template.docx",
          "image_path": "doc_editor/templates/logo.png",
          "elements": [
            {"agency_name": "ООО Организация"},
            {"title": "ГОСТ Р 1.5-2004"}
          ]
        }
      },
      "numbering": {
        "headers": {
          "enabled": true,
          "left": "Левый колонтитул",
          "right": "Правый колонтитул",
          "page_numbers": true
        }
      }
    }
  }
}
```

**Response:**
```
200 OK
Content-Type: application/vnd.openxmlformats-officedocument.wordprocessingml.document
[binary DOCX file]
```

---

## Конфигурация (YAML)

### Структура конфигурационного файла

```yaml
document:
  # Общие настройки
  general:
    # Поля страницы
    margins:
      left: "20mm"
      right: "10mm"
      top: "20mm"
      bottom: "20mm"
    
    # Шрифты
    fonts:
      main:
        family: Arial
        size: 12pt
        bold: false
        italic: false
      
      # Стили для заголовков
      header1:
        family: Arial
        size: 14pt
        bold: true
      
      header2:
        family: Arial
        size: 13pt
        bold: true
      
      header3:
        family: Arial
        size: 12pt
        bold: true
      
      # Стили для специальных элементов
      appendices:
        family: Arial
        size: 11pt
      
      notes:
        family: Arial
        size: 10pt
      
      # Количество уровней заголовков
      headerNum: 3
    
    # Межстрочные интервалы
    spacing:
      line: 1.5
      exceptions:
        - first_edition: single  # Исключение для первого абзаца
  
  # Структура документа
  structure:
    # Конфигурация титульного листа
    title_page:
      enabled: true
      template: "title_page_template"
      template_path: "doc_editor/templates/title_page_template.docx"
      image_path: "doc_editor/templates/logo.png"
      appendix: "А"
      elements:
        - agency_name: "Федеральное агентство"
        - standart_type: "МЕЖГОСУДАРСТВЕННАЯ"
        - designation: "ГОСТ Р 1.5—2004"
        - title: "Межгосударственная система по стандартизации"
        - status: "Проект стандарта"
        - city: "Москва"
        - publisher_info: "Стандартинформ"
        - current_year: "2024"
  
  # Нумерация и колонтитулы
  numbering:
    headers:
      enabled: true
      left: "Обозначение стандарта (без международных кодов)"
      right: "ГОСТ Р\n(проект, первая редакция)"
      page_numbers: true
    
    pages:
      start: 1
      style: "decimal"
```

### Поддерживаемые единицы измерения

- **Миллиметры:** `20mm`
- **Точки:** `12pt`
- **Дюймы:** `1in`
- **Сантиметры:** `2cm`

---

## Компоненты системы

Для подробного описания архитектуры см. [ARCHITECTURE.md](ARCHITECTURE.md).

### Основные модули

| Модуль | Назначение |
|--------|-----------|
| `doc_editor/editor.py` | Главный интерфейс редактора |
| `doc_editor/pipeline.py` | Оркестратор обработки |
| `doc_editor/models/config.py` | Типизированные конфигурации |
| `doc_editor/parsers/config_parser.py` | Парсинг YAML |
| `doc_editor/processors/style_processor.py` | Применение стилей и шрифтов |
| `doc_editor/processors/title_processor.py` | Добавление титульного листа |
| `doc_editor/processors/header_footer_processor.py` | Управление колонтитулами |
| `doc_editor/processors/margins_processor.py` | Установка полей страницы |

---

## Примеры использования

### Пример 1: Простое форматирование

```python
from doc_editor.editor import DocumentEditor

editor = DocumentEditor('document.docx')
editor.load_config('simple_config.yaml')
editor.apply_config()
editor.save('formatted.docx')
```

### Пример 2: Использование с шаблоном титульного листа

```yaml
# config.yaml
document:
  structure:
    title_page:
      enabled: true
      template_path: "templates/title.docx"
      image_path: "templates/logo.png"
      elements:
        - title: "Мой документ"
        - author: "Иван Петров"
        - year: "2024"
```

```python
editor = DocumentEditor('main_content.docx')
editor.load_config('config.yaml')
editor.apply_config()
editor.save('output.docx')
```

### Пример 3: REST API запрос (curl)

```bash
curl -X POST http://localhost:5000/api/process_document \
  -H "Content-Type: application/json" \
  -d '{
    "document_url": "https://example.com/doc.docx",
    "config": {
      "document": {
        "general": {
          "margins": {"left": "20mm", "right": "10mm", "top": "20mm", "bottom": "20mm"},
          "fonts": {
            "main": {"family": "Arial", "size": "12pt"}
          },
          "spacing": {"line": 1.5}
        },
        "structure": {"title_page": {"enabled": false}},
        "numbering": {"headers": {"enabled": false}}
      }
    }
  }' \
  --output result.docx
```

### Пример 4: Python запрос через requests

```python
import requests
import json

url = 'http://localhost:5000/api/process_document'
payload = {
    'document_url': 'https://example.com/document.docx',
    'config': {
        'document': {
            'general': {
                'margins': {'left': '20mm', 'right': '10mm', 'top': '20mm', 'bottom': '20mm'},
                'fonts': {'main': {'family': 'Arial', 'size': '12pt'}},
                'spacing': {'line': 1.5}
            },
            'structure': {'title_page': {'enabled': False}},
            'numbering': {'headers': {'enabled': False}}
        }
    }
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    with open('output.docx', 'wb') as f:
        f.write(response.content)
else:
    print(f"Ошибка: {response.status_code}")
    print(response.json())
```

---

## Docker развертывание

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "main.py"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  docx-processor:
    build: .
    ports:
      - "5000:5000"
    environment:
      FLASK_ENV: production
    volumes:
      - ./doc_editor/templates:/app/doc_editor/templates
      - ./uploads:/app/uploads
```

### Запуск

```bash
docker-compose up -d
```

---

## Обработка ошибок

Система использует кастомные исключения:

```python
from doc_editor.models.exceptions import (
    DocumentFormattingError,
    ConfigValidationError,
    ConfigParsingError
)

try:
    editor = DocumentEditor('doc.docx')
    editor.load_config('config.yaml')
    editor.apply_config()
    editor.save('output.docx')
except ConfigParsingError as e:
    print(f"Ошибка парсинга конфигурации: {e}")
except ConfigValidationError as e:
    print(f"Ошибка валидации конфигурации: {e}")
except DocumentFormattingError as e:
    print(f"Ошибка форматирования документа: {e}")
```

---

## Логирование

Все операции логируются. Примеры логов:

```
doc_editor.editor - INFO - Документ загружен: test.docx
doc_editor.parsers.config_parser - INFO - YAML конфигурация загружена: config.yaml
doc_editor.pipeline - INFO - Начало выполнения pipeline обработки документа
doc_editor.processors.style_processor - INFO - Начало применения стилей
doc_editor.processors.title_processor - INFO - Начало добавления титульного листа
doc_editor.processors.header_footer_processor - INFO - Начало применения колонтитулов
doc_editor.editor - INFO - Документ сохранен: output.docx
```

---

## Структура проекта

```
formatingDocx/
├── main.py                          # Flask приложение (REST API)
├── local.py                         # Локальный тестер
├── requirements.txt                 # Зависимости
├── Dockerfile                       # Docker образ
├── ARCHITECTURE.md                  # Подробное описание архитектуры
├── README.md                        # Этот файл
├── doc_editor/
│   ├── __init__.py                  # Экспорты
│   ├── editor.py                    # Главный интерфейс
│   ├── pipeline.py                  # Оркестратор
│   ├── utils.py                     # Утилиты
│   ├── models/
│   │   ├── __init__.py
│   │   ├── config.py                # Типизированные конфигурации
│   │   └── exceptions.py            # Исключения
│   ├── parsers/
│   │   ├── __init__.py
│   │   └── config_parser.py         # Парсинг YAML
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── style_processor.py       # Применение стилей
│   │   ├── title_processor.py       # Титульный лист
│   │   ├── header_footer_processor.py # Колонтитулы
│   │   └── margins_processor.py     # Поля страницы
│   ├── templates/
│   │   ├── title_page_template.docx # Шаблон титула
│   │   └── logo.png                 # Логотип
│   └── tests/
│       └── test_data/
│           ├── test.docx            # Тестовый документ
│           └── formatConfig.yaml    # Тестовая конфигурация
└── doc_editor/old/                  # Архивированные старые модули
    ├── style.py
    ├── title.py
    └── colontitul.py
```

---

## Требования к системе

- **Python:** 3.11+
- **ОС:** Linux, macOS, Windows
- **Зависимости:** см. `requirements.txt`
- **Оперативная память:** 512 MB минимум
- **Место на диске:** 100 MB для работы

---

## Интеграция с другими системами

### 1. Как библиотека

```python
# Импортировать в свой проект
from doc_editor import DocumentEditor, ConfigParser, DocumentFormattingError

def process_my_document(input_path, config_data):
    try:
        editor = DocumentEditor(input_path)
        editor.config = ConfigParser.from_dict(config_data)
        editor.apply_config()
        editor.save('result.docx')
        return True
    except DocumentFormattingError as e:
        log_error(str(e))
        return False
```

### 2. Как микросервис

```
Your App → HTTP POST → /api/process_document → DOCX File
```

### 3. В очереди задач (Celery)

```python
from celery import Celery
from doc_editor import DocumentEditor

celery = Celery(__name__)

@celery.task
def process_document_async(doc_path, config_path):
    editor = DocumentEditor(doc_path)
    editor.load_config(config_path)
    editor.apply_config()
    editor.save('output.docx')
```

---

## Решение проблем

### Проблема: "Шрифт не применяется"

**Причина:** Тема (theme) в DOCX переопределяет явно установленные шрифты.

**Решение:** Система автоматически устанавливает шрифты на трех уровнях:
1. На уровне стиля
2. На уровне run'а
3. На уровне XML (`w:rFonts`)

Убедитесь, что шрифт указан в конфигурации:
```yaml
fonts:
  main:
    family: Arial  # Должен быть установлен
```

### Проблема: "Колонтитулы не появляются"

**Причина:** Они могут быть отключены в конфигурации.

**Решение:**
```yaml
numbering:
  headers:
    enabled: true  # Убедитесь, что true
    left: "Текст"
    right: "Текст"
```

### Проблема: "Титульный лист не добавляется"

**Причина:** Путь к шаблону некорректен или файл не найден.

**Решение:**
```yaml
structure:
  title_page:
    enabled: true
    template_path: "doc_editor/templates/title_page_template.docx"  # Проверить путь
    image_path: "doc_editor/templates/logo.png"  # Проверить путь
```

---

## Лицензия

MIT

---

## Контакты и поддержка

Для вопросов и проблем см. [ARCHITECTURE.md](ARCHITECTURE.md) для деталей архитектуры или откройте issue в репозитории.

---

## История изменений

### v1.0.0 (текущая)
- ✅ Рефакторинг архитектуры: models/parsers/processors/pipeline
- ✅ Типизированная конфигурация (dataclasses)
- ✅ Поддержка титульных листов с шаблонизацией (docxtpl)
- ✅ Управление колонтитулами и номерами страниц
- ✅ Применение стилей и шрифтов на уровне XML
- ✅ Установка полей страницы
- ✅ REST API через Flask
- ✅ Логирование всех операций
- ✅ Обработка ошибок и исключения

### Планы на будущее
- Unit и интеграционные тесты
- CI/CD pipeline (GitHub Actions)
- Асинхронная обработка (Celery)
- Кэширование конфигураций
- Метрики и мониторинг
- Поддержка дополнительных форматов
