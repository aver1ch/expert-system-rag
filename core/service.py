from typing import List, Optional, Any, Dict
import json
import difflib
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter



class AnalyzeRequest(BaseModel):
    text: str


class ErrorItem(BaseModel):
    category: str  # exact_duplicate | partial_duplicate | punctuation | style
    message: str
    location: Optional[str] = None
    source: Optional[str] = None


class AnalyzeResponse(BaseModel):
    errors: List[ErrorItem]


app = FastAPI(title="Core text analysis service")


def _init_llm() -> Ollama:
    """
    Инициализация локальной модели Ollama.
    Используем ту же модель, что и в скриптах core/main.py / main2.py.
    """
    ollama_host = os.getenv("OLLAMA_HOST", "host.docker.internal:11434")
    return Ollama(model="deepseek-r1:7b", base_url=f"http://{ollama_host}")


def _init_embeddings() -> HuggingFaceEmbeddings:
    """
    Инициализация эмбеддингов E5-base-v2.

    В main2.py используется нормализация и device="mps" для Mac, повторяем это.
    """
    return HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _init_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Загрузка уже построенного FAISS-индекса из директории core/faiss_db.
    """
    try:
        db = FAISS.load_local(
            "faiss_db",
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:  # pragma: no cover - защита от отсутствующего индекса
        raise RuntimeError(f"Не удалось загрузить FAISS индекс: {exc}") from exc
    return db


llm = _init_llm()
embeddings = _init_embeddings()
vector_store = _init_vector_store(embeddings)
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)


def _detect_duplicates(text: str) -> List[ErrorItem]:
    """
    Поиск прямых и частичных дубликатов во входящем тексте
    на основе уже существующей базы (FAISS + E5).

    Подход:
    - режем входящий текст на чанки так же, как и документы;
    - для каждого чанка ищем ближайший фрагмент в индексе;
    - сравниваем текст чанка и найденного фрагмента по строковому сходству;
    - если очень высокое сходство — считаем «прямым» дубликатом,
      если просто высокое — «частичным».
    """
    errors: List[ErrorItem] = []

    if not text.strip():
        return errors

    chunks = splitter.split_text(text)

    for chunk in chunks:
        # Для E5-запросов используем префикс "query:"
        query = "query: " + chunk
        try:
            matches = vector_store.similarity_search(query, k=1)
        except Exception:
            # Если по каким-то причинам не удалось выполнить поиск, пропускаем чанк
            continue

        if not matches:
            continue

        match_doc = matches[0]
        source_text = match_doc.page_content
        # В индексировании мы добавляли префикс "passage:", убираем его
        if source_text.startswith("passage: "):
            source_text_clean = source_text[len("passage: ") :]
        else:
            source_text_clean = source_text

        ratio = difflib.SequenceMatcher(None, chunk, source_text_clean).ratio()

        if ratio >= 0.97:
            category = "exact_duplicate"
            message_prefix = "Прямое дублирование"
        elif ratio >= 0.85:
            category = "partial_duplicate"
            message_prefix = "Частичное дублирование"
        else:
            continue

        source_name = match_doc.metadata.get("source")

        errors.append(
            ErrorItem(
                category=category,
                message=f"{message_prefix} с фрагментом из документа: {source_name}",
                location=chunk[:300],
                source=source_name,
            )
        )

    return errors


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Модель deepseek-r1 часто оборачивает ответ в служебные теги.
    Пытаемся аккуратно вытащить JSON, взяв подстроку от первой '{' до последней '}'.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    json_candidate = text[start : end + 1]
    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        return None


def _detect_language_errors(text: str) -> List[ErrorItem]:
    """
    Поиск пунктуационных и речевых (стилистических) ошибок через LLM.

    Модель получает чёткую инструкцию вернуть JSON вида:
    {
      "errors": [
        {"category": "punctuation" | "style", "message": "...", "location": "..."}
      ]
    }
    """
    if not text.strip():
        return []

    prompt = f"""
Ты выступаешь как профессиональный русскоязычный лингвистический редактор.

Твоя задача:
1. Найти пунктуационные ошибки (запятые, тире, кавычки и т.п.).
2. Найти речевые/стилистические ошибки (неудачные формулировки, повторы, канцеляризмы и т.п.).
3. Не предлагай переписывать весь текст целиком — указывай только конкретные проблемные места.

ВАЖНО:
- Ответ должен быть ТОЛЬКО в виде одного JSON-объекта.
- Не добавляй никакого поясняющего текста до или после JSON.
- Формат строго:
{{
  "errors": [
    {{
      "category": "punctuation" или "style",
      "message": "краткое объяснение ошибки на русском",
      "location": "короткий фрагмент текста или описание места"
    }}
  ]
}}

Текст для анализа:
\"\"\"{text}\"\"\"
"""

    raw_response = llm.invoke(prompt)

    data = _extract_json(raw_response)
    if not data:
        return []

    items = data.get("errors") or []
    errors: List[ErrorItem] = []

    for item in items:
        try:
            category = str(item.get("category", "")).strip() or "style"
            if category not in {"punctuation", "style"}:
                category = "style"

            message = str(item.get("message", "")).strip()
            if not message:
                continue

            location = str(item.get("location", "")).strip() or None

            errors.append(
                ErrorItem(
                    category=category,
                    message=message,
                    location=location,
                )
            )
        except Exception:
            # Защита от неожиданных структур в ответе модели
            continue

    return errors


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Главная точка входа для backend:

    - находит дубликаты (прямые и частичные) по базе;
    - находит пунктуационные и речевые ошибки;
    - возвращает единый список ошибок.
    """
    text = request.text or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Поле 'text' не должно быть пустым")

    duplicate_errors = _detect_duplicates(text)
    language_errors = _detect_language_errors(text)

    all_errors = duplicate_errors + language_errors

    return AnalyzeResponse(errors=all_errors)


# Для локального запуска: `python -m uvicorn service:app --reload`
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

