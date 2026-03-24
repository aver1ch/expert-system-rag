from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import difflib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

from docx import Document as DocxDocument
from fastapi import FastAPI, HTTPException, Request
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pypdf import PdfReader


class AnalyzeRequest(BaseModel):
    text: str
    title: Optional[str] = None


class ErrorItem(BaseModel):
    category: str
    message: str
    location: Optional[str] = None
    source: Optional[str] = None
    suggestion: Optional[str] = None
    replacement: Optional[str] = None


class AnalyzeSummary(BaseModel):
    exact_duplicate_percent: float = 0.0
    partial_duplicate_percent: float = 0.0


class AnalyzeResponse(BaseModel):
    errors: List[ErrorItem]
    summary: AnalyzeSummary


app = FastAPI(title="Core text analysis service")


splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
_llm: Optional[Ollama] = None
_vector_store: Optional[FAISS] = None
_vector_store_init_done = False
_rules_vector_store: Optional[FAISS] = None
_rules_vector_store_init_done = False
_llm_lock = threading.Lock()
_vector_lock = threading.Lock()
_rules_vector_lock = threading.Lock()


def _log(req_id: str, message: str) -> None:
    print(f"[ANALYZE][{req_id}] {message}", flush=True)


def _init_llm() -> Ollama:
    ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:70b")
    return Ollama(model=ollama_model, base_url=f"http://{ollama_host}")


def _get_llm() -> Ollama:
    global _llm
    if _llm is not None:
        return _llm
    with _llm_lock:
        if _llm is None:
            _llm = _init_llm()
    return _llm


def _init_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _extract_text_from_doc(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        return "\n".join(par.text for par in doc.paragraphs)
    return ""


def _build_vector_store_from_dir(
    source_dir: Path,
    embeddings: HuggingFaceEmbeddings,
    text_prefix: str,
) -> Optional[FAISS]:
    if not source_dir.exists() or not source_dir.is_dir():
        return None

    texts: List[str] = []
    metadatas: List[Dict[str, str]] = []

    for path in sorted(source_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".pdf", ".docx"}:
            continue
        try:
            raw = _extract_text_from_doc(path)
        except Exception:
            continue
        if not raw.strip():
            continue

        for chunk in splitter.split_text(raw):
            if not chunk.strip():
                continue
            texts.append(text_prefix + chunk)
            metadatas.append({"source": path.name})

    if not texts:
        return None

    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)


def _get_vector_store() -> Optional[FAISS]:
    global _vector_store, _vector_store_init_done
    if _vector_store_init_done:
        return _vector_store
    with _vector_lock:
        if _vector_store_init_done:
            return _vector_store
        try:
            embeddings = _init_embeddings()
            docs_dir = Path(os.getenv("RAG_DOCS_DIR", "/app/docs"))
            index_dir = Path(os.getenv("RAG_INDEX_DIR", "/app/faiss_db/docs"))
            force_rebuild = os.getenv("RAG_REBUILD", "0").lower() in {"1", "true", "yes", "on"}

            if index_dir.exists() and not force_rebuild:
                _vector_store = FAISS.load_local(
                    str(index_dir),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"[RAG] loaded docs index from {index_dir}", flush=True)
            else:
                _vector_store = _build_vector_store_from_dir(docs_dir, embeddings, "passage: ")
                if _vector_store is not None:
                    index_dir.mkdir(parents=True, exist_ok=True)
                    _vector_store.save_local(str(index_dir))
                    print(f"[RAG] built docs index and saved to {index_dir}", flush=True)
        except Exception as exc:
            print(f"[RAG] vector store init failed: {exc}", flush=True)
            _vector_store = None
        _vector_store_init_done = True
    return _vector_store


def _get_rules_vector_store() -> Optional[FAISS]:
    global _rules_vector_store, _rules_vector_store_init_done
    if _rules_vector_store_init_done:
        return _rules_vector_store
    with _rules_vector_lock:
        if _rules_vector_store_init_done:
            return _rules_vector_store
        try:
            embeddings = _init_embeddings()
            rules_dir = Path(os.getenv("RULES_DOCS_DIR", "/app/rules"))
            index_dir = Path(os.getenv("RULES_INDEX_DIR", "/app/faiss_db/rules"))
            force_rebuild = os.getenv("RULES_REBUILD", "0").lower() in {"1", "true", "yes", "on"}

            if index_dir.exists() and not force_rebuild:
                _rules_vector_store = FAISS.load_local(
                    str(index_dir),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"[RAG] loaded rules index from {index_dir}", flush=True)
            else:
                _rules_vector_store = _build_vector_store_from_dir(rules_dir, embeddings, "rule: ")
                if _rules_vector_store is not None:
                    index_dir.mkdir(parents=True, exist_ok=True)
                    _rules_vector_store.save_local(str(index_dir))
                    print(f"[RAG] built rules index and saved to {index_dir}", flush=True)
        except Exception as exc:
            print(f"[RAG] rules vector store init failed: {exc}", flush=True)
            _rules_vector_store = None
        _rules_vector_store_init_done = True
    return _rules_vector_store


def _build_rules_context(text: str, req_id: str) -> str:
    rules_store = _get_rules_vector_store()
    if rules_store is None:
        _log(req_id, "rules context skipped: rules vector store is unavailable")
        return ""

    top_k_raw = os.getenv("RULES_TOP_K", "4").strip()
    try:
        top_k = max(1, int(top_k_raw))
    except ValueError:
        top_k = 4

    query = "query: " + text[:4000]
    try:
        matches = rules_store.similarity_search(query, k=top_k)
    except Exception as exc:
        _log(req_id, f"rules context retrieval failed err={exc}")
        return ""

    if not matches:
        _log(req_id, "rules context retrieval returned 0 matches")
        return ""

    blocks: List[str] = []
    for i, match in enumerate(matches, start=1):
        source = match.metadata.get("source", "unknown")
        body = match.page_content
        if body.startswith("rule: "):
            body = body[len("rule: ") :]
        blocks.append(f"[Rule {i}] source={source}\n{body[:900]}")

    _log(req_id, f"rules context ready chunks={len(blocks)}")
    return "\n\n".join(blocks)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _detect_duplicates(text: str, req_id: str) -> Tuple[List[ErrorItem], AnalyzeSummary]:
    started = time.perf_counter()
    errors: List[ErrorItem] = []
    summary = AnalyzeSummary(exact_duplicate_percent=0.0, partial_duplicate_percent=0.0)

    vector_store = _get_vector_store()
    if not text.strip() or vector_store is None:
        _log(req_id, "duplicates skipped: empty text or vector store is unavailable")
        return errors, summary

    chunks = splitter.split_text(text)
    if not chunks:
        _log(req_id, "duplicates skipped: no chunks")
        return errors, summary

    total_chars = max(1, len(text))
    exact_chars = 0
    partial_chars = 0

    verbatim_thr_raw = os.getenv("DUP_VERBATIM_SIM", "0.97").strip()
    semantic_thr_raw = os.getenv("DUP_SEMANTIC_SIM", "0.85").strip()
    try:
        verbatim_thr = float(verbatim_thr_raw)
    except ValueError:
        verbatim_thr = 0.97
    try:
        semantic_thr = float(semantic_thr_raw)
    except ValueError:
        semantic_thr = 0.85

    for chunk in chunks:
        query = "query: " + chunk
        try:
            matches = vector_store.similarity_search_with_score(query, k=1)
        except Exception:
            continue

        if not matches:
            continue

        match_doc, score = matches[0]
        source_text = match_doc.page_content
        source_text_clean = source_text[len("passage: ") :] if source_text.startswith("passage: ") else source_text

        # score is squared L2 distance for normalized embeddings -> cosine ~= 1 - score/2
        try:
            sim = 1.0 - (float(score) / 2.0)
        except Exception:
            sim = 0.0
        sim = max(0.0, min(1.0, sim))

        ratio = difflib.SequenceMatcher(None, chunk, source_text_clean).ratio()

        if sim >= verbatim_thr and ratio >= 0.97:
            category = "exact_duplicate"
            message_prefix = "Дословное дублирование"
            exact_chars += len(chunk)
        elif sim >= semantic_thr:
            category = "partial_duplicate"
            message_prefix = "Смысловое дублирование"
            partial_chars += len(chunk)
        else:
            continue

        source_name = match_doc.metadata.get("source")
        errors.append(
            ErrorItem(
                category=category,
                message=f"{message_prefix} с материалом из базы: {source_name} (similarity={sim:.2f})",
                location=chunk[:300],
                source=source_name,
                suggestion="Переформулируйте этот фрагмент, добавьте уникальные факты и измените структуру предложения.",
                replacement=None,
            )
        )

    summary.exact_duplicate_percent = round(min(100.0, (exact_chars / total_chars) * 100.0), 2)
    summary.partial_duplicate_percent = round(min(100.0, (partial_chars / total_chars) * 100.0), 2)
    _log(req_id, f"duplicates done ms={((time.perf_counter() - started) * 1000):.0f} count={len(errors)}")
    return errors, summary


def _detect_language_errors(text: str, req_id: str, title: str = "") -> List[ErrorItem]:
    if not text.strip():
        return []

    ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:70b")

    probe_started = time.perf_counter()
    try:
        with urllib.request.urlopen(f"http://{ollama_host}/api/version", timeout=8) as resp:
            probe_status = resp.status
        _log(req_id, f"ollama_probe ok host={ollama_host} status={probe_status} ms={((time.perf_counter() - probe_started) * 1000):.0f}")
    except Exception as exc:
        _log(req_id, f"ollama_probe failed host={ollama_host} ms={((time.perf_counter() - probe_started) * 1000):.0f} err={exc}")

    rules_context = _build_rules_context(text, req_id)
    rules_section = rules_context if rules_context else "Нет дополнительных правил из базы."

    effective_title = title.strip() or "Не указано"

    prompt = f"""
Ты выступаешь как профессиональный русскоязычный редактор.

Дополнительные правила и нормы:
{rules_section}

Нужно выполнить задачи:
1) Проверить соответствие оформления и изложения требованиям ГОСТ Р 1.5 и ГОСТ 1.5.
2) Найти орфографические, пунктуационные, грамматические, стилистические и речевые ошибки.
3) Оценить соответствие содержания основного текста наименованию проекта стандарта.
4) Поиск дублирования выполняется отдельно другим модулем, НЕ добавляй в ответ отдельные ошибки дублирования.

Наименование проекта стандарта:
{effective_title}

Найди и верни ошибки строго по категориям:
- punctuation (пунктуация)
- style (стилистика/речь)
- grammar (грамматика)
- spelling (орфография)
- gost_compliance (несоответствие требованиям ГОСТ Р 1.5 / ГОСТ 1.5)
- title_scope (несоответствие содержания наименованию проекта стандарта)

Верни только JSON строго в формате:
{{
  "errors": [
    {{
      "category": "punctuation" | "style" | "grammar" | "spelling" | "gost_compliance" | "title_scope",
      "message": "краткое объяснение проблемы",
      "location": "короткий проблемный фрагмент",
      "suggestion": "что лучше сделать",
      "replacement": "минимальный исправленный вариант фрагмента; пустая строка, если не применимо"
    }}
  ]
}}

Текст для анализа:
<<<TEXT>>>
{text}
<<<END_TEXT>>>
"""

    _log(req_id, f"llm_invoke start model={ollama_model} prompt_chars={len(prompt)}")
    llm_started = time.perf_counter()
    try:
        raw_response = _get_llm().invoke(prompt)
    except Exception as exc:
        _log(req_id, f"llm_invoke failed after={((time.perf_counter() - llm_started) * 1000):.0f}ms err={exc}")
        return [
            ErrorItem(
                category="style",
                message=f"LLM недоступна через Ollama: {exc}",
                suggestion="Проверьте SSH-туннель и доступность удаленной модели.",
            )
        ]

    _log(
        req_id,
        f"llm_invoke done in={((time.perf_counter() - llm_started) * 1000):.0f}ms response_chars={len(raw_response or '')}",
    )

    parse_started = time.perf_counter()
    data = _extract_json(raw_response)
    if not data:
        _log(req_id, "llm_response has no valid json payload")
        return []
    _log(req_id, f"llm_json_parse ok ms={((time.perf_counter() - parse_started) * 1000):.0f}")

    items = data.get("errors") or []
    errors: List[ErrorItem] = []
    for item in items:
        try:
            category = str(item.get("category", "")).strip().lower() or "style"
            if category not in {"punctuation", "style", "grammar", "spelling", "gost_compliance", "title_scope"}:
                category = "style"

            message = str(item.get("message", "")).strip()
            if not message:
                continue

            errors.append(
                ErrorItem(
                    category=category,
                    message=message,
                    location=str(item.get("location", "")).strip() or None,
                    suggestion=str(item.get("suggestion", "")).strip() or None,
                    replacement=str(item.get("replacement", "")).strip() or None,
                )
            )
        except Exception:
            continue

    _log(req_id, f"llm_errors_extracted count={len(errors)}")
    return errors


@app.on_event("startup")
def warmup_llm() -> None:
    if os.getenv("LLM_WARMUP_ENABLED", "1").lower() not in {"1", "true", "yes", "on"}:
        return

    req_id = "startup-warmup"
    warmup_prompt = os.getenv("LLM_WARMUP_PROMPT", "hello, deep seek")
    _log(req_id, "warmup started")
    started = time.perf_counter()
    try:
        response = _get_llm().invoke(warmup_prompt)
    except Exception as exc:
        _log(req_id, f"warmup failed after={((time.perf_counter() - started) * 1000):.0f}ms err={exc}")
        raise RuntimeError(f"LLM warmup failed: {exc}") from exc

    preview = (response or "").strip().replace("\n", " ")[:120]
    _log(
        req_id,
        f"warmup done in={((time.perf_counter() - started) * 1000):.0f}ms response_chars={len(response or '')} preview={preview}",
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, http_request: Request) -> AnalyzeResponse:
    req_id = (http_request.headers.get("X-Request-ID", "") or "").strip() or f"req-{int(time.time() * 1000)}"
    text = request.text or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Поле 'text' не должно быть пустым")

    _log(req_id, f"start text_len={len(text.strip())}")
    total_started = time.perf_counter()

    duplicate_errors, summary = _detect_duplicates(text, req_id=req_id)
    language_errors = _detect_language_errors(text, req_id=req_id, title=request.title or "")

    all_errors = duplicate_errors + language_errors
    _log(req_id, f"done total_ms={((time.perf_counter() - total_started) * 1000):.0f} errors={len(all_errors)}")
    return AnalyzeResponse(errors=all_errors, summary=summary)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
