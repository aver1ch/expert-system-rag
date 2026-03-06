import { useEffect, useMemo, useRef, useState } from 'react'
import { Document, Packer, Paragraph } from 'docx'
import { jsPDF } from 'jspdf'
import mammoth from 'mammoth'
import * as pdfjs from 'pdfjs-dist'
import pdfWorkerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import './App.css'

pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerSrc

const CATEGORY_META = {
  exact_duplicate: { label: 'Полное дублирование', cls: 'exact_duplicate' },
  partial_duplicate: { label: 'Частичное дублирование', cls: 'partial_duplicate' },
  punctuation: { label: 'Пунктуация', cls: 'punctuation' },
  style: { label: 'Стиль', cls: 'style' },
  grammar: { label: 'Грамматика', cls: 'grammar' },
  spelling: { label: 'Орфография', cls: 'spelling' },
}

const ANALYZE_PROGRESS_STEPS = [
  { key: 'prepare', label: 'Подготовка текста', upTo: 20 },
  { key: 'request', label: 'Запрос в backend/core', upTo: 45 },
  { key: 'model', label: 'Ожидание ответа модели', upTo: 85 },
  { key: 'post', label: 'Обработка результата', upTo: 99 },
  { key: 'done', label: 'Готово', upTo: 100 },
]

function buildProgressState(percent) {
  const normalized = Math.max(0, Math.min(100, Math.round(percent)))
  const current = ANALYZE_PROGRESS_STEPS.find((step) => normalized <= step.upTo) || ANALYZE_PROGRESS_STEPS.at(-1)
  const stepIndex = Math.max(
    0,
    ANALYZE_PROGRESS_STEPS.findIndex((step) => step.key === current.key),
  )
  return {
    percent: normalized,
    label: current.label,
    stepKey: current.key,
    stepIndex,
  }
}

const escapeHtml = (value) =>
  value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')

const withLineBreaks = (value) => escapeHtml(value).replaceAll('\n', '<br/>')

function mapErrorsToText(text, errors) {
  const busy = []
  return errors.map((err) => {
    const fragment = (err.location || '').trim()
    if (!fragment || fragment.length < 2) {
      return { ...err, start: -1, end: -1 }
    }

    let start = text.indexOf(fragment)
    while (start !== -1) {
      const end = start + fragment.length
      const overlap = busy.some((item) => start < item.end && item.start < end)
      if (!overlap) {
        busy.push({ start, end })
        return { ...err, start, end }
      }
      start = text.indexOf(fragment, start + 1)
    }

    return { ...err, start: -1, end: -1 }
  })
}

function buildHighlightedHtml(text, errors, activeErrorId) {
  const ranges = errors
    .filter((err) => err.start >= 0 && err.end > err.start)
    .sort((a, b) => a.start - b.start)

  if (ranges.length === 0) {
    return withLineBreaks(text)
  }

  let html = ''
  let cursor = 0

  for (const err of ranges) {
    if (err.start > cursor) {
      html += withLineBreaks(text.slice(cursor, err.start))
    }

    const cls = CATEGORY_META[err.category]?.cls || 'style'
    const activeCls = err.id === activeErrorId ? ' active' : ''
    html += `<mark class="hl hl-${cls}${activeCls}">${withLineBreaks(text.slice(err.start, err.end))}</mark>`
    cursor = err.end
  }

  if (cursor < text.length) {
    html += withLineBreaks(text.slice(cursor))
  }

  return html
}

async function extractTextFromPdf(file) {
  const data = new Uint8Array(await file.arrayBuffer())
  const pdf = await pdfjs.getDocument({ data }).promise
  let output = ''
  for (let i = 1; i <= pdf.numPages; i += 1) {
    const page = await pdf.getPage(i)
    const content = await page.getTextContent()
    output += content.items.map((item) => item.str || '').join(' ') + '\n\n'
  }
  return output.trim()
}

async function extractTextFromDocx(file) {
  const buffer = await file.arrayBuffer()
  const result = await mammoth.extractRawText({ arrayBuffer: buffer })
  return (result.value || '').trim()
}

function App() {
  const editorRef = useRef(null)
  const progressTickerRef = useRef(null)

  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [documents, setDocuments] = useState([])
  const [selectedDocId, setSelectedDocId] = useState(null)

  const [text, setText] = useState('')
  const [errors, setErrors] = useState([])
  const [activeErrorId, setActiveErrorId] = useState(null)
  const [summary, setSummary] = useState({ exact_duplicate_percent: 0, partial_duplicate_percent: 0 })

  const [loadingDocs, setLoadingDocs] = useState(false)
  const [loadingAnalyze, setLoadingAnalyze] = useState(false)
  const [busyUpload, setBusyUpload] = useState(false)
  const [message, setMessage] = useState('')
  const [progress, setProgress] = useState(buildProgressState(0))

  const mappedErrors = useMemo(() => mapErrorsToText(text, errors), [text, errors])
  const activeError = mappedErrors.find((err) => err.id === activeErrorId) || null

  useEffect(() => {
    if (!editorRef.current) {
      return
    }
    editorRef.current.innerHTML = buildHighlightedHtml(text, mappedErrors, activeErrorId)
  }, [text, mappedErrors, activeErrorId])

  useEffect(() => {
    return () => {
      if (progressTickerRef.current) {
        clearInterval(progressTickerRef.current)
      }
    }
  }, [])

  const clearAnalysis = () => {
    setErrors([])
    setActiveErrorId(null)
    setSummary({ exact_duplicate_percent: 0, partial_duplicate_percent: 0 })
  }

  const fetchDocuments = async () => {
    setLoadingDocs(true)
    try {
      const res = await fetch('/documents')
      if (!res.ok) {
        throw new Error('Не удалось получить документы')
      }
      const data = await res.json()
      setDocuments(data || [])
    } catch (err) {
      console.error(err)
      setMessage('Ошибка загрузки списка документов')
    } finally {
      setLoadingDocs(false)
    }
  }

  useEffect(() => {
    fetchDocuments()
  }, [])

  const openDocument = async (id) => {
    try {
      const res = await fetch(`/documents/${id}`)
      if (!res.ok) {
        throw new Error('Не удалось открыть документ')
      }
      const doc = await res.json()
      setSelectedDocId(id)
      setText(doc.current_text || '')
      clearAnalysis()
      setMessage('')
    } catch (err) {
      console.error(err)
      setMessage('Ошибка открытия документа')
    }
  }

  const uploadDocument = async (event) => {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }

    setBusyUpload(true)
    setMessage('')

    try {
      let extractedText = ''
      const lower = file.name.toLowerCase()
      if (lower.endsWith('.pdf')) {
        extractedText = await extractTextFromPdf(file)
      } else if (lower.endsWith('.docx')) {
        extractedText = await extractTextFromDocx(file)
      } else {
        throw new Error('Поддерживаются только PDF и DOCX')
      }

      const formData = new FormData()
      formData.append('file', file)
      formData.append('name', file.name)
      formData.append('text', extractedText)

      const res = await fetch('/documents/upload', {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        throw new Error('Ошибка загрузки документа')
      }

      const newDoc = await res.json()
      await fetchDocuments()
      await openDocument(newDoc.id)
    } catch (err) {
      console.error(err)
      setMessage('Не удалось загрузить документ')
    } finally {
      setBusyUpload(false)
      event.target.value = ''
    }
  }

  const saveDocument = async () => {
    if (!selectedDocId) {
      setMessage('Сначала выберите документ из списка')
      return
    }

    try {
      const res = await fetch(`/documents/${selectedDocId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!res.ok) {
        throw new Error('Ошибка сохранения')
      }
      clearAnalysis()
      setMessage('Документ сохранён, результаты анализа сброшены')
      await fetchDocuments()
    } catch (err) {
      console.error(err)
      setMessage('Не удалось сохранить документ')
    }
  }

  const deleteDocument = async (docId) => {
    try {
      const res = await fetch(`/documents/${docId}`, { method: 'DELETE' })
      if (!res.ok) {
        throw new Error('Ошибка удаления')
      }

      if (docId === selectedDocId) {
        setSelectedDocId(null)
        setText('')
        clearAnalysis()
      }

      await fetchDocuments()
    } catch (err) {
      console.error(err)
      setMessage('Не удалось удалить документ')
    }
  }

  const downloadStored = async (docId, docName) => {
    try {
      const res = await fetch(`/documents/${docId}/download`)
      if (!res.ok) {
        throw new Error('Ошибка скачивания')
      }
      const blob = await res.blob()
      const link = document.createElement('a')
      link.href = URL.createObjectURL(blob)
      link.download = docName
      link.click()
      URL.revokeObjectURL(link.href)
    } catch (err) {
      console.error(err)
      setMessage('Не удалось скачать файл')
    }
  }

  const runAnalyze = async () => {
    const trimmed = text.trim()
    if (!trimmed) {
      setMessage('Текст документа пустой')
      return
    }

    if (progressTickerRef.current) {
      clearInterval(progressTickerRef.current)
    }

    setLoadingAnalyze(true)
    setMessage('')
    setProgress(buildProgressState(6))

    progressTickerRef.current = setInterval(() => {
      setProgress((prev) => {
        if (prev.percent >= 92) {
          return prev
        }
        const next = prev.percent < 60 ? prev.percent + 5 : prev.percent + 2
        return buildProgressState(next)
      })
    }, 700)

    try {
      const nextReqId = `req-${(globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`).replaceAll('-', '')}`
      setProgress(buildProgressState(22))

      const res = await fetch('/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Request-ID': nextReqId,
        },
        body: JSON.stringify({ text: trimmed }),
      })

      if (!res.ok) {
        const errText = await res.text()
        throw new Error(errText)
      }

      setProgress(buildProgressState(90))
      const data = await res.json()
      const nextErrors = (data.errors || []).map((item, idx) => ({ ...item, id: `err-${idx}` }))
      setErrors(nextErrors)
      setSummary(data.summary || { exact_duplicate_percent: 0, partial_duplicate_percent: 0 })
      setActiveErrorId(nextErrors[0]?.id || null)
      setProgress(buildProgressState(100))
    } catch (err) {
      console.error(err)
      setMessage('Не удалось выполнить анализ. Проверьте backend/core и подключение к HPC.')
      setProgress(buildProgressState(0))
    } finally {
      if (progressTickerRef.current) {
        clearInterval(progressTickerRef.current)
      }
      progressTickerRef.current = null
      setLoadingAnalyze(false)
    }
  }

  const applySuggestion = (item) => {
    if (!item || !item.location || !item.replacement) {
      return
    }

    const start = item.start >= 0 ? item.start : text.indexOf(item.location)
    if (start < 0) {
      return
    }

    const end = start + item.location.length
    setText(text.slice(0, start) + item.replacement + text.slice(end))
    setErrors((prev) => prev.filter((err) => err.id !== item.id))
    setActiveErrorId(null)
  }

  const rejectSuggestion = (id) => {
    setErrors((prev) => prev.filter((item) => item.id !== id))
    if (id === activeErrorId) {
      setActiveErrorId(null)
    }
  }

  const handleEditorInput = (event) => {
    setText(event.currentTarget.innerText || '')
  }

  const exportDocx = async () => {
    const paragraphs = text.split('\n').map((line) => new Paragraph(line || ' '))
    const doc = new Document({ sections: [{ children: paragraphs }] })
    const blob = await Packer.toBlob(doc)
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = 'edited-document.docx'
    link.click()
    URL.revokeObjectURL(link.href)
  }

  const exportPdf = () => {
    const doc = new jsPDF({ unit: 'pt', format: 'a4' })
    doc.setFontSize(11)
    const maxWidth = doc.internal.pageSize.getWidth() - 80
    const lines = doc.splitTextToSize(text || ' ', maxWidth)

    let y = 50
    for (const line of lines) {
      if (y > doc.internal.pageSize.getHeight() - 40) {
        doc.addPage()
        y = 50
      }
      doc.text(line, 40, y)
      y += 16
    }

    doc.save('edited-document.pdf')
  }

  return (
    <div className="app-root">
      <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-head">
          <h2>Документы</h2>
          <button type="button" onClick={() => setSidebarOpen(false)}>
            Скрыть
          </button>
        </div>

        <label className="upload-button">
          {busyUpload ? 'Загрузка...' : 'Загрузить документ'}
          <input type="file" accept=".pdf,.docx" hidden disabled={busyUpload} onChange={uploadDocument} />
        </label>

        <div className="documents-list">
          {loadingDocs && <p className="muted">Загружаем...</p>}
          {!loadingDocs && documents.length === 0 && <p className="muted">Нет документов</p>}
          {documents.map((doc) => (
            <div key={doc.id} className={`doc-item ${doc.id === selectedDocId ? 'active' : ''}`}>
              <button type="button" className="doc-open" onClick={() => openDocument(doc.id)}>
                {doc.name}
              </button>
              <div className="doc-actions">
                <button type="button" onClick={() => downloadStored(doc.id, doc.name)}>
                  Скачать
                </button>
                <button type="button" onClick={() => deleteDocument(doc.id)}>
                  Удалить
                </button>
              </div>
            </div>
          ))}
        </div>
      </aside>

      <main className="main-wrap">
        <header className="topbar">
          {!sidebarOpen && (
            <button type="button" onClick={() => setSidebarOpen(true)}>
              Документы
            </button>
          )}
          <h1>RAG-анализ документов</h1>
          <div className="top-actions">
            <button type="button" onClick={saveDocument}>
              Сохранить
            </button>
            <button type="button" onClick={exportDocx} disabled={!text.trim()}>
              Скачать DOCX
            </button>
            <button type="button" onClick={exportPdf} disabled={!text.trim()}>
              Скачать PDF
            </button>
          </div>
        </header>

        <section className="dup-stats">
          <div className="stat-card">Полное дублирование: {summary.exact_duplicate_percent.toFixed(2)}%</div>
          <div className="stat-card">Частичное дублирование: {summary.partial_duplicate_percent.toFixed(2)}%</div>
        </section>

        {message && <div className="message">{message}</div>}

        <div className="analyze-row">
          <button type="button" className="analyze-button" onClick={runAnalyze} disabled={loadingAnalyze}>
            {loadingAnalyze ? 'Анализируем...' : 'Запустить анализ'}
          </button>
        </div>

        <div className="progress-wrap" aria-live="polite">
          <div className="progress-top">
            <span>{progress.label || 'Ожидание запуска'}</span>
            <span>{progress.percent}%</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress.percent}%` }} />
          </div>
          <div className="progress-steps">
            {ANALYZE_PROGRESS_STEPS.slice(0, 4).map((step, idx) => (
              <span
                key={step.key}
                className={`progress-step${idx < progress.stepIndex ? ' done' : ''}${idx === progress.stepIndex ? ' active' : ''}`}
              >
                {step.label}
              </span>
            ))}
          </div>
        </div>

        <section className="windows-grid">
          <div className="feedback-window">
            <h3>Разбор ошибок и предложения</h3>
            {mappedErrors.length === 0 && <p className="muted">Запустите анализ, чтобы увидеть замечания.</p>}

            {mappedErrors.map((item) => {
              const meta = CATEGORY_META[item.category] || { label: item.category, cls: 'style' }
              return (
                <div
                  key={item.id}
                  className={`feedback-item feedback-${meta.cls}${item.id === activeErrorId ? ' active' : ''}`}
                  onClick={() => setActiveErrorId(item.id)}
                >
                  <div className="feedback-title">{meta.label}</div>
                  <div className="feedback-message">{item.message}</div>
                  {item.location && <div className="feedback-fragment">Фрагмент: {item.location}</div>}
                  {item.suggestion && <div className="feedback-suggestion">Совет: {item.suggestion}</div>}
                  {item.replacement && <div className="feedback-replacement">Замена: {item.replacement}</div>}
                  <div className="feedback-actions">
                    <button type="button" onClick={() => applySuggestion(item)} disabled={!item.replacement}>
                      Согласиться
                    </button>
                    <button type="button" onClick={() => rejectSuggestion(item.id)}>
                      Отклонить
                    </button>
                  </div>
                </div>
              )
            })}
          </div>

          <div className="editor-window">
            <h3>Текстовый редактор</h3>
            <div className="editor" ref={editorRef} contentEditable suppressContentEditableWarning onInput={handleEditorInput} />
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
