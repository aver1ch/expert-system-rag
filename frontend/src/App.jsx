import { useState } from 'react'
import './App.css'

function App() {
  const [text, setText] = useState('')
  const [errors, setErrors] = useState([])
  const [loading, setLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')

  const handleAnalyze = async () => {
    setErrorMessage('')
    setErrors([])

    const trimmed = text.trim()
    if (!trimmed) {
      setErrorMessage('Введите текст для анализа.')
      return
    }

    setLoading(true)
    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: trimmed }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || `Ошибка запроса: ${response.status}`)
      }

      const data = await response.json()
      setErrors(data.errors ?? [])
    } catch (err) {
      console.error(err)
      setErrorMessage('Не удалось выполнить анализ. Проверьте, что backend и core-сервис запущены.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-root">
      <h1>Анализ документа</h1>
      <p className="subtitle">
        Вставьте текст документа, чтобы найти дубликаты по базе, пунктуационные и речевые ошибки.
      </p>

      <textarea
        className="input-textarea"
        placeholder="Вставьте сюда текст документа..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={12}
      />

      <button className="analyze-button" onClick={handleAnalyze} disabled={loading}>
        {loading ? 'Анализируем...' : 'Проверить документ'}
      </button>

      {errorMessage && <div className="error-message">{errorMessage}</div>}

      <div className="results-block">
        <h2>Найденные ошибки</h2>
        {!loading && errors.length === 0 && !errorMessage && (
          <p className="muted">Ошибки будут показаны здесь.</p>
        )}

        {errors.length > 0 && (
          <ul className="errors-list">
            {errors.map((err, idx) => (
              <li key={idx} className={`error-item error-${err.category}`}>
                <div className="error-header">
                  <span className="error-category">
                    {err.category === 'exact_duplicate' && 'Прямое дублирование'}
                    {err.category === 'partial_duplicate' && 'Частичное дублирование'}
                    {err.category === 'punctuation' && 'Пунктуационная ошибка'}
                    {err.category === 'style' && 'Речевая/стилистическая ошибка'}
                    {!['exact_duplicate', 'partial_duplicate', 'punctuation', 'style'].includes(
                      err.category,
                    ) && err.category}
                  </span>
                  {err.source && <span className="error-source">Источник: {err.source}</span>}
                </div>
                <div className="error-message-text">{err.message}</div>
                {err.location && (
                  <div className="error-location">
                    <span className="label">Фрагмент:</span> {err.location}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

export default App
