import { useEffect, useState } from 'react'

const LABELS = {
  akiec: 'Actinic keratoses / intraepithelial carcinoma',
  bcc: 'Basal cell carcinoma',
  bkl: 'Benign keratosis-like lesions',
  df: 'Dermatofibroma',
  mel: 'Melanoma',
  nv: 'Melanocytic nevi',
  vasc: 'Vascular lesions',
}

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState('Checking model availability…')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    fetch('/api/health')
      .then((response) => response.json())
      .then((data) => setStatus(data.model_ready ? 'Model is ready.' : 'Model is not trained yet.'))
      .catch(() => setStatus('Backend is unavailable. Start the API server.'))
  }, [])

  function chooseFile(event) {
    const selected = event.target.files?.[0]
    setFile(selected || null)
    setResult(null)
    if (preview) URL.revokeObjectURL(preview)
    setPreview(selected ? URL.createObjectURL(selected) : '')
  }

  async function submit(event) {
    event.preventDefault()
    if (!file) return
    setBusy(true)
    setResult(null)
    const data = new FormData()
    data.append('file', file)
    try {
      const response = await fetch('/api/predict', { method: 'POST', body: data })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || 'Prediction request failed.')
      setResult(payload)
    } catch (requestProblem) {
      setResult({ problem: requestProblem.message })
    } finally {
      setBusy(false)
    }
  }

  return (
    <main>
      <header>
        <p className="eyebrow">HAM10000 · RESEARCH INTERFACE</p>
        <h1>Skin Lesion Image Classifier</h1>
        <p>Upload a dermoscopic image to run a trained seven-class CNN.</p>
      </header>
      <section className="warning">
        <strong>Not medical advice.</strong> This software is for research and education only. It is not a diagnostic device and must not guide treatment or replace a qualified clinician.
      </section>
      <section className="card">
        <h2>Image input</h2>
        <p className="muted">{status} Supported formats: JPEG, PNG, WebP. Maximum upload: 10 MB.</p>
        <form onSubmit={submit}>
          <label className="picker">
            <input type="file" accept="image/jpeg,image/png,image/webp" onChange={chooseFile} />
            <span>{file ? file.name : 'Choose an image'}</span>
          </label>
          {preview && <img className="preview" src={preview} alt="Selected lesion preview" />}
          <button type="submit" disabled={!file || busy}>{busy ? 'Analyzing…' : 'Run research classification'}</button>
        </form>
      </section>
      {result && (
        <section className="card result" aria-live="polite">
          {result.problem ? (
            <p className="problem">{result.problem}</p>
          ) : (
            <>
              <h2>Model output</h2>
              <p className="prediction">Top class: <strong>{LABELS[result.predicted_class]}</strong></p>
              <p className="muted">{result.disclaimer}</p>
              <ul>
                {result.scores.slice(0, 3).map((score) => (
                  <li key={score.code}><span>{LABELS[score.code]}</span><b>{(score.probability * 100).toFixed(1)}%</b></li>
                ))}
              </ul>
            </>
          )}
        </section>
      )}
    </main>
  )
}

export default App
