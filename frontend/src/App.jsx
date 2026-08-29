import { useEffect, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1'

function App() {
  const [imageFile, setImageFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  useEffect(() => {
    if (!imageFile) {
      setPreviewUrl(null)
      return
    }

    const url = URL.createObjectURL(imageFile)
    setPreviewUrl(url)

    return () => URL.revokeObjectURL(url)
  }, [imageFile])

  const handleChange = (event) => {
    const file = event.target.files?.[0]

    setError(null)
    setResult(null)

    if (!file) {
      setImageFile(null)
      return
    }

    if (!file.type.startsWith('image/')) {
      setImageFile(null)
      setError('Please select a valid image file.')
      return
    }

    setImageFile(file)
  }

  const analyzeImage = async () => {
    if (!imageFile) {
      setError('Please select an image first.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', imageFile)

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData,
      })

      let data = null

      try {
        data = await response.json()
      } catch {
        data = null
      }

      if (!response.ok) {
        const message =
          data?.detail ||
          data?.message ||
          `Analysis failed (HTTP ${response.status})`

        throw new Error(message)
      }

      setResult(data)
    } catch (err) {
      setError(err.message || 'Unable to connect to the VisionIQ API.')
    } finally {
      setLoading(false)
    }
  }

  const analyzeAnother = () => {
    setImageFile(null)
    setPreviewUrl(null)
    setError(null)
    setResult(null)
  }

  return (
    <main className="app">
      <header className="header">
        <div>
          <h1>VisionIQ</h1>
          <p>AI-powered image quality analysis</p>
        </div>
      </header>

      <section className="card">
        <h2>Analyze an image</h2>

        <p className="description">
          Upload an image and VisionIQ will evaluate its quality and identify
          potential degradation.
        </p>

        <label className="upload-area">
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp,image/bmp"
            onChange={handleChange}
            disabled={loading}
          />

          {previewUrl ? (
            <img
              src={previewUrl}
              alt="Selected preview"
              className="preview"
            />
          ) : (
            <div className="upload-placeholder">
              <strong>Choose an image</strong>
              <span>JPG, PNG, WEBP or BMP</span>
            </div>
          )}
        </label>

        {imageFile && (
          <p className="filename">
            Selected: <strong>{imageFile.name}</strong>
          </p>
        )}

        <button
          className="analyze-button"
          onClick={analyzeImage}
          disabled={!imageFile || loading}
        >
          {loading ? 'Analyzing...' : 'Analyze Image'}
        </button>

        {error && (
          <div className="error" role="alert">
            <strong>Error:</strong> {error}
          </div>
        )}
      </section>

      {result && (
        <section className="card result-card">
          <div className="result-header">
            <h2>Analysis Result</h2>
            <div className="score">
              {formatNumber(result.quality_score)} / 100
            </div>
          </div>

          <div className="result-grid">
            <div className="result-item">
              <span className="label">Quality</span>
              <strong>{result.quality_label || 'Unknown'}</strong>
            </div>

            <div className="result-item">
              <span className="label">Clean vs Degraded</span>
              <strong>
                {percentage(result.clean_vs_degraded_prob)}
              </strong>
            </div>

            <div className="result-item">
              <span className="label">Stage 1 Degraded Probability</span>
              <strong>
                {percentage(result.stage1_degraded_prob)}
              </strong>
            </div>

            {result.stage2_prediction && (
              <div className="result-item">
                <span className="label">Detected Issue</span>
                <strong>{result.stage2_prediction}</strong>
              </div>
            )}

            {result.stage2_confidence !== undefined &&
              result.stage2_confidence !== null && (
                <div className="result-item">
                  <span className="label">Issue Confidence</span>
                  <strong>
                    {percentage(result.stage2_confidence)}
                  </strong>
                </div>
              )}
          </div>

          {Array.isArray(result.issues) && result.issues.length > 0 && (
            <div className="issues">
              <h3>Issues</h3>

              <div className="issue-list">
                {result.issues.map((issue, index) => (
                  <div className="issue" key={`${issue.type}-${index}`}>
                    <div>
                      <strong>{issue.type}</strong>
                      {issue.severity && (
                        <span className="severity">
                          {issue.severity}
                        </span>
                      )}
                    </div>

                    {issue.confidence !== undefined &&
                      issue.confidence !== null && (
                        <span>
                          Confidence: {percentage(issue.confidence)}
                        </span>
                      )}
                  </div>
                ))}
              </div>
            </div>
          )}

          <button className="secondary-button" onClick={analyzeAnother}>
            Analyze another image
          </button>
        </section>
      )}
    </main>
  )
}

function formatNumber(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return '—'
  }

  return Number(value).toFixed(1)
}

function percentage(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return '—'
  }

  return `${(Number(value) * 100).toFixed(1)}%`
}

export default App
