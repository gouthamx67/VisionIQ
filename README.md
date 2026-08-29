# VisionIQ

Image quality analysis application that evaluates uploaded images for degradation issues
including blur, motion blur, noise, compression artifacts, and exposure problems.

## What VisionIQ Does

VisionIQ analyzes images and classifies them into quality categories using a hierarchical
machine learning pipeline:

- **Stage 1**: Binary classifier (clean vs degraded) using Random Forest
- **Stage 2**: Multi-class degradation type classifier (blur, compression, noise,
  overexposure, underexposure) using Random Forest
- **Decision layer**: Independent computer-vision rules provide conservative overrides
  to catch obvious degradations that the ML classifier might miss

## Classification Pipeline (High Level)

1. Image is loaded and features are extracted (brightness, laplacian variance,
   entropy, texture, compression artifacts, etc.)
2. **Stage 1** RF predicts probability of degradation
3. **CV gate** checks for independent evidence of degradation (strong brightness,
   laplacian, noise patterns)
4. If Stage 1 probability >= 0.50 OR strong CV evidence, **Stage 2** RF runs to
   identify the degradation type
5. **Decision layer** applies CV corrections for uncertain RF predictions
6. Final quality score is computed from identified issues

## Supported Quality Categories

| Category | Description |
|---|---|
| **ACCEPTABLE** | Score >= 80, no significant issues |
| **DEGRADED** | Score 50-79, one or more quality issues detected |
| **POTENTIALLY_DEFECTIVE** | Score < 50, significant quality problems |

Specific issue types: `blur`, `motion_blur`, `underexposure`, `overexposure`,
`noise`, `compression`

## Architecture

- **Backend**: FastAPI + Python + OpenCV + scikit-learn Random Forest models
- **Frontend**: React + Vite, communicates with backend via REST API
- **ML Pipeline**: Two-stage hierarchical classification with CV-independent gates
- **Deployment**: Docker Compose (backend on port 8000, frontend on port 5173)
- **Database**: SQLite for storing analysis history

## Installation (Local)

```bash
# Clone the repository
git clone <repo-url>
cd visioniq

# Create a Python virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the backend
uvicorn api:app --host 0.0.0.0 --port 8000

# In a separate terminal, run the frontend
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

The backend will be available at `http://localhost:8000` and the frontend at
`http://localhost:5173`.

## Running Tests

```bash
PYTHONPATH=. python -m pytest -q tests/
```

All 4 existing tests verify health check and image upload functionality.

## Running Evaluation

```bash
PYTHONPATH=. python evaluate_decision.py
```

This evaluates the ML pipeline on the test dataset (112 samples). Expected results:

```
STAGE 2 RF ACCURACY: 0.7573
FINAL ACCURACY: 0.7321
FINAL CONFIDENCE: 0.8225
CV override impact: 0 / 112
```

## Running with Docker

```bash
docker compose up --build
```

or

```bash
docker compose up -d  # detached mode
```

Services:
- **Backend**: `http://localhost:8000` (API docs: `http://localhost:8000/docs`)
- **Frontend**: `http://localhost:5173`

## How to Use the Application

1. Open the frontend at `http://localhost:5173`
2. Select an image file (JPEG, PNG, WEBP, or BMP)
3. Click "Analyze Image"
4. Wait for the analysis to complete - the quality score, label, and detected issues will be displayed
5. Use "Analyze another image" to process a different image without refreshing

## Expected Evaluation Result (Baseline)

The ML experimentation phase is complete. The current implementation represents
the best practical result obtained:

- **Evaluated**: 112 samples
- **Stage 2 Random Forest accuracy**: 75.73%
- **Final accuracy**: 73.21%
- **Final confidence**: 82.25%
- **CV override impact**: 0 / 112 (no predictions overridden by CV rules)

## Known Limitations

- Stage 2 accuracy (75.73%) and final accuracy (73.21%) reflect the best practical
  result from the completed experimentation phase
- Motion blur and compression diagnostics were already investigated during
  experimentation
- CV override rules have 0/112 impact on the evaluation set
- The pipeline was tested on 112 samples from the generated test dataset
- Oversized image handling is not explicitly limited
- Only JPEG, PNG, WEBP, and BMP file types are supported