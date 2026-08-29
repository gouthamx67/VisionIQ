import os
import sys
import tempfile
import shutil
import logging
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from quality.decision import analyze_image
from database import SessionLocal, engine
import models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("visioniq")

# Create the database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="VisionIQ API")

# Allow React Frontend (port 3000) to access API (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"message": "VisionIQ API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.post("/api/v1/analyze")
async def analyze_image_endpoint(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    logger.info(f"Received upload: {file.filename} (type: {file.content_type})")
    
    if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/bmp"]:
        logger.warning(f"Rejected unsupported type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload JPG, PNG, WEBP, or BMP.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name
    
    try:
        result = analyze_image(temp_path)
        logger.info(f"Analysis complete for {file.filename}: score={result['quality_score']}, label={result['quality_label']}, time={result['inference_time_ms']}ms")
        
        db_analysis = models.Analysis(
            filename=file.filename,
            quality_score=result["quality_score"],
            quality_label=result["quality_label"],
            issues=result["issues"],
            clean_vs_degraded_prob=result["clean_vs_degraded_prob"]
        )
        db.add(db_analysis)
        db.commit()
        db.refresh(db_analysis)
        
        result["id"] = db_analysis.id
        result["filename"] = file.filename
        return result
    except Exception as e:
        logger.error(f"Error analyzing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal analysis error")
    finally:
        os.remove(temp_path)

@app.get("/api/v1/analyses")
def get_analyses(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    logger.info(f"Fetching analyses: skip={skip}, limit={limit}")
    analyses = db.query(models.Analysis).order_by(models.Analysis.created_at.desc()).offset(skip).limit(limit).all()
    return analyses

@app.get("/api/v1/analyses/{analysis_id}")
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.query(models.Analysis).filter(models.Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis

@app.post("/api/v1/batch")
async def analyze_batch(files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    logger.info(f"Batch analysis started with {len(files)} files")
    results = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
        try:
            result = analyze_image(temp_path)
            db_analysis = models.Analysis(
                filename=file.filename,
                quality_score=result["quality_score"],
                quality_label=result["quality_label"],
                issues=result["issues"],
                clean_vs_degraded_prob=result["clean_vs_degraded_prob"]
            )
            db.add(db_analysis)
            db.commit()
            result["filename"] = file.filename
            results.append(result)
        finally:
            os.remove(temp_path)
    logger.info(f"Batch analysis complete: {len(results)} files")
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting VisionIQ API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
