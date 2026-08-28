import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File
from quality.decision import analyze_image

app = FastAPI(title="VisionIQ API")

@app.get("/")
def read_root():
    return {"message": "VisionIQ API is running!"}

@app.post("/analyze")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name
    
    try:
        result = analyze_image(temp_path)
        return result
    finally:
        os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
