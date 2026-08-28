import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    # Updated assertion to handle the new timestamp field
    assert response.json()["status"] == "healthy"
    assert "timestamp" in response.json()

def test_upload_image():
    with open("dataset/raw/clean/" + open("dataset/splits/train.txt").readline().strip(), "rb") as f:
        response = client.post("/api/v1/analyze", files={"file": ("test.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    assert "quality_score" in response.json()
