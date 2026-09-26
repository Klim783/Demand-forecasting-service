from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint():
    payload = {
        "store_id": 1,
        "start_date": "2015-08-01",
        "horizon_days": 7,
        "promo": 1,
        "school_holiday": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["store_id"] == 1
    assert len(data["forecasts"]) == 7
    assert data["total_forecasted_sales"] >= 0