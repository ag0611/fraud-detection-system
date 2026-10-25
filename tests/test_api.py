from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

pytestmark = pytest.mark.skipif(
    not Path("model/final_model.joblib").exists(), reason="model file nahi hai"
)


def sample():
    d = {f"V{i}": 0.0 for i in range(1, 29)}
    d.update(Time=1000.0, Amount=25.0)
    return d


def test_health_and_predict():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/predict", json=sample())
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["fraud_probability"] <= 1
        assert body["decision"] in ("FRAUD", "LEGIT")
        assert len(body["top_reasons"]) == 3


def test_validation_error():
    with TestClient(app) as c:
        assert c.post("/predict", json={"Time": 1.0}).status_code == 422


def test_threshold_override():
    with TestClient(app) as c:
        r = c.post("/predict?threshold=0.0", json=sample())
        assert r.json()["decision"] == "FRAUD"
