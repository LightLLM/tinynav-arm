from fastapi.testclient import TestClient

from tinynav.api import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_blocked_scenario_stops() -> None:
    response = client.get("/api/frame/blocked")
    assert response.status_code == 200
    assert response.json()["decision"]["action"] == "STOP"
