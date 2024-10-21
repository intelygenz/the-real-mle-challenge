import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from api.api import create_app
from api.estimators.ny_classifier import NYClassifier


@pytest.fixture
def dummy_classifier():
    model = DummyClassifier(strategy="most_frequent")
    model.fit([[0]], [0])  # Dummy fit

    classifier = NYClassifier(model)

    return classifier


@pytest.fixture
def client(dummy_classifier):
    app = create_app(dummy_classifier)

    return TestClient(app)


def test_health_check(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_estimate(client):
    listing = {
        "id": 1001,
        "accommodates": 4,
        "room_type": "Entire home/apt",
        "beds": 2,
        "bedrooms": 1,
        "bathrooms": 2,
        "neighbourhood": "Brooklyn",
        "tv": 1,
        "elevator": 1,
        "internet": 0,
        "latitude": 40.71383,
        "longitude": -73.9658
    }
    response = client.post("/estimate", json=listing)

    assert response.status_code == 200
    assert "id" in response.json()
    assert "price_category" in response.json()
