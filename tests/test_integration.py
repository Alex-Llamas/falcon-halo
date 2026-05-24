import pytest
from fastapi.testclient import TestClient
import os
from pathlib import Path

@pytest.fixture
def client(parquet_db, ants_folder):
    """Provides a TestClient for the FastAPI app, pointing to the test database and ANTS_FOLDER."""
    os.environ["PARQUET_DIR"] = str(parquet_db)
    os.environ["ANTS_FOLDER"] = str(ants_folder)
    from api import app
    # Re-trigger lifespan to load the mock data
    with TestClient(app) as c:
        yield c

def test_frontend_served(client):
    """Verify that the frontend index.html is served at the root."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Genetic Trait Enrichment Visualizer" in response.text
    # Verify that the API Base URL input is removed
    assert "API Base URL" not in response.text
    # Verify that relative path is used for tissue signature
    assert "url = `/api/v1/traits/${currentPhenotype.name}/genes/${currentGene}/tissue_signature?threshold=${threshold}`" in response.text

def test_tissue_signature_communication(client):
    """Verify that the frontend would be able to communicate with the tissue API."""
    # This simulates what the frontend would call
    trait = "T2D"
    gene = "COMT"
    threshold = 0.01
    url = f"/api/v1/traits/{trait}/genes/{gene}/tissue_signature?threshold={threshold}"

    response = client.get(url)
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert "signature_value" in data[0]
    assert data[0]["tissue"] == "vagina"

def test_cors_not_enabled(client):
    """Verify that CORS headers are not present by default."""
    response = client.get("/api/v1/traits", headers={"Origin": "http://example.com"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
