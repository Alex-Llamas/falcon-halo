import pytest
from fastapi.testclient import TestClient
import os

@pytest.fixture
def client(parquet_db):
    """Provides a TestClient for the FastAPI app, pointing to the test database."""
    os.environ["PARQUET_DIR"] = str(parquet_db)
    from api import app
    with TestClient(app) as c:
        yield c

def test_get_traits(client):
    response = client.get("/api/v1/traits")
    assert response.status_code == 200
    traits = response.json()
    assert "T2D" in traits
    assert "Height" in traits

def test_get_gene(client):
    # Test case-insensitive search
    response = client.get("/api/v1/genes/comt")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert data[0]["GENE"] == "COMT"

def test_get_gene_not_found(client):
    response = client.get("/api/v1/genes/NONEXISTENT")
    assert response.status_code == 404

def test_get_variant(client):
    response = client.get("/api/v1/variants/rs165722")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert data[0]["RSID"] == "rs165722"
    assert "v2g_mappings" in data[0]
    assert len(data[0]["v2g_mappings"]) > 0

def test_get_trait_data_pagination(client):
    # genes dataset
    response = client.get("/api/v1/traits/T2D?dataset=genes&limit=1&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1

    # variants dataset
    response = client.get("/api/v1/traits/T2D?dataset=variants&limit=1")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1

def test_get_gene_comprehensive(client):
    response = client.get("/api/v1/genes/COMT/comprehensive")
    assert response.status_code == 200
    data = response.json()
    assert data["gene"] == "COMT"
    assert len(data["traits"]) > 0

    trait_entry = data["traits"][0]
    assert "gene_data" in trait_entry
    assert "variants" in trait_entry
    assert len(trait_entry["variants"]) > 0

def test_get_trait_gene_comprehensive(client):
    response = client.get("/api/v1/traits/T2D/genes/COMT/comprehensive")
    assert response.status_code == 200
    data = response.json()
    assert data["trait_name"] == "T2D"
    assert data["gene_name"] == "COMT"
    assert "gene_data" in data
    assert len(data["variants"]) > 0

def test_get_gene_light(client):
    response = client.get("/api/v1/light/genes/COMT")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    # Check columns
    assert set(data[0].keys()) == {"GENE", "PROBABILITY", "P_VALUE", "trait"}

def test_get_variant_light(client):
    response = client.get("/api/v1/light/variants/rs165722")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    # Check columns
    assert set(data[0].keys()) == {"RSID", "CHR", "POS", "PROBABILITY", "v2g_value", "trait"}

def test_get_gene_comprehensive_light(client):
    response = client.get("/api/v1/light/genes/COMT/comprehensive")
    assert response.status_code == 200
    data = response.json()
    assert data["gene"] == "COMT"
    assert len(data["traits"]) > 0

    trait_entry = data["traits"][0]
    assert set(trait_entry["gene_data"].keys()) == {"GENE", "PROBABILITY", "P_VALUE", "trait"}
    if len(trait_entry["variants"]) > 0:
        assert set(trait_entry["variants"][0].keys()) == {"RSID", "CHR", "POS", "PROBABILITY", "v2g_value", "trait"}

def test_get_trait_gene_comprehensive_light(client):
    response = client.get("/api/v1/light/traits/T2D/genes/COMT/comprehensive")
    assert response.status_code == 200
    data = response.json()
    assert data["trait_name"] == "T2D"
    assert data["gene_name"] == "COMT"
    assert set(data["gene_data"].keys()) == {"GENE", "PROBABILITY", "P_VALUE", "trait"}
    if len(data["variants"]) > 0:
        assert set(data["variants"][0].keys()) == {"RSID", "CHR", "POS", "PROBABILITY", "v2g_value"}
