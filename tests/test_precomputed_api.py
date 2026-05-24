import pytest
from fastapi.testclient import TestClient
import os
import shutil
from pathlib import Path
import duckdb
import importlib
import api

# Setup temporary parquet dir for testing
TEST_PARQUET_DIR = "/tmp/test_api_precomputed_db"

@pytest.fixture(scope="module", autouse=True)
def setup_test_data():
    if os.path.exists(TEST_PARQUET_DIR):
        shutil.rmtree(TEST_PARQUET_DIR)
    os.makedirs(TEST_PARQUET_DIR)

    # Create mock parquet file for precomputed signatures
    con = duckdb.connect(':memory:')
    con.execute("CREATE TABLE sigs (gene VARCHAR, trait VARCHAR, annotation VARCHAR, tissue VARCHAR, value DOUBLE)")
    con.execute("INSERT INTO sigs VALUES ('GALNT2', 'TGtoHDL', 'enhancer', 'heart', 8.9481)")
    con.execute("INSERT INTO sigs VALUES ('GALNT2', 'TGtoHDL', 'promoter', 'blood_vessel', 1.1739)")
    con.execute("INSERT INTO sigs VALUES ('PPARG', 'T2D', 'enhancer', 'adipose_tissue', 10.5)")
    con.execute("INSERT INTO sigs VALUES ('PPARG', 'CAD', 'enhancer', 'heart', 3.1)")

    target_path = Path(TEST_PARQUET_DIR) / "precomputed_signatures"
    con.execute(f"COPY sigs TO '{target_path}' (FORMAT PARQUET, PARTITION_BY (gene))")

    # Update environment and reload api module to pick up new constants
    os.environ["PARQUET_DIR"] = TEST_PARQUET_DIR
    importlib.reload(api)

    yield

    if os.path.exists(TEST_PARQUET_DIR):
        shutil.rmtree(TEST_PARQUET_DIR)

def test_get_precomputed_signatures_per_trait():
    from api import app
    with TestClient(app) as client:
        response = client.get("/api/v1/precomputed_signatures/gene/GALNT2?trait=TGtoHDL")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]['trait'] == 'TGtoHDL'
        assert data[0]['signature_value'] == 8.9481

def test_get_precomputed_signatures_global_sum():
    from api import app
    with TestClient(app) as client:
        response = client.get("/api/v1/precomputed_signatures/gene/PPARG?aggregation=sum")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        for item in data:
            if item['tissue'] == 'adipose_tissue':
                assert item['signature_value'] == 10.5
            elif item['tissue'] == 'heart':
                assert item['signature_value'] == 3.1

def test_get_precomputed_signatures_annotation_filter():
    from api import app
    with TestClient(app) as client:
        response = client.get("/api/v1/precomputed_signatures/gene/GALNT2?trait=TGtoHDL&annotation=enhancer")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]['annotation'] == 'enhancer'
        assert data[0]['signature_value'] == 8.9481
