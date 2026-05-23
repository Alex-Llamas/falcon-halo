import pytest
import duckdb
from pathlib import Path

def test_etl_output_structure(parquet_db):
    """Verify the partitioned Parquet structure."""
    assert (parquet_db / "genes.parquet").exists()
    assert (parquet_db / "variants.parquet").exists()
    assert (parquet_db / "v2g.parquet").exists()

    # Check partitions
    assert (parquet_db / "genes.parquet" / "trait=T2D").exists()
    assert (parquet_db / "genes.parquet" / "trait=Height").exists()

def test_etl_data_content(parquet_db):
    """Verify the content and types of the processed data."""
    con = duckdb.connect()

    # Query genes
    res = con.execute(f"SELECT * FROM read_parquet('{parquet_db}/genes.parquet/**/*.parquet') WHERE trait='T2D'").df()
    assert len(res) > 0
    assert "COMT" in res["GENE"].values
    assert res["P_VALUE"].dtype == "float64"
    assert res["START"].dtype == "int64"

    # Query variants
    res_var = con.execute(f"SELECT * FROM read_parquet('{parquet_db}/variants.parquet/**/*.parquet')").df()
    assert "rs165722" in res_var["RSID"].values
    assert res_var["POS"].dtype == "int64"

    # Query v2g
    res_v2g = con.execute(f"SELECT * FROM read_parquet('{parquet_db}/v2g.parquet/**/*.parquet')").df()
    assert "rs165722" in res_v2g["rsID"].values
    assert res_v2g["Value"].dtype == "float64"

def test_etl_trait_column_injection(parquet_db):
    """Verify that the trait column is correctly injected into every row."""
    con = duckdb.connect()
    traits = con.execute(f"SELECT DISTINCT trait FROM read_parquet('{parquet_db}/genes.parquet/**/*.parquet')").fetchall()
    trait_list = [t[0] for t in traits]
    assert "T2D" in trait_list
    assert "Height" in trait_list
