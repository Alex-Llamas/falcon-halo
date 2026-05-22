from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
import duckdb
import os
import logging
from pathlib import Path
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
PARQUET_DIR = os.getenv("PARQUET_DIR", "./parquet_db/")
GENES_PARQUET = str(Path(PARQUET_DIR) / "genes.parquet/**/*.parquet")
VARIANTS_PARQUET = str(Path(PARQUET_DIR) / "variants.parquet/**/*.parquet")
V2G_PARQUET = str(Path(PARQUET_DIR) / "v2g.parquet/**/*.parquet")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DuckDB connection
    logging.info("Initializing DuckDB connection")
    app.state.db = duckdb.connect(database=':memory:')
    yield
    logging.info("Closing DuckDB connection")
    app.state.db.close()

app = FastAPI(title="Genomic Data API", lifespan=lifespan)

@app.get("/api/v1/traits")
async def get_traits():
    try:
        res = app.state.db.execute(f"SELECT DISTINCT trait FROM read_parquet('{GENES_PARQUET}')").fetchall()
        return [r[0] for r in res]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/genes/{gene_name}")
async def get_gene(gene_name: str):
    try:
        query = f"SELECT * FROM read_parquet('{GENES_PARQUET}') WHERE UPPER(GENE) = UPPER(?)"
        df = app.state.db.execute(query, [gene_name]).df()
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Gene {gene_name} not found")
        return df.to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/variants/{rsid}")
async def get_variant(rsid: str):
    try:
        var_query = f"SELECT * FROM read_parquet('{VARIANTS_PARQUET}') WHERE UPPER(RSID) = UPPER(?)"
        var_df = app.state.db.execute(var_query, [rsid]).df()

        if var_df.empty:
            raise HTTPException(status_code=404, detail=f"Variant {rsid} not found")

        v2g_query = f"SELECT * FROM read_parquet('{V2G_PARQUET}') WHERE UPPER(rsID) = UPPER(?)"
        v2g_df = app.state.db.execute(v2g_query, [rsid]).df()

        results = var_df.to_dict(orient="records")
        for record in results:
            trait = record['trait']
            trait_v2g = v2g_df[v2g_df['trait'] == trait].to_dict(orient="records")
            record['v2g_mappings'] = trait_v2g

        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/traits/{trait_name}")
async def get_trait_data(
    trait_name: str,
    dataset: str = Query("genes", pattern="^(genes|variants|v2g)$"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    try:
        parquet_path = str(Path(PARQUET_DIR) / f"{dataset}.parquet" / f"trait={trait_name}" / "*.parquet")
        if not Path(PARQUET_DIR).joinpath(f"{dataset}.parquet", f"trait={trait_name}").exists():
            raise HTTPException(status_code=404, detail=f"Trait {trait_name} not found for dataset {dataset}")

        query = f"SELECT * FROM read_parquet('{parquet_path}') LIMIT ? OFFSET ?"
        df = app.state.db.execute(query, [limit, offset]).df()
        return df.to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/genes/{gene_name}/comprehensive")
async def get_gene_comprehensive(gene_name: str):
    try:
        gene_query = f"SELECT * FROM read_parquet('{GENES_PARQUET}') WHERE UPPER(GENE) = UPPER(?)"
        gene_df = app.state.db.execute(gene_query, [gene_name]).df()

        if gene_df.empty:
            raise HTTPException(status_code=404, detail=f"Gene {gene_name} not found")

        join_query = f"""
            SELECT
                v.*,
                v2g.Value AS v2g_value
            FROM read_parquet('{VARIANTS_PARQUET}') v
            INNER JOIN read_parquet('{V2G_PARQUET}') v2g
                ON UPPER(v.RSID) = UPPER(v2g.rsID) AND v.trait = v2g.trait
            WHERE UPPER(v2g.Gene) = UPPER(?)
        """
        variants_df = app.state.db.execute(join_query, [gene_name]).df()

        response = {
            "gene": gene_name,
            "traits": []
        }

        for _, gene_row in gene_df.iterrows():
            trait = gene_row['trait']
            trait_variants = variants_df[variants_df['trait'] == trait]

            response["traits"].append({
                "trait_name": trait,
                "gene_data": gene_row.to_dict(),
                "variants": trait_variants.to_dict(orient="records")
            })

        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/traits/{trait_name}/genes/{gene_name}/comprehensive")
async def get_trait_gene_comprehensive(trait_name: str, gene_name: str):
    try:
        gene_query = f"SELECT * FROM read_parquet('{GENES_PARQUET}') WHERE UPPER(GENE) = UPPER(?) AND trait = ?"
        gene_df = app.state.db.execute(gene_query, [gene_name, trait_name]).df()

        if gene_df.empty:
            raise HTTPException(status_code=404, detail=f"Gene {gene_name} not found for trait {trait_name}")

        join_query = f"""
            SELECT
                v.*,
                v2g.Value AS v2g_value
            FROM read_parquet('{VARIANTS_PARQUET}') v
            INNER JOIN read_parquet('{V2G_PARQUET}') v2g
                ON UPPER(v.RSID) = UPPER(v2g.rsID) AND v.trait = v2g.trait
            WHERE UPPER(v2g.Gene) = UPPER(?) AND v.trait = ?
        """
        variants_df = app.state.db.execute(join_query, [gene_name, trait_name]).df()

        return {
            "trait_name": trait_name,
            "gene_name": gene_name,
            "gene_data": gene_df.iloc[0].to_dict(),
            "variants": variants_df.to_dict(orient="records")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
