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
ANTS_FOLDER = os.getenv("ANTS_FOLDER", "/humgen/diabetes/loki/data/annotation/common/ATACseq/")
GENES_PARQUET = str(Path(PARQUET_DIR) / "genes.parquet/**/*.parquet")
VARIANTS_PARQUET = str(Path(PARQUET_DIR) / "variants.parquet/**/*.parquet")
V2G_PARQUET = str(Path(PARQUET_DIR) / "v2g.parquet/**/*.parquet")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DuckDB connection
    logging.info("Initializing DuckDB connection")
    app.state.db = duckdb.connect(database=':memory:')

    # Pre-compute/Load genomic regions if ANTS_FOLDER exists
    if Path(ANTS_FOLDER).exists():
        logging.info(f"Loading genomic regions from {ANTS_FOLDER}")
        try:
            # We use filename=True to extract annotation and tissue from the filename
            # Filename format: {annotation}___{tissue}.csv
            # We use list of files to read_csv
            csv_files = [str(f) for f in Path(ANTS_FOLDER).glob("*.csv")]
            if csv_files:
                # Create a table for genomic regions
                # Column 0: chr, 1: start, 2: end, 4: tissue (from file content)
                app.state.db.execute(f"""
                    CREATE TABLE genomic_regions_raw AS
                    SELECT
                        column0 AS chr,
                        column1 AS start,
                        column2 AS end,
                        column4 AS tissue,
                        regexp_extract(filename, '([^/]+)___([^/.]+)\\.csv$', 1) AS annotation,
                        regexp_extract(filename, '([^/]+)___([^/.]+)\\.csv$', 2) AS tissue_from_file
                    FROM read_csv({csv_files}, sep='\t', header=False, filename=True, all_varchar=True)
                """)
                # Handle Chromosome mapping as in example and compute absolute positions
                # Using double quotes around reserved keyword "end"
                app.state.db.execute("""
                    CREATE TABLE genomic_regions AS
                    SELECT
                        annotation, tissue, tissue_from_file,
                        CAST("start" AS BIGINT) + (
                            CASE
                                WHEN chr = 'X' THEN 23
                                WHEN chr = 'Y' THEN 24
                                WHEN chr = 'MT' THEN 25
                                ELSE TRY_CAST(chr AS BIGINT)
                            END * CAST(1000000000 AS BIGINT)
                        ) AS abs_start,
                        CAST("end" AS BIGINT) + (
                            CASE
                                WHEN chr = 'X' THEN 23
                                WHEN chr = 'Y' THEN 24
                                WHEN chr = 'MT' THEN 25
                                ELSE TRY_CAST(chr AS BIGINT)
                            END * CAST(1000000000 AS BIGINT)
                        ) AS abs_end
                    FROM genomic_regions_raw;
                    CREATE INDEX idx_abs_pos ON genomic_regions (abs_start, abs_end);
                    DROP TABLE genomic_regions_raw;
                """)
                logging.info("Genomic regions loaded successfully.")
            else:
                logging.warning(f"No CSV files found in {ANTS_FOLDER}")
        except Exception as e:
            logging.error(f"Failed to load genomic regions: {e}")
    else:
        logging.warning(f"ANTS_FOLDER {ANTS_FOLDER} does not exist. Tissue signature APIs will be unavailable.")

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

# Light APIs
@app.get("/api/v1/light/genes/{gene_name}")
async def get_gene_light(gene_name: str):
    try:
        query = f"SELECT GENE, PROBABILITY, P_VALUE, trait FROM read_parquet('{GENES_PARQUET}') WHERE UPPER(GENE) = UPPER(?) AND PROBABILITY >= 0.01"
        df = app.state.db.execute(query, [gene_name]).df()
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Gene {gene_name} not found or doesn't meet probability threshold")
        return df.to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/light/variants/{rsid}")
async def get_variant_light(rsid: str):
    try:
        # We need to join with v2g to check v2g_value >= 0.01
        query = f"""
            SELECT
                v.RSID, v.CHR, v.POS, v.PROBABILITY, v2g.Value AS v2g_value, v.trait
            FROM read_parquet('{VARIANTS_PARQUET}') v
            INNER JOIN read_parquet('{V2G_PARQUET}') v2g
                ON UPPER(v.RSID) = UPPER(v2g.rsID) AND v.trait = v2g.trait
            WHERE UPPER(v.RSID) = UPPER(?) AND v.PROBABILITY >= 0.01 AND v2g.Value >= 0.01
        """
        df = app.state.db.execute(query, [rsid]).df()
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Variant {rsid} not found or doesn't meet thresholds")
        return df.to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/light/genes/{gene_name}/comprehensive")
async def get_gene_comprehensive_light(gene_name: str):
    try:
        gene_query = f"SELECT GENE, PROBABILITY, P_VALUE, trait FROM read_parquet('{GENES_PARQUET}') WHERE UPPER(GENE) = UPPER(?) AND PROBABILITY >= 0.01"
        gene_df = app.state.db.execute(gene_query, [gene_name]).df()

        if gene_df.empty:
            raise HTTPException(status_code=404, detail=f"Gene {gene_name} not found or doesn't meet probability threshold")

        join_query = f"""
            SELECT
                v.RSID, v.CHR, v.POS, v.PROBABILITY, v2g.Value AS v2g_value, v.trait
            FROM read_parquet('{VARIANTS_PARQUET}') v
            INNER JOIN read_parquet('{V2G_PARQUET}') v2g
                ON UPPER(v.RSID) = UPPER(v2g.rsID) AND v.trait = v2g.trait
            WHERE UPPER(v2g.Gene) = UPPER(?) AND v.PROBABILITY >= 0.01 AND v2g.Value >= 0.01
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

@app.get("/api/v1/light/traits/{trait_name}/genes/{gene_name}/comprehensive")
async def get_trait_gene_comprehensive_light(trait_name: str, gene_name: str):
    try:
        gene_query = f"SELECT GENE, PROBABILITY, P_VALUE, trait FROM read_parquet('{GENES_PARQUET}') WHERE UPPER(GENE) = UPPER(?) AND trait = ? AND PROBABILITY >= 0.01"
        gene_df = app.state.db.execute(gene_query, [gene_name, trait_name]).df()

        if gene_df.empty:
            raise HTTPException(status_code=404, detail=f"Gene {gene_name} not found for trait {trait_name} or doesn't meet probability threshold")

        join_query = f"""
            SELECT
                v.RSID, v.CHR, v.POS, v.PROBABILITY, v2g.Value AS v2g_value
            FROM read_parquet('{VARIANTS_PARQUET}') v
            INNER JOIN read_parquet('{V2G_PARQUET}') v2g
                ON UPPER(v.RSID) = UPPER(v2g.rsID) AND v.trait = v2g.trait
            WHERE UPPER(v2g.Gene) = UPPER(?) AND v.trait = ? AND v.PROBABILITY >= 0.01 AND v2g.Value >= 0.01
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

@app.get("/api/v1/traits/{trait_name}/genes/{gene_name}/tissue_signature")
async def get_trait_gene_tissue_signature(
    trait_name: str,
    gene_name: str,
    threshold: float = Query(0.1, description="Threshold for both gene and variant probabilities")
):
    try:
        # Check if genomic_regions table exists
        try:
            app.state.db.execute("SELECT 1 FROM genomic_regions LIMIT 1")
        except:
            raise HTTPException(status_code=503, detail="Tissue signature data not loaded (ANTS_FOLDER missing or empty)")

        # 1. Get filtered variants and their probabilities
        # We need absolute position for join
        var_query = f"""
            WITH filtered_variants AS (
                SELECT
                    v.RSID, v.CHR, v.POS, v.PROBABILITY, v2g.Value AS v2g_value,
                    CASE
                        WHEN v.CHR = 'X' THEN 23
                        WHEN v.CHR = 'Y' THEN 24
                        WHEN v.CHR = 'MT' THEN 25
                        ELSE TRY_CAST(v.CHR AS BIGINT)
                    END AS chr_num,
                    CAST(v.POS AS BIGINT) + (chr_num * CAST(1000000000 AS BIGINT)) AS abs_pos
                FROM read_parquet('{VARIANTS_PARQUET}') v
                INNER JOIN read_parquet('{V2G_PARQUET}') v2g
                    ON UPPER(v.RSID) = UPPER(v2g.rsID) AND v.trait = v2g.trait
                INNER JOIN read_parquet('{GENES_PARQUET}') g
                    ON g.GENE = v2g.Gene AND g.trait = v.trait
                WHERE UPPER(v2g.Gene) = UPPER(?)
                  AND v.trait = ?
                  AND v.PROBABILITY >= ?
                  AND g.PROBABILITY >= ?
            )
            SELECT
                r.annotation, r.tissue, r.tissue_from_file,
                SUM(fv.v2g_value * fv.PROBABILITY) AS signature_value
            FROM filtered_variants fv
            INNER JOIN genomic_regions r
                ON fv.abs_pos >= r.abs_start AND fv.abs_pos <= r.abs_end
            GROUP BY r.annotation, r.tissue, r.tissue_from_file
            ORDER BY signature_value DESC
        """
        df = app.state.db.execute(var_query, [gene_name, trait_name, threshold, threshold]).df()
        return df.to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in tissue_signature: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/genes/{gene_name}/tissue_signature")
async def get_gene_tissue_signature(
    gene_name: str,
    threshold: float = Query(0.1, description="Threshold for both gene and variant probabilities")
):
    try:
        try:
            app.state.db.execute("SELECT 1 FROM genomic_regions LIMIT 1")
        except:
            raise HTTPException(status_code=503, detail="Tissue signature data not loaded (ANTS_FOLDER missing or empty)")

        var_query = f"""
            WITH filtered_variants AS (
                SELECT
                    v.RSID, v.CHR, v.POS, v.PROBABILITY, v2g.Value AS v2g_value,
                    CASE
                        WHEN v.CHR = 'X' THEN 23
                        WHEN v.CHR = 'Y' THEN 24
                        WHEN v.CHR = 'MT' THEN 25
                        ELSE TRY_CAST(v.CHR AS BIGINT)
                    END AS chr_num,
                    CAST(v.POS AS BIGINT) + (chr_num * CAST(1000000000 AS BIGINT)) AS abs_pos
                FROM read_parquet('{VARIANTS_PARQUET}') v
                INNER JOIN read_parquet('{V2G_PARQUET}') v2g
                    ON UPPER(v.RSID) = UPPER(v2g.rsID) AND v.trait = v2g.trait
                INNER JOIN read_parquet('{GENES_PARQUET}') g
                    ON g.GENE = v2g.Gene AND g.trait = v.trait
                WHERE UPPER(v2g.Gene) = UPPER(?)
                  AND v.PROBABILITY >= ?
                  AND g.PROBABILITY >= ?
            )
            SELECT
                r.annotation, r.tissue, r.tissue_from_file,
                SUM(fv.v2g_value * fv.PROBABILITY) AS signature_value
            FROM filtered_variants fv
            INNER JOIN genomic_regions r
                ON fv.abs_pos >= r.abs_start AND fv.abs_pos <= r.abs_end
            GROUP BY r.annotation, r.tissue, r.tissue_from_file
            ORDER BY signature_value DESC
        """
        df = app.state.db.execute(var_query, [gene_name, threshold, threshold]).df()
        return df.to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in gene tissue_signature: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
