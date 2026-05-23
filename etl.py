import argparse
import logging
from pathlib import Path
import duckdb

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_args():
    parser = argparse.ArgumentParser(description="ETL pipeline for genomic data.")
    parser.add_argument("--base-dir", default="/humgen/diabetes/loki/pipelines/real_data/kp5/projects/bottom-line/", help="Base directory to scan for traits.")
    parser.add_argument("--output-dir", default="./parquet_db/", help="Output directory for Parquet files.")
    return parser.parse_args()

def process_trait(con, trait_name, trait_path, output_dir):
    pegs_path = trait_path / "1000G" / "no_filter" / "default" / "pegs"
    if not pegs_path.exists():
        logging.warning(f"Path not found: {pegs_path}")
        return

    patterns = {
        "genes": "pegs1.*.genes",
        "variants": "pegs1.*.variants",
        "v2g": "pegs1.*.v2g"
    }

    schema_genes = {
        "GENE": "VARCHAR",
        "PROBABILITY": "DOUBLE",
        "GENE_R": "DOUBLE",
        "GENE_STAT": "VARCHAR",
        "WINDOW": "DOUBLE",
        "DECAY": "DOUBLE",
        "WINDOW_R": "DOUBLE",
        "WINDOW_STAT": "VARCHAR",
        "BETA": "DOUBLE",
        "SE": "DOUBLE",
        "P_VALUE": "DOUBLE",
        "NEG_LOG_P": "DOUBLE",
        "CHR": "VARCHAR",
        "START": "BIGINT",
        "END": "BIGINT",
        "NEAREST_TO_LEAD": "VARCHAR",
        "CLUMP": "VARCHAR"
    }

    schema_variants = {
        "RSID": "VARCHAR",
        "CHR": "VARCHAR",
        "POS": "BIGINT",
        "REF": "VARCHAR",
        "ALT": "VARCHAR",
        "PROBABILITY": "DOUBLE",
        "PRIOR": "DOUBLE",
        "BETA": "DOUBLE",
        "SE": "DOUBLE",
        "Z_SCORE": "DOUBLE",
        "P_VALUE": "DOUBLE",
        "GENE_1": "VARCHAR",
        "LINK_SC_1": "DOUBLE",
        "GENE_2": "VARCHAR",
        "LINK_SC_2": "DOUBLE",
        "GENE_3": "VARCHAR",
        "LINK_SC_3": "VARCHAR",
        "LINK_SC_S2G_genes": "VARCHAR",
        "S2G_scores": "VARCHAR",
        "NORM_BETA": "VARCHAR",
        "LOCAL_SIGMA_2": "DOUBLE",
        "LEAD_SNP": "VARCHAR",
        "NEAREST_GENE": "VARCHAR",
        "NEAREST_DISTANCE": "VARCHAR",
        "CLUMP": "VARCHAR",
        "GWAS_Z": "DOUBLE",
        "GWAS_P": "DOUBLE",
        "GWAS_BETA": "DOUBLE",
        "GWAS_SE": "DOUBLE",
        "GWAS_AF": "DOUBLE",
        "GWAS_N": "DOUBLE"
    }

    schema_v2g = {
        "rsID": "VARCHAR",
        "Gene": "VARCHAR",
        "Value": "DOUBLE"
    }

    for key, pattern in patterns.items():
        files = list(pegs_path.glob(pattern))
        if not files:
            logging.warning(f"No files found for {key} in {pegs_path}")
            continue

        logging.info(f"Processing {len(files)} {key} files for trait {trait_name}")
        schema = schema_genes if key == "genes" else schema_variants if key == "variants" else schema_v2g
        types_str = ", ".join([f"'{col}': '{dtype}'" for col, dtype in schema.items()])
        file_list = [str(f) for f in files]

        trait_output_dir = Path(output_dir) / f"{key}.parquet" / f"trait={trait_name}"
        trait_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # We use header=False and skip=1 because some pipeline outputs have
            # malformed headers (e.g. fewer columns in header than in data).
            query = f"""
                COPY (
                    SELECT *, '{trait_name}' as trait
                    FROM read_csv({file_list}, sep='\t', header=False, skip=1, columns={{{types_str}}}, auto_detect=False)
                ) TO '{trait_output_dir / 'data.parquet'}' (FORMAT PARQUET, COMPRESSION 'SNAPPY')
            """
            con.execute(query)
        except Exception as e:
            logging.error(f"Error processing {key} for {trait_name}: {e}")

def main():
    setup_logging()
    args = get_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_dir.exists():
        logging.error(f"Base directory {base_dir} does not exist.")
        return

    con = duckdb.connect(database=':memory:')
    for trait_path in base_dir.iterdir():
        if trait_path.is_dir():
            trait_name = trait_path.name
            logging.info(f"Starting ETL for trait: {trait_name}")
            process_trait(con, trait_name, trait_path, output_dir)

    logging.info("ETL process completed.")

if __name__ == "__main__":
    main()
