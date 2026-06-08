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
    parser.add_argument("--signature-matrix", help="Path to pre-computed tissue signature matrix TSV.")
    return parser.parse_args()

def process_signature_matrix(con, matrix_path, output_dir):
    matrix_path = Path(matrix_path)
    if not matrix_path.exists():
        logging.error(f"Signature matrix file not found: {matrix_path}")
        return

    logging.info(f"Processing signature matrix: {matrix_path}")

    target_dir = Path(output_dir) / "precomputed_signatures"
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Columns in TSV: gene, trait, annotation___tissue, biosample, value
        # We split annotation___tissue and ignore biosample
        query = f"""
            COPY (
                SELECT
                    gene,
                    trait,
                    split_part("annotation___tissue", '___', 1) AS annotation,
                    split_part("annotation___tissue", '___', 2) AS tissue,
                    value
                FROM read_csv('{matrix_path}',
                    header=True,
                    sep='\t',
                    columns={{
                        'gene': 'VARCHAR',
                        'trait': 'VARCHAR',
                        'annotation___tissue': 'VARCHAR',
                        'biosample': 'VARCHAR',
                        'value': 'DOUBLE'
                    }})
            ) TO '{target_dir}' (FORMAT PARQUET, PARTITION_BY (gene), COMPRESSION 'SNAPPY', OVERWRITE_OR_IGNORE)
        """
        con.execute(query)
        logging.info(f"Signature matrix processed and saved to {target_dir}")
    except Exception as e:
        logging.error(f"Error processing signature matrix: {e}")

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
        # Only read chromosome files from 1 to 22
        files = []
        for chrom in range(1, 23):
            # Substitute '*' with chrom in pattern
            chr_pattern = pattern.replace("*", str(chrom))
            files.extend(list(pegs_path.glob(chr_pattern)))

        if not files:
            logging.warning(f"No files found for {key} (Chr 1-22) in {pegs_path}")
            continue

        logging.info(f"Processing {len(files)} {key} files (Chr 1-22) for trait {trait_name}")
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

    con = duckdb.connect(database=':memory:')

    # Process traits if base_dir exists and contains directories
    if base_dir.exists():
        for trait_path in base_dir.iterdir():
            if trait_path.is_dir():
                trait_name = trait_path.name
                logging.info(f"Starting ETL for trait: {trait_name}")
                process_trait(con, trait_name, trait_path, output_dir)
    else:
        logging.warning(f"Base directory {base_dir} does not exist. Skipping trait processing.")

    # Process signature matrix if provided
    if args.signature_matrix:
        process_signature_matrix(con, args.signature_matrix, output_dir)

    logging.info("ETL process completed.")

if __name__ == "__main__":
    main()
