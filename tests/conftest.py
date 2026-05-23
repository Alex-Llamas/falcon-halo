import pytest
from pathlib import Path
import shutil
import os

@pytest.fixture
def mock_genomic_data(tmp_path):
    """Creates a mock directory structure with genomic TSV files."""
    base_dir = tmp_path / "raw_data"
    traits = ["T2D", "Height"]
    chromosomes = ["21", "22"]

    gene_content = "GENE\tPROBABILITY\tGENE_R\tGENE_STAT\tWINDOW\tDECAY\tWINDOW_R\tWINDOW_STAT\tBETA\tSE\tP_VALUE\tNEG_LOG_P\tCHR\tSTART\tEND\tNEAREST_TO_LEAD\tCLUMP\n"
    gene_content += "COMT\t0.05\t0.9\tTrue\t0.0\t-1500.0\t1.0\tTrue\t0.1\t0.01\t1e-10\t10.0\t22\t19948863\t19950000\tTrue\tNone\n"

    # Correct columns count matching etl.py (31 columns)
    # 1.RSID 2.CHR 3.POS 4.REF 5.ALT 6.PROBABILITY 7.PRIOR 8.BETA 9.SE 10.Z_SCORE 11.P_VALUE
    # 12.GENE_1 13.LINK_SC_1 14.GENE_2 15.LINK_SC_2 16.GENE_3 17.LINK_SC_3 18.LINK_SC_S2G_genes
    # 19.S2G_scores 20.NORM_BETA 21.LOCAL_SIGMA_2 22.LEAD_SNP 23.NEAREST_GENE 24.NEAREST_DISTANCE
    # 25.CLUMP 26.GWAS_Z 27.GWAS_P 28.GWAS_BETA 29.GWAS_SE 30.GWAS_AF 31.GWAS_N

    variant_content = "RSID\tCHR\tPOS\tREF\tALT\tPROBABILITY\tPRIOR\tBETA\tSE\tZ_SCORE\tP_VALUE\tGENE_1\tLINK_SC_1\tGENE_2\tLINK_SC_2\tGENE_3\tLINK_SC_3\tLINK_SC_S2G_genes\tS2G_scores\tNORM_BETA\tLOCAL_SIGMA_2\tLEAD_SNP\tNEAREST_GENE\tNEAREST_DISTANCE\tCLUMP\tGWAS_Z\tGWAS_P\tGWAS_BETA\tGWAS_SE\tGWAS_AF\tGWAS_N\n"
    variant_content += "rs165722\t22\t19949013\tC\tT\t0.5\t1e-4\t4e-3\t1e-3\t3.6\t3e-4\tCOMT\t0.8\tARVCF\t0.4\tTANGO2\t0.2\tNone\tNone\t8e-3\t1e-6\tFalse\tCOMT\t1\t22_1\t7.0\t1e-12\t0.01\t0.001\t0.5\t3.6e6\n"

    v2g_content = "rsID\tGene\tValue\n"
    v2g_content += "rs165722\tCOMT\t0.8\n"

    for trait in traits:
        pegs_path = base_dir / trait / "1000G" / "no_filter" / "default" / "pegs"
        pegs_path.mkdir(parents=True)
        for chr in chromosomes:
            (pegs_path / f"pegs1.{chr}.genes").write_text(gene_content)
            (pegs_path / f"pegs.{chr}.variants").write_text(variant_content)
            (pegs_path / f"pegs.{chr}.v2g").write_text(v2g_content)

    return base_dir

@pytest.fixture
def parquet_db(mock_genomic_data, tmp_path):
    """Runs ETL on mock data and returns the output directory."""
    from etl import process_trait
    import duckdb

    output_dir = tmp_path / "parquet_db"
    con = duckdb.connect(database=':memory:')

    for trait_path in mock_genomic_data.iterdir():
        if trait_path.is_dir():
            process_trait(con, trait_path.name, trait_path, output_dir)

    return output_dir
