from optparse import OptionParser
import pandas as pd
import sqlite3
import json
import os
import gc
import time


def main():
    parser = OptionParser("usage: %prog [options]")

    # Base arguments
    parser.add_option("", "--falcon-prefix", help="Folder to the results from FALCON")
    parser.add_option("", "--falcon-suffix", default='/', help="Path of the results after the trait name")
    parser.add_option("", "--traits", action='append', default=[], help="list of traits to include")
    parser.add_option("", "--output", help="Prefix for all output files.")
    parser.add_option("", "--target-file", default="pegs1.wg.genes", help="Name of the TSV file to read for each trait")

    # Threshold options
    parser.add_option("", "--gene_pp_th", type='float', default=0.01,
                      help="keep records with Probability/Value higher than threshold")
    parser.add_option("", "--gene_p_th", type='float', default=1.0,
                      help="keep records with P-Value less than threshold")

    (options, _) = parser.parse_args()

    # Validate required base arguments
    if not options.falcon_prefix or not options.output:
        parser.error("Both --falcon-prefix and --output are required.")

    # Determine file type based on target_file extension
    target_file = options.target_file
    if target_file.endswith(".wg.genes"):
        file_type = "genes"
    elif target_file.endswith(".wg.variants"):
        file_type = "variants"
    elif target_file.endswith(".wg.v2g"):
        file_type = "v2g"
    else:
        parser.error("Unsupported target-file extension. Supported extensions are: *.wg.genes, *.wg.variants, *.wg.v2g")

    # Determine the list of traits
    if len(options.traits) > 0:
        list_of_traits = options.traits
    else:
        list_of_traits = sorted([d for d in os.listdir(options.falcon_prefix)
                                 if os.path.isdir(os.path.join(options.falcon_prefix, d)) and not d.startswith('.')])

    # Dynamic File Paths setup
    output_files = []
    file2_path = f"{options.output}.trait_sorted.json"
    output_files.append(file2_path)

    if file_type == "genes":
        file1_path = f"{options.output}.GENE_sorted.json"
        file3_path = f"{options.output}.region_sorted.json"
        output_files.extend([file1_path, file3_path])
    elif file_type == "variants":
        file1_path = f"{options.output}.RSID_sorted.json"
        file3_path = f"{options.output}.region_sorted.json"
        output_files.extend([file1_path, file3_path])
    elif file_type == "v2g":
        rsid_path = f"{options.output}.RSID_sorted.json"
        gene_path = f"{options.output}.GENE_sorted.json"
        output_files.extend([rsid_path, gene_path])

    db_path = f"{options.output}_temp_aggregation.db"

    # Remove existing files to prevent appending to old data
    for path in output_files + [db_path]:
        if os.path.exists(path):
            os.remove(path)

    # Initialize SQLite database connection
    conn = sqlite3.connect(db_path)
    table_name = "data_table"

    print('Phase 1: Processing and filtering data per trait into temp database...')
    total_start_time = time.time()
    rows_inserted = 0

    for trait in list_of_traits:
        start_time = time.time()
        trait_dir = os.path.join(options.falcon_prefix, trait)
        suffix = options.falcon_suffix.lstrip('/')
        tsv_path = os.path.join(trait_dir, suffix, options.target_file)

        if not os.path.exists(tsv_path):
            print(f"Warning: File not found for trait {trait} at '{tsv_path}'. Skipping.")
            continue

        try:
            df = pd.read_csv(tsv_path, sep='\t')

            # Apply Specific Logic per File Type
            if file_type == "genes":
                if 'PROBABILITY' in df.columns and 'P_VALUE' in df.columns:
                    df = df[(df['PROBABILITY'] > options.gene_pp_th) & (df['P_VALUE'] < options.gene_p_th)]
                else:
                    print(f"Warning: Expected columns missing in {trait}. Skipping filtering.")
            elif file_type == "variants":
                if 'PROBABILITY' in df.columns:
                    df = df[df['PROBABILITY'] > options.gene_pp_th]
                else:
                    print(f"Warning: 'PROBABILITY' missing in {trait}. Skipping filtering.")
            elif file_type == "v2g":
                if 'Value' in df.columns:
                    df = df[df['Value'] > options.gene_pp_th]
                else:
                    print(f"Warning: 'Value' missing in {trait}. Skipping filtering.")

            if df.empty:
                print(f"Trait {trait} skipped: No rows passed the thresholds.")
                continue

            # Append the trait column
            df['trait'] = trait

            # Write chunk to the SQLite database
            df.to_sql(table_name, conn, if_exists='append', index=False)
            rows_inserted += len(df)

            print(f"Trait: {trait} | Inserted {len(df)} rows | Time: {time.time() - start_time:.2f}s")

            # Force memory cleanup
            del df
            gc.collect()

        except Exception as e:
            print(f"An error occurred while processing {trait}: {e}")

    if rows_inserted == 0:
        print("No valid data was aggregated. Exiting.")
        conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)
        exit(1)

    print(f"\nPhase 1 Complete. {rows_inserted} total rows aggregated in {time.time() - total_start_time:.2f} seconds.")

    # Phase 2: Indexing the database to make sorting fast
    print("\nPhase 2: Building database indexes for rapid sorting...")
    cursor = conn.cursor()
    cursor.execute(f"CREATE INDEX idx_trait ON {table_name}(trait);")

    if file_type == "genes":
        cursor.execute(f"CREATE INDEX idx_gene ON {table_name}(GENE);")
        cursor.execute(f"CREATE INDEX idx_region ON {table_name}(trait, CHR, START);")
    elif file_type == "variants":
        cursor.execute(f"CREATE INDEX idx_rsid ON {table_name}(RSID);")
        cursor.execute(f"CREATE INDEX idx_region ON {table_name}(trait, CHR, POS);")
    elif file_type == "v2g":
        cursor.execute(f"CREATE INDEX idx_rsid ON {table_name}(rsID);")
        cursor.execute(f"CREATE INDEX idx_gene ON {table_name}(Gene);")

    conn.commit()

    # Helper function to query the DB and write to JSON lines
    def write_sorted_json(order_by_col, output_filepath):
        print(f"Generating sorted file: {output_filepath} (Ordered by {order_by_col})")
        export_start = time.time()
        query = f"SELECT * FROM {table_name} ORDER BY {order_by_col}"

        with open(output_filepath, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(pd.read_sql_query(query, conn, chunksize=50000)):
                chunk_str = chunk.to_json(orient='records', lines=True)
                if chunk_str:
                    # Prevent empty lines by only prefixing newlines after the first chunk
                    if i > 0:
                        f.write('\n')
                    f.write(chunk_str)

        print(f"Finished {output_filepath} in {time.time() - export_start:.2f} seconds.")

    # Phase 3: Export the sorted data
    print("\nPhase 3: Exporting to JSON files...")

    if file_type == "genes":
        write_sorted_json('GENE', file1_path)
        write_sorted_json('trait', file2_path)
        write_sorted_json('trait, CHR, START', file3_path)
    elif file_type == "variants":
        write_sorted_json('RSID', file1_path)
        write_sorted_json('trait', file2_path)
        write_sorted_json('trait, CHR, POS', file3_path)
    elif file_type == "v2g":
        write_sorted_json('rsID', rsid_path)
        write_sorted_json('Gene', gene_path)
        write_sorted_json('trait', file2_path)

    # Cleanup
    conn.close()
    if os.path.exists(db_path):
        os.remove(db_path)

    print("\nAll done! Temporary database cleaned up.")


if __name__ == '__main__':
    main()