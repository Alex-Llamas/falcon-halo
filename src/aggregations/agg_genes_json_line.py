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

    # Threshold options (updated to match your request)
    parser.add_option("", "--gene_pp_th", type='float', default=0.01,
                      help="keep genes with Probability higher than threshold")
    parser.add_option("", "--gene_p_th", type='float', default=1.0, help="keep genes with P-Value less than threshold")

    (options, _) = parser.parse_args()

    # Validate required base arguments
    if not options.falcon_prefix or not options.output:
        parser.error("Both --falcon-prefix and --output are required.")

    # Determine the list of traits
    if len(options.traits) > 0:
        list_of_traits = options.traits
    else:
        list_of_traits = sorted([d for d in os.listdir(options.falcon_prefix)
                                 if os.path.isdir(os.path.join(options.falcon_prefix, d)) and not d.startswith('.')])

    # File paths
    file1_path = options.output + ".gene_sorted.jsonl"
    file2_path = options.output + ".trait_sorted.jsonl"
    db_path = options.output + "_temp_aggregation.db"

    # Remove existing files to prevent appending to old data
    for path in [file1_path, file2_path, db_path]:
        if os.path.exists(path):
            os.remove(path)

    # Initialize SQLite database connection
    conn = sqlite3.connect(db_path)
    table_name = "genes_data"

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
            # Read only the necessary columns to save memory during read (optional but recommended)
            # If you want all columns, you can remove the `usecols` argument.
            df = pd.read_csv(tsv_path, sep='\t')

            # Apply Filters
            # Ensure columns exist before filtering to avoid KeyErrors
            if 'PROBABILITY' in df.columns and 'P_VALUE' in df.columns:
                df = df[(df['PROBABILITY'] > options.gene_pp_th) & (df['P_VALUE'] < options.gene_p_th)]
            else:
                print(f"Warning: Expected columns missing in {trait}. Skipping filtering.")

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
        os.remove(db_path)
        exit(1)

    print(f"\nPhase 1 Complete. {rows_inserted} total rows aggregated in {time.time() - total_start_time:.2f} seconds.")

    # Phase 2: Indexing the database to make sorting fast
    print("\nPhase 2: Building database indexes for rapid sorting...")
    cursor = conn.cursor()
    cursor.execute(f"CREATE INDEX idx_gene ON {table_name}(GENE);")
    cursor.execute(f"CREATE INDEX idx_trait ON {table_name}(trait);")
    conn.commit()

    # Helper function to query the DB and write to JSONL in chunks
    def write_sorted_jsonl(order_by_col, output_filepath):
        print(f"Generating sorted file: {output_filepath} (Ordered by {order_by_col})")
        export_start = time.time()

        # Using Pandas to read from SQLite in chunks to avoid memory spikes
        query = f"SELECT * FROM {table_name} ORDER BY {order_by_col}"

        # Open the file in write mode
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for chunk in pd.read_sql_query(query, conn, chunksize=50000):
                # Write the chunk to the file as JSON lines
                chunk.to_json(f, orient='records', lines=True)
                # Ensure a newline is present after the chunk (pandas .to_json doesn't append trailing newline)
                f.write('\n')

        print(f"Finished {output_filepath} in {time.time() - export_start:.2f} seconds.")

    # Phase 3: Export the sorted data
    print("\nPhase 3: Exporting to JSONL files...")
    write_sorted_jsonl('GENE', file1_path)
    write_sorted_jsonl('trait', file2_path)

    # Cleanup
    conn.close()
    if os.path.exists(db_path):
        os.remove(db_path)

    print("\nAll done! Temporary database cleaned up.")


if __name__ == '__main__':
    main()