import pyarrow.parquet as pq
import pandas as pd
import sys

def inspect(path, rows=0):
    print(f"\n--- Inspecting: {path} ---")
    try:
        table = pq.read_table(path)
        print("Schema:")
        for field in table.schema:
            print(f"{field.name}: {field.type}")
        if rows > 0:
            df = table.to_pandas().head(rows)
            print(f"\nFirst {rows} rows:")
            print(df.to_string())
    except Exception as e:
        print(f"Error reading {path}: {e}")

# Dataset 01
inspect("data/train_dataset_01/data/chunk-000/file-000.parquet", rows=1)
inspect("data/train_dataset_01/meta/episodes/chunk-000/file-000.parquet", rows=3)

# Dataset 02
inspect("data/train_dataset_02/data/chunk-000/file-000.parquet", rows=0)
inspect("data/train_dataset_02/meta/episodes/chunk-000/file-000.parquet", rows=0)
