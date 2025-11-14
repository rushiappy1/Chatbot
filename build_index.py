"""Build a FAISS index from a CSV file or via pyodbc SQL query.

Usage examples:
  python build_index.py --csv example.csv --text-column text
  python build_index.py --sql "SELECT id, text FROM docs" --conn "DRIVER={SQL Server};SERVER=.;DATABASE=db;UID=user;PWD=pwd"

This script writes:
  - data/index.faiss (FAISS index)
  - data/metadata.parquet (ids, texts)

It uses sentence-transformers 'all-MiniLM-L6-v2' for embeddings.
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def chunk_text(text, max_tokens=500, sep='\n'):
    # naive chunker by characters; adjust as needed
    if not isinstance(text, str):
        return []
    text = text.strip()
    if len(text) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_tokens
        chunks.append(text[start:end])
        start = end
    return chunks


def load_data_from_csv(path, text_column=None):
    df = pd.read_csv(path)
    out = []
    for idx, row in df.iterrows():
        # Create a text summary from multiple columns
        txt = f"Date: {row['Date']}\nEmployee: {row['EmployeeName']} (ID: {row['emp_id']})\n"
        txt += f"Vehicle: {row['vehicleNumber']}\nTarget: {row['Target']}\n"
        txt += f"Waste Collection: Mixed={row['mixed_waste']}, Segregated={row['segregate_waste']}\n"
        txt += f"Houses: Total={row['TotalHouseCount']}, Not Collected={row['Not_collected']}\n"
        txt += f"Duty: {row['duty_on_time']} to {row['duty_off_time']} ({row['working_time']})\n"
        txt += f"First Scan: {row['FirstHouseScan']}, Last Scan: {row['LastHouseScan']}"
        
        chunks = chunk_text(txt)
        for i, c in enumerate(chunks):
            out.append({'id': f"{idx}-{i}", 'text': c})
    return pd.DataFrame(out)


def load_data_from_sql(sql, conn_str, text_column='text', id_column='id'):
    import pyodbc
    import pandas as pd
    conn = pyodbc.connect(conn_str)
    df = pd.read_sql(sql, conn)
    out = []
    for _, row in df.iterrows():
        txt = row[text_column]
        idv = row[id_column]
        chunks = chunk_text(txt)
        for i, c in enumerate(chunks):
            out.append({'id': f"{idv}-{i}", 'text': c})
    return pd.DataFrame(out)


def build_index(df, model_name='all-MiniLM-L6-v2', out_dir='data', dim=384):
    from sentence_transformers import SentenceTransformer
    import faiss

    os.makedirs(out_dir, exist_ok=True)
    texts = df['text'].tolist()
    ids = df['id'].tolist()

    print(f"Encoding {len(texts)} text chunks with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype('float32')

    # normalize for IP/ cosine
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path = os.path.join(out_dir, 'index.faiss')
    faiss.write_index(index, index_path)

    meta_path = os.path.join(out_dir, 'metadata.parquet')
    df[['id','text']].to_parquet(meta_path, index=False)

    print(f"Wrote index to {index_path} and metadata to {meta_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', help='Path to CSV file (fallback)')
    p.add_argument('--text-column', default='text', help='Text column name in CSV')
    p.add_argument('--sql', help='SQL query to run via pyodbc')
    p.add_argument('--conn', help='pyodbc connection string')
    p.add_argument('--out', default='data', help='Output directory')
    args = p.parse_args()

    if not args.csv and not (args.sql and args.conn):
        print('Provide --csv or both --sql and --conn')
        sys.exit(1)

    if args.csv:
        df = load_data_from_csv(args.csv, args.text_column)
    else:
        df = load_data_from_sql(args.sql, args.conn)

    build_index(df, out_dir=args.out)


if __name__ == '__main__':
    main()
