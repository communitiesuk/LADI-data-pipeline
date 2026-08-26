"""
Stage 4: Upload embeddings into Azure Postgres with pgvector.

Reads a JSONL or CSV file of embedded docs and upserts each row into the
target table. Skips rows already present by URL. Logs progress every 100 rows.

Usage:
    python stages/04_upload.py --input data/embeddings_gpt51_50k.jsonl --table ladi_gpt51
    python stages/04_upload.py --input data/embeddings_combined_08072026.csv --table embedding_sample
"""
import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# Column length limits matching the Postgres schema
COL_LIMITS = {
    'url':       250,
    'title':     250,
    'authority': 100,
    'year':      10,
    'themes':    250,
    'doc_type':  100,
}


def get_connection():
    return psycopg2.connect(
        host=os.environ['LADI_DB_HOST'],
        port=int(os.environ.get('LADI_DB_PORT', '5432')),
        user=os.environ['LADI_DB_USER'],
        password=os.environ['LADI_DB_PASSWORD'],
        dbname=os.environ['LADI_DB_NAME'],
        sslmode='require',
    )


def truncate(value, limit):
    if value and isinstance(value, str) and len(value) > limit:
        return value[:limit]
    return value


def load_existing_urls(cur, table):
    cur.execute(f'SELECT url FROM {table}')
    return {row[0] for row in cur.fetchall()}


def upsert_row(cur, table, row):
    embedding = row['embedding']
    if isinstance(embedding, str):
        embedding = json.loads(embedding)
    vec_str = '[' + ','.join(str(v) for v in embedding) + ']'

    cur.execute(
        f"""
        INSERT INTO {table} (url, title, year, summary, themes, doc_type, authority, text, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (url) DO UPDATE SET
            title     = EXCLUDED.title,
            year      = EXCLUDED.year,
            summary   = EXCLUDED.summary,
            themes    = EXCLUDED.themes,
            doc_type  = EXCLUDED.doc_type,
            authority = EXCLUDED.authority,
            text      = EXCLUDED.text,
            embedding = EXCLUDED.embedding
        """,
        (
            truncate(row.get('url'), 250),
            truncate(row.get('title'), 250),
            truncate(str(row.get('year', '')), 10) if row.get('year') else None,
            row.get('summary'),
            truncate(str(row.get('themes', '')), 250),
            truncate(str(row.get('doc_type', '')), 100) if row.get('doc_type') else None,
            truncate(row.get('authority'), 100),
            row.get('text') or row.get('document_text'),
            vec_str,
        ),
    )


def main():
    p = argparse.ArgumentParser(description='Load embeddings CSV into Postgres')
    p.add_argument('--input', required=True, help='Path to embeddings CSV')
    p.add_argument('--table', default=os.environ.get('LADI_DB_TABLE', 'embedding_sample'), help='Target table name')
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')

    log.info(f'Loading {input_path} → {args.table}')
    if input_path.suffix == '.jsonl':
        rows = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        if 'embedding' in r:
                            rows.append(r)
                    except json.JSONDecodeError:
                        pass
        df = pd.DataFrame(rows)
        log.info(f'Read {len(df):,} rows from JSONL')
    else:
        df = pd.read_csv(input_path)
        log.info(f'Read {len(df):,} rows from CSV')

    conn = get_connection()
    cur = conn.cursor()

    # Check if url column has a unique constraint — needed for ON CONFLICT
    cur.execute("""
        SELECT COUNT(*) FROM information_schema.table_constraints tc
        JOIN information_schema.constraint_column_usage ccu
          ON tc.constraint_name = ccu.constraint_name
        WHERE tc.table_name = %s
          AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
          AND ccu.column_name = 'url'
    """, (args.table,))
    has_constraint = cur.fetchone()[0] > 0

    if not has_constraint:
        log.warning('No UNIQUE constraint on url column — adding one now')
        cur.execute(f'ALTER TABLE {args.table} ADD CONSTRAINT {args.table}_url_unique UNIQUE (url)')
        conn.commit()
        log.info('Unique constraint added')

    existing_urls = load_existing_urls(cur, args.table)
    log.info(f'Found {len(existing_urls):,} existing rows in {args.table}')

    new_rows = df[~df['url'].isin(existing_urls)]
    update_rows = df[df['url'].isin(existing_urls)]
    log.info(f'To insert: {len(new_rows):,} | To update: {len(update_rows):,} | Total: {len(df):,}')

    done = 0
    errors = 0
    for _, row in df.iterrows():
        try:
            upsert_row(cur, args.table, row)
            done += 1
            if done % 100 == 0:
                conn.commit()
                log.info(f'Progress: {done:,}/{len(df):,} upserted ({errors} errors)')
        except Exception as e:
            log.warning(f'Error on {row.get("url", "?")}: {e}')
            conn.rollback()
            errors += 1

    conn.commit()
    log.info(f'Done: {done:,} upserted, {errors} errors')
    cur.close()
    conn.close()


if __name__ == '__main__':
    main()
