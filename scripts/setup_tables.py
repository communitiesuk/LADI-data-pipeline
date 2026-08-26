"""
Create the three LADI model-comparison tables in Postgres.

Creates ladi_gpt51, ladi_gpt5mini, ladi_gpt5nano — same schema as
embedding_sample. Safe to re-run; uses IF NOT EXISTS.

Usage:
    python scripts/setup_tables.py
    python scripts/setup_tables.py --drop  # drop and recreate (destructive!)
"""
import argparse
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

TABLES = ['ladi_gpt51', 'ladi_gpt5mini', 'ladi_gpt5nano']

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    id        SERIAL PRIMARY KEY,
    url       VARCHAR(500) UNIQUE NOT NULL,
    title     VARCHAR(500),
    year      VARCHAR(10),
    summary   TEXT,
    themes    VARCHAR(500),
    doc_type  VARCHAR(100),
    authority VARCHAR(200),
    text      TEXT,
    embedding vector(3072)
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS {table}_authority_idx ON {table} (authority);
"""


def get_connection():
    return psycopg2.connect(
        host=os.environ['LADI_DB_HOST'],
        port=int(os.environ.get('LADI_DB_PORT', '5432')),
        user=os.environ['LADI_DB_USER'],
        password=os.environ['LADI_DB_PASSWORD'],
        dbname=os.environ['LADI_DB_NAME'],
        sslmode='require',
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--drop', action='store_true',
                   help='Drop and recreate tables (destructive — all data lost)')
    args = p.parse_args()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    for table in TABLES:
        if args.drop:
            print(f"Dropping {table} ...")
            cur.execute(f"DROP TABLE IF EXISTS {table};")

        print(f"Creating {table} ...")
        cur.execute(CREATE_SQL.format(table=table))
        cur.execute(CREATE_INDEX_SQL.format(table=table))

    conn.commit()

    # Verify
    for table in TABLES:
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        count = cur.fetchone()[0]
        print(f"  {table}: {count:,} rows")

    cur.close()
    conn.close()
    print("Done.")


if __name__ == '__main__':
    main()
