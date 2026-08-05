"""
One-off script to normalise domain-name authority entries to proper council names.

Some rows in the crawl had website domains as the Authority Name rather than
the proper council name. This script fixes them in Postgres and in local
JSONL/CSV files.

Usage:
    python fix_authority_names.py
    python fix_authority_names.py --dry-run
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

AUTHORITY_LOOKUP = {
    "adur-worthing.gov.uk":             "Adur District Council",
    "arun.gov.uk":                       "Arun District Council",
    "bathnes.gov.uk":                    "Bath and North East Somerset Council",
    "bedford.gov.uk":                    "Bedford Borough Council",
    "blaby.gov.uk":                      "Blaby District Council",
    "glostext.gloucestershire.gov.uk":   "Gloucestershire County Council",
    "gloucester.moderngov.co.uk":        "Gloucester City Council",
    "sevenoaks.moderngov.co.uk":         "Sevenoaks District Council",
    "southribble.moderngov.co.uk":       "South Ribble Borough Council",
    "westminster.gov.uk":                "City of Westminster",
    "worcestershire.gov.uk":             "Worcestershire County Council",
    "york.gov.uk":                       "City of York Council",
}


def fix_postgres(dry_run=False):
    conn = psycopg2.connect(
        host=os.environ['LADI_DB_HOST'],
        port=int(os.environ.get('LADI_DB_PORT', '5432')),
        user=os.environ['LADI_DB_USER'],
        password=os.environ['LADI_DB_PASSWORD'],
        dbname=os.environ['LADI_DB_NAME'],
        sslmode='require',
    )
    cur = conn.cursor()
    total_updated = 0
    for domain, proper_name in AUTHORITY_LOOKUP.items():
        cur.execute(
            "SELECT COUNT(*) FROM embedding_sample WHERE authority = %s",
            (domain,)
        )
        count = cur.fetchone()[0]
        if count == 0:
            continue
        print(f'  {domain!r} → {proper_name!r}  ({count} docs)')
        if not dry_run:
            cur.execute(
                "UPDATE embedding_sample SET authority = %s WHERE authority = %s",
                (proper_name, domain)
            )
            total_updated += count
    if not dry_run:
        conn.commit()
        print(f'\nPostgres: {total_updated} rows updated')
    else:
        print('\n(dry run — no changes made)')
    cur.close()
    conn.close()


def fix_jsonl(path: Path, dry_run=False):
    if not path.exists():
        print(f'  {path} not found, skipping')
        return
    records = []
    fixed = 0
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                old = rec.get('authority', '')
                if old in AUTHORITY_LOOKUP:
                    rec['authority'] = AUTHORITY_LOOKUP[old]
                    fixed += 1
                records.append(rec)
            except Exception:
                pass
    print(f'  {path}: {fixed} records fixed')
    if not dry_run:
        with open(path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec) + '\n')


def fix_csv(path: Path, dry_run=False):
    if not path.exists():
        print(f'  {path} not found, skipping')
        return
    df = pd.read_csv(path)
    fixed = df['authority'].isin(AUTHORITY_LOOKUP).sum()
    df['authority'] = df['authority'].map(lambda x: AUTHORITY_LOOKUP.get(x, x))
    print(f'  {path}: {fixed} rows fixed')
    if not dry_run:
        df.to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true', help='Show what would change without making changes')
    args = p.parse_args()

    print('=== Fixing Postgres ===')
    fix_postgres(dry_run=args.dry_run)

    print('\n=== Fixing local JSONL files ===')
    fix_jsonl(Path('data/embeddings_combined.jsonl'), dry_run=args.dry_run)
    fix_jsonl(Path('data/embeddings_10k.jsonl'), dry_run=args.dry_run)
    fix_jsonl(Path('data/embeddings_sample.jsonl'), dry_run=args.dry_run)

    print('\n=== Fixing local CSV files ===')
    fix_csv(Path('data/embeddings_combined_08072026.csv'), dry_run=args.dry_run)

    print('\nDone.')


if __name__ == '__main__':
    main()
