#!/usr/bin/env python
"""
Script to migrate all tables from a local SQLite database (crib_cache.db)
to a remote PostgreSQL database.

Usage:
    python scripts/migrate_sqlite_to_postgres.py

Requirements:
    - Install dependencies: pip install sqlalchemy psycopg2-binary
    - Place crib_cache.db in the project root or update the path below.

Environment:
    - This script should be run from the root of crib_ai_trainer2.
"""
import os
import logging
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.engine import reflection
from sqlalchemy.schema import CreateTable
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths and connection strings
SQLITE_DB_PATH = os.path.abspath("crib_cache.db")
POSTGRES_URL = "need to lookup"


def migrate_sqlite_to_postgres(sqlite_path: str, postgres_url: str):
    """Migrate all tables and data from SQLite to PostgreSQL."""
    # Connect to SQLite
    sqlite_engine = create_engine(f"sqlite:///{sqlite_path}")
    # Connect to PostgreSQL
    postgres_engine = create_engine(postgres_url)

    sqlite_metadata = MetaData()
    sqlite_metadata.reflect(bind=sqlite_engine)

    # Reflect tables from SQLite
    inspector = reflection.Inspector.from_engine(sqlite_engine)
    table_names = inspector.get_table_names()
    logger.info(f"Found tables in SQLite: {table_names}")

    for table_name in table_names:
        logger.info(f"Migrating table: {table_name}")
        # Reflect table from SQLite
        table = Table(table_name, sqlite_metadata, autoload_with=sqlite_engine)
        # Recreate table in PostgreSQL
        pg_metadata = MetaData()
        pg_table = Table(table_name, pg_metadata)
        for col in table.columns:
            pg_table.append_column(col.copy())
        try:
            pg_table.drop(bind=postgres_engine, checkfirst=True)
            pg_table.create(bind=postgres_engine)
            logger.info(f"Created table {table_name} in PostgreSQL.")
        except SQLAlchemyError as e:
            logger.error(f"Error creating table {table_name}: {e}")
            continue
        # Copy data
        conn_sqlite = sqlite_engine.connect()
        conn_pg = postgres_engine.connect()
        try:
            rows = conn_sqlite.execute(table.select()).fetchall()
            if rows:
                # Use row._mapping for SQLAlchemy 1.4+ compatibility
                conn_pg.execute(pg_table.insert(), [dict(row._mapping) for row in rows])
                logger.info(f"Inserted {len(rows)} rows into {table_name}.")
            else:
                logger.info(f"No data to insert for {table_name}.")
        except SQLAlchemyError as e:
            logger.error(f"Error copying data for {table_name}: {e}")
        finally:
            conn_sqlite.close()
            conn_pg.close()

    logger.info("Migration complete.")


if __name__ == "__main__":
    migrate_sqlite_to_postgres(SQLITE_DB_PATH, POSTGRES_URL)
