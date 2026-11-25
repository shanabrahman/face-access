"""Database backend selector.

Set DATABASE_URL to use Supabase PostgreSQL. Without it, local development
continues to use the original SQLite implementation.
"""
import os

if os.environ.get("DATABASE_URL"):
    from database_postgres import *  # noqa: F401,F403
else:
    from database_sqlite import *  # noqa: F401,F403
