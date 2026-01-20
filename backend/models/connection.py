"""
Database Connection Manager

Handles database connections and session management.
"""

import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

from .database import Base

load_dotenv()

logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """Get database URL from environment."""
    # Check for Vercel Postgres URL first (POSTGRES_URL), then DATABASE_URL
    url = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
    if not url:
        # Default to SQLite for development
        return "sqlite:///./cbb_betting.db"

    # Vercel Postgres uses postgres:// but SQLAlchemy requires postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    return url


# Create engine with appropriate settings based on database type
_db_url = get_database_url()
_engine_kwargs = {
    "echo": os.getenv("DEBUG", "false").lower() == "true",
}

# SQLite doesn't support pool_pre_ping
if not _db_url.startswith("sqlite"):
    _engine_kwargs["pool_pre_ping"] = True
    _engine_kwargs["pool_size"] = 5
    _engine_kwargs["max_overflow"] = 10

engine = create_engine(_db_url, **_engine_kwargs)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def drop_db():
    """Drop all database tables (use with caution!)."""
    Base.metadata.drop_all(bind=engine)
    logger.info("Database tables dropped")


@contextmanager
def get_session() -> Session:
    """Get a database session with automatic cleanup."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """Dependency for FastAPI endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
