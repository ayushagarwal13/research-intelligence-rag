import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "research_intelligence.db"
)


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DEFAULT_DATABASE_PATH.as_posix()}",
)


def normalize_database_url(database_url: str) -> str:
    """
    Normalize PostgreSQL connection URLs so that
    SQLAlchemy explicitly uses the psycopg 3 driver.

    SQLite URLs are returned unchanged.
    """

    if database_url.startswith("postgres://"):
        return database_url.replace(
            "postgres://",
            "postgresql+psycopg://",
            1,
        )

    if database_url.startswith("postgresql://"):
        return database_url.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )

    return database_url


DATABASE_URL = normalize_database_url(
    DATABASE_URL
)


if DATABASE_URL.startswith("sqlite"):
    connect_args = {
        "check_same_thread": False,
    }
else:
    connect_args = {}


engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def get_session():
    """
    Create a database session.
    """

    return SessionLocal()
