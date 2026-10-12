from app.main import app
from app.database.base import Base
from app.api.dependencies import get_db
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from fastapi.testclient import TestClient
from fastapi import FastAPI
import pytest
from collections.abc import Generator



@pytest.fixture
def test_database():
    """Create an isolated in-memory SQLite database for one test."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    Base.metadata.create_all(bind=engine)

    yield session_factory

    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def api_client(
    test_database,
):
    """Provide a TestClient connected to the test database."""
    def override_get_db() -> Generator[Session, None, None]:
        db = test_database()

        try:
            yield db
        finally:
            db.close()

    previous_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)

