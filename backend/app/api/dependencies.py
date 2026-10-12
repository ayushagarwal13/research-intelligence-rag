from collections.abc import Generator

from fastapi import Request

from app.database.session import SessionLocal
from app.services.query_service import QueryService


def get_db() -> Generator:
    """
    Provide a database session for the duration
    of an API request.
    """
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def get_query_service(
    request: Request,
) -> QueryService:
    """
    Lazily create and reuse the query service from
    the shared application runtime.
    """
    container = getattr(
        request.app.state,
        "container",
        None,
    )

    if container is None:
        from app.services.application_container import (
            ApplicationContainer,
        )

        container = ApplicationContainer()
        request.app.state.container = container

    return container.query_service
