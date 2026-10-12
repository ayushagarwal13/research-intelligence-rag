from unittest.mock import patch

from app.api.dependencies import get_db


def test_get_db_closes_session():
    mock_session = patch(
        "app.api.dependencies.SessionLocal"
    ).start()

    try:
        session = mock_session.return_value

        generator = get_db()

        yielded_session = next(generator)

        assert yielded_session is session
        mock_session.assert_called_once_with()

        try:
            next(generator)
        except StopIteration:
            pass

        session.close.assert_called_once_with()

    finally:
        patch.stopall()
