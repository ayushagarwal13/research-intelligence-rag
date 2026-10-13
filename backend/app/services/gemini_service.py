import logging
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from app.services.llm_service import (
    LLMService,
    LLMServiceUnavailableError,
)


logger = logging.getLogger(__name__)

load_dotenv()


class GeminiService(LLMService):
    """
    Gemini implementation of the LLM service.

    Temporary server-side failures are retried with
    bounded exponential backoff.

    Provider failures are logged without logging API keys
    or prompt contents.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set."
            )

        if max_retries < 0:
            raise ValueError(
                "max_retries cannot be negative."
            )

        if retry_delay < 0:
            raise ValueError(
                "retry_delay cannot be negative."
            )

        self.client = genai.Client(api_key=api_key)

        self.model_name = (
            model_name
            or os.getenv(
                "GEMINI_MODEL",
                "gemini-2.5-flash",
            )
        )

        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(self, prompt: str) -> str:
        """Generate a response from Gemini with bounded retries."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        attempt = 0

        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )

                if not response.text:
                    raise ValueError(
                        "Gemini returned an empty response."
                    )

                return response.text.strip()

            except errors.ClientError as exc:
                logger.error(
                    "Gemini client error: model=%s status=%s",
                    self.model_name,
                    exc.code,
                )

                if exc.code == 429:
                    raise LLMServiceUnavailableError(
                        "The LLM service is temporarily unavailable "
                        "because the Gemini API rate limit or quota "
                        "has been exhausted. Please try again later."
                    ) from exc

                raise

            except errors.ServerError as exc:
                provider_message = str(
                    getattr(exc, "message", "")
                    or "No provider message supplied."
                )

                # Keep logs bounded and redact the API key defensively.
                provider_message = provider_message.replace(
                    api_key, "[REDACTED]"
                )[:500]

                logger.warning(
                    "Gemini server error: model=%s status=%s "
                    "attempt=%s max_retries=%s provider_message=%s",
                    self.model_name,
                    exc.code,
                    attempt + 1,
                    self.max_retries + 1,
                    provider_message,
                )

                if attempt >= self.max_retries:
                    raise LLMServiceUnavailableError(
                        "The LLM service is temporarily unavailable. "
                        "Please try again later."
                    ) from exc

                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)
                attempt += 1


