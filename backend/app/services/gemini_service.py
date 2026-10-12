import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

from app.services.llm_service import (
    LLMService,
    LLMServiceUnavailableError,
)


load_dotenv()


class GeminiService(LLMService):
    """
    Gemini implementation of the LLM service.

    Temporary server-side failures are retried with
    bounded exponential backoff.

    Rate-limit and quota failures are surfaced as a
    service-unavailable error after the SDK has exhausted
    its own transient retry behavior.
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

        self.client = genai.Client(
            api_key=api_key
        )

        self.model_name = (
            model_name
            or os.getenv(
                "GEMINI_MODEL",
                "gemini-2.5-flash",
            )
        )

        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a response from Gemini.

        Temporary 5xx server errors are retried using
        bounded exponential backoff.

        Gemini 429 quota/rate-limit failures are not retried
        by this application because the Google GenAI SDK already
        performs transient retry handling. Once those retries
        are exhausted, the error is converted into a controlled
        service-unavailable exception.
        """

        if not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        attempt = 0

        while True:
            try:
                response = (
                    self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                    )
                )

                if not response.text:
                    raise ValueError(
                        "Gemini returned an empty response."
                    )

                return response.text.strip()

            except errors.ClientError as exc:
                if exc.code == 429:
                    raise LLMServiceUnavailableError(
                        "The LLM service is temporarily unavailable "
                        "because the Gemini API rate limit or quota "
                        "has been exhausted. Please try again later."
                    ) from exc

                raise

            except errors.ServerError as exc:
                if attempt >= self.max_retries:
                    raise LLMServiceUnavailableError(
                        "The LLM service is temporarily unavailable. "
                        "Please try again later."
                    ) from exc

                delay = (
                    self.retry_delay
                    * (2 ** attempt)
                )

                time.sleep(delay)
                attempt += 1
