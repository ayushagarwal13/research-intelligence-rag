
import json
import logging
import os
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

from app.services.llm_service import (
    LLMService,
    LLMServiceUnavailableError,
)

logger = logging.getLogger(__name__)

load_dotenv()


class GroqService(LLMService):
    """Generate responses through the Groq OpenAI-compatible API."""

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        model_name: str | None = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        timeout: float = 45.0,
    ):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set."
            )

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        if retry_delay < 0:
            raise ValueError("retry_delay cannot be negative.")

        if timeout <= 0:
            raise ValueError("timeout must be positive.")

        self.api_key = api_key
        self.model_name = (
            model_name
            or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """Generate text with bounded retries for transient failures."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": 2048,
            "temperature": 0,
        }

        request = urllib.request.Request(
            self.API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ResearchIntelligenceRAG/1.0",
            },
            method="POST",
        )

        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout,
                ) as response:
                    result = json.loads(
                        response.read().decode("utf-8")
                    )

                choices = result.get("choices") or []
                if not choices:
                    raise ValueError(
                        "Groq returned no completion choices."
                    )

                message = choices[0].get("message") or {}
                content = message.get("content")

                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        "Groq returned an empty response."
                    )

                return content.strip()

            except urllib.error.HTTPError as exc:
                status = exc.code
                try:
                    # Read the provider error without logging secrets.
                    error_body = exc.read().decode(
                        "utf-8", errors="replace"
                    )
                    error_data = json.loads(error_body)
                    provider_message = (
                        error_data.get("error", {}).get("message", "")
                    )
                except Exception:
                    provider_message = ""

                logger.warning(
                    "Groq API error: model=%s status=%s attempt=%s",
                    self.model_name,
                    status,
                    attempt + 1,
                )

                retryable = status == 429 or status >= 500

                if retryable and attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue

                if retryable:
                    raise LLMServiceUnavailableError(
                        "The Groq service is temporarily unavailable "
                        "or its rate limit has been reached. Please try again."
                    ) from exc

                raise RuntimeError(
                    f"Groq API request failed with HTTP {status}: "
                    f"{provider_message[:300]}"
                ) from exc

            except (TimeoutError, urllib.error.URLError) as exc:
                logger.warning(
                    "Groq connection error: model=%s attempt=%s",
                    self.model_name,
                    attempt + 1,
                )

                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue

                raise LLMServiceUnavailableError(
                    "Could not connect to the Groq service. Please try again."
                ) from exc

        raise LLMServiceUnavailableError(
            "The Groq service is temporarily unavailable."
        )
