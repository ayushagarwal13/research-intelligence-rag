from abc import ABC, abstractmethod


class LLMServiceUnavailableError(RuntimeError):
    """
    Raised when the configured LLM provider is temporarily
    unavailable or its quota/rate limit has been exhausted.
    """


class LLMService(ABC):
    """
    Abstract interface for large language model services.

    Concrete implementations such as GeminiService
    must implement the generate() method.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a response from the supplied prompt.

        Args:
            prompt: Prompt sent to the language model.

        Returns:
            Generated text response.
        """
        raise NotImplementedError
