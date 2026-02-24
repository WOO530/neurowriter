"""LLM Client abstraction layer for NeuroWriter"""
from typing import Optional, Iterator
import logging
from abc import ABC, abstractmethod
from openai import OpenAI
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL)


class LLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        pass

    @abstractmethod
    def generate_streaming(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate text from a prompt with streaming"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI GPT-4o client implementation"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize OpenAI client

        Args:
            api_key: OpenAI API key. If None, uses config.OPENAI_API_KEY
            model: Model name. If None, uses config.OPENAI_MODEL
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.OPENAI_MODEL

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set. Please check your .env file or config.")

        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"OpenAI client initialized with model: {self.model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from a prompt

        Args:
            prompt: The user prompt
            system_prompt: System message to set context
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters for OpenAI API

        Returns:
            Generated text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            content = response.choices[0].message.content
            return content or ""
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def generate_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """Generate text from a prompt with streaming

        Args:
            prompt: The user prompt
            system_prompt: System message to set context
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters for OpenAI API

        Yields:
            Generated text chunks
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            with self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            ) as response:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error generating streaming text: {e}")
            raise


# Factory function to get LLM client
def get_llm_client(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> LLMClient:
    """Get an LLM client instance

    Args:
        provider: LLM provider ('openai' or 'claude')
        api_key: API key for the provider
        model: Model name

    Returns:
        LLMClient instance
    """
    if provider == "openai":
        return OpenAIClient(api_key=api_key, model=model)
    elif provider == "claude":
        # Future: implement Claude client
        raise NotImplementedError("Claude provider not yet implemented")
    else:
        raise ValueError(f"Unknown provider: {provider}")
