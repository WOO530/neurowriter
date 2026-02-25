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

    # Reasoning models that do not support temperature/top_p
    _REASONING_PREFIXES = ("gpt-5", "o3", "o4")

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

    def _is_reasoning_model(self) -> bool:
        """Check if current model is a reasoning model (no temperature/top_p support)"""
        return self.model.startswith(self._REASONING_PREFIXES)

    def _build_messages(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> list:
        """Build messages list with correct role for model type

        Reasoning models (gpt-5, o3, o4) use 'developer' role;
        standard models use 'system' role.
        """
        messages = []
        if system_prompt:
            role = "developer" if self._is_reasoning_model() else "system"
            messages.append({"role": role, "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    # Reasoning effort → (token multiplier, min tokens)
    _EFFORT_TIERS = {
        "none":   (1, 2048),     # no reasoning — output only
        "low":    (2, 4096),
        "medium": (3, 8192),
        "high":   (5, 16384),
        "xhigh":  (8, 32768),   # GPT-5.2 max-quality tier
    }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        reasoning_effort: str = "medium",
        **kwargs
    ) -> str:
        """Generate text from a prompt

        Args:
            prompt: The user prompt
            system_prompt: System message to set context
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            reasoning_effort: "none"|"low"|"medium"|"high"|"xhigh" — controls
                reasoning model token budget and effort. Ignored for
                non-reasoning models.
            **kwargs: Additional parameters for OpenAI API

        Returns:
            Generated text
        """
        messages = self._build_messages(prompt, system_prompt)

        try:
            params = {
                "model": self.model,
                "messages": messages,
                **kwargs,
            }

            if self._is_reasoning_model():
                multiplier, min_tokens = self._EFFORT_TIERS.get(
                    reasoning_effort, self._EFFORT_TIERS["medium"]
                )
                if max_tokens is not None:
                    params["max_completion_tokens"] = max(max_tokens * multiplier, min_tokens)

                # Pass reasoning_effort to the API (required for GPT-5.2
                # whose default is "none"; harmless for GPT-5 default "medium")
                params["reasoning_effort"] = reasoning_effort

                # temperature is only allowed when effort="none"
                if reasoning_effort == "none":
                    params["temperature"] = temperature
                else:
                    params.pop("temperature", None)
                params.pop("top_p", None)
            else:
                if max_tokens is not None:
                    params["max_tokens"] = max_tokens
                params["temperature"] = temperature

            logger.debug(
                f"API call: model={self.model}, reasoning={self._is_reasoning_model()}, "
                f"effort={reasoning_effort}, msgs={len(messages)}"
            )

            response = self.client.chat.completions.create(**params)

            msg = response.choices[0].message
            content = msg.content
            finish_reason = response.choices[0].finish_reason

            # If model refused the request, surface it
            refusal = getattr(msg, "refusal", None)
            if refusal:
                raise ValueError(f"Model refused request: {refusal}")

            # If content is empty, raise with diagnostic info instead of silent ""
            if not content:
                diag = (
                    f"Empty response from {self.model} | "
                    f"finish_reason={finish_reason} | "
                    f"usage={response.usage} | "
                    f"role={messages[0]['role'] if messages else '?'}"
                )
                logger.error(diag)
                raise ValueError(diag)

            return content
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def generate_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        reasoning_effort: str = "medium",
        **kwargs
    ) -> Iterator[str]:
        """Generate text from a prompt with streaming

        Args:
            prompt: The user prompt
            system_prompt: System message to set context
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            reasoning_effort: "none"|"low"|"medium"|"high"|"xhigh" — ignored for non-reasoning models
            **kwargs: Additional parameters for OpenAI API

        Yields:
            Generated text chunks
        """
        messages = self._build_messages(prompt, system_prompt)

        try:
            params = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                **kwargs,
            }

            if self._is_reasoning_model():
                multiplier, min_tokens = self._EFFORT_TIERS.get(
                    reasoning_effort, self._EFFORT_TIERS["medium"]
                )
                if max_tokens is not None:
                    params["max_completion_tokens"] = max(max_tokens * multiplier, min_tokens)

                params["reasoning_effort"] = reasoning_effort

                if reasoning_effort == "none":
                    params["temperature"] = temperature
                else:
                    params.pop("temperature", None)
                params.pop("top_p", None)
            else:
                if max_tokens is not None:
                    params["max_tokens"] = max_tokens
                params["temperature"] = temperature

            with self.client.chat.completions.create(**params) as response:
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
