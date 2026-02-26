"""LLM Client abstraction layer for NeuroWriter"""
from typing import Optional, Iterator
import logging
from abc import ABC, abstractmethod
from openai import OpenAI, AzureOpenAI
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

    # Reasoning effort → reasoning token budget (added ON TOP of output tokens)
    # max_completion_tokens = max_tokens (output) + reasoning budget
    _REASONING_BUDGETS = {
        "none":   0,        # no reasoning — output only
        "low":    8192,
        "medium": 16384,
        "high":   32768,
        "xhigh":  65536,    # GPT-5.2 max-quality tier
    }

    # Effort downgrade map for retry on reasoning token overflow
    _EFFORT_DOWNGRADE = {
        "xhigh": "high",
        "high":  "medium",
        "medium": "low",
        "low":   "none",
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
                reasoning_budget = self._REASONING_BUDGETS.get(
                    reasoning_effort, self._REASONING_BUDGETS["medium"]
                )
                if max_tokens is not None:
                    # max_completion_tokens = output + reasoning budget
                    params["max_completion_tokens"] = max_tokens + reasoning_budget
                elif reasoning_budget > 0:
                    # No explicit max_tokens: set a reasonable total budget
                    params["max_completion_tokens"] = 4096 + reasoning_budget

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
                f"effort={reasoning_effort}, "
                f"max_completion_tokens={params.get('max_completion_tokens', params.get('max_tokens', 'default'))}, "
                f"msgs={len(messages)}"
            )

            response = self.client.chat.completions.create(**params)

            msg = response.choices[0].message
            content = msg.content
            finish_reason = response.choices[0].finish_reason

            # If model refused the request, surface it
            refusal = getattr(msg, "refusal", None)
            if refusal:
                raise ValueError(f"Model refused request: {refusal}")

            # Empty response — likely reasoning consumed all tokens
            if not content:
                if finish_reason == "length" and self._is_reasoning_model():
                    lower_effort = self._EFFORT_DOWNGRADE.get(reasoning_effort)
                    if lower_effort is not None:
                        logger.warning(
                            f"Reasoning overflow: effort={reasoning_effort} used all "
                            f"{response.usage.completion_tokens} tokens. "
                            f"Retrying with effort={lower_effort}"
                        )
                        return self.generate(
                            prompt=prompt, system_prompt=system_prompt,
                            temperature=temperature, max_tokens=max_tokens,
                            reasoning_effort=lower_effort, **kwargs
                        )
                # No retry possible or different failure — raise
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
                reasoning_budget = self._REASONING_BUDGETS.get(
                    reasoning_effort, self._REASONING_BUDGETS["medium"]
                )
                if max_tokens is not None:
                    params["max_completion_tokens"] = max_tokens + reasoning_budget
                elif reasoning_budget > 0:
                    params["max_completion_tokens"] = 4096 + reasoning_budget

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


class AzureOpenAIClient(OpenAIClient):
    """Azure OpenAI client — inherits all generate/streaming/reasoning logic
    from OpenAIClient, using AzureOpenAI SDK client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,        # Azure deployment name
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        base_model: Optional[str] = None,   # 실제 모델명 (reasoning 감지용)
    ):
        # Do NOT call super().__init__() — it creates an OpenAI() client
        self.api_key = api_key or config.AZURE_OPENAI_API_KEY
        self.model = model or config.AZURE_OPENAI_DEPLOYMENT

        azure_endpoint = azure_endpoint or config.AZURE_OPENAI_ENDPOINT
        api_version = api_version or config.AZURE_OPENAI_API_VERSION

        if not self.api_key:
            raise ValueError("Azure OpenAI API key not set.")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint not set.")
        if not self.model:
            raise ValueError("Azure OpenAI deployment name not set.")

        self._base_model = base_model or ""

        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        logger.info(
            f"Azure OpenAI client initialized: deployment={self.model}, "
            f"base_model={self._base_model or '(not set)'}, endpoint={azure_endpoint}"
        )

    def _is_reasoning_model(self) -> bool:
        """Check if current model is a reasoning model.

        Uses base_model (actual model name) if set, otherwise falls back
        to deployment name. This handles cases where Azure deployment names
        are arbitrary (e.g. 'my-company-gpt51-deployment').
        """
        check_name = self._base_model or self.model
        return check_name.startswith(self._REASONING_PREFIXES)


# Factory function to get LLM client
def get_llm_client(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    base_model: Optional[str] = None,
) -> LLMClient:
    """Get an LLM client instance

    Args:
        provider: LLM provider ('openai', 'azure_openai', or 'claude')
        api_key: API key for the provider
        model: Model name (or Azure deployment name)
        azure_endpoint: Azure OpenAI endpoint URL
        api_version: Azure OpenAI API version
        base_model: Actual model name for reasoning detection (Azure only)

    Returns:
        LLMClient instance
    """
    if provider == "openai":
        return OpenAIClient(api_key=api_key, model=model)
    elif provider == "azure_openai":
        return AzureOpenAIClient(
            api_key=api_key,
            model=model,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            base_model=base_model,
        )
    elif provider == "claude":
        # Future: implement Claude client
        raise NotImplementedError("Claude provider not yet implemented")
    else:
        raise ValueError(f"Unknown provider: {provider}")
