"""
Unified LLM Client supporting Anthropic Claude and Google Gemini.
Supports native structured outputs for guaranteed valid JSON responses.
"""
import json
import os
import math
import re
import time
from typing import Optional, Dict, Any, Tuple, List
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, response_schema: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """Generate a response from the LLM.
        
        Args:
            prompt: The input prompt
            response_schema: Optional JSON schema dict to enforce structured output.
                             When provided, the response is guaranteed to be valid JSON.
            **kwargs: Additional arguments
            
        Returns:
            Tuple containing raw response text and optional token logprob data.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM is available (API key present, etc.)."""
        pass


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("anthropic package not installed. Run: pip install anthropic")
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def generate(self, prompt: str, response_schema: Optional[Dict[str, Any]] = None, max_tokens: int = 2000, **kwargs) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        if not self.is_available():
            raise RuntimeError("Anthropic client not available")
        
        # Build message parameters
        message_params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        # Use structured outputs when schema is provided
        if response_schema:
            # Use the beta API with structured outputs
            message = self.client.beta.messages.create(
                **message_params,
                betas=["structured-outputs-2025-11-13"],
                output_format={
                    "type": "json_schema",
                    "json_schema": response_schema
                }
            )
        else:
            message = self.client.messages.create(**message_params)
        
        return message.content[0].text, None


class GoogleClient(BaseLLMClient):
    """Google Gemini client.

    Logprob support on the Gemini v1beta API is model-dependent. Known as of
    writing:
      - supported: gemini-2.0-flash, gemini-2.0-flash-lite,
                   gemini-1.5-flash, gemini-1.5-pro
      - NOT supported: gemini-2.5-flash, gemini-2.5-pro
    You can override the model with the GOOGLE_MODEL env var, e.g.
        export GOOGLE_MODEL=gemini-2.0-flash
    to get per-token confidence (and a real sequence_logprob signal in the
    composite-confidence score). When unsupported, we detect it once and
    stop sending the flag to avoid paying a double round-trip on every call.
    """

    # Models that we know do NOT support `response_logprobs` on v1beta.
    # We pre-seed the cache to avoid the double round-trip on the first call.
    _KNOWN_NO_LOGPROBS = {"gemini-2.5-flash", "gemini-2.5-pro",
                          "gemini-2.5-flash-preview", "gemini-2.5-pro-preview"}

    def __init__(self, model: Optional[str] = None, temperature: float = 0.1,
                 max_429_retries: Optional[int] = None,
                 initial_backoff: float = 2.0):
        if max_429_retries is None:
            try:
                max_429_retries = int(os.environ.get("GOOGLE_429_RETRIES", "3"))
            except ValueError:
                max_429_retries = 3
        # Default to gemini-2.0-flash because it supports response_logprobs,
        # which is critical for the composite-confidence / ambiguity signal.
        # Override with GOOGLE_MODEL env var if you want a different model.
        self.model = model or os.environ.get("GOOGLE_MODEL") or "gemini-2.0-flash"
        self.temperature = temperature
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.client = None
        self.max_429_retries = max_429_retries
        self.initial_backoff = initial_backoff
        # Tri-state capability cache: None = unknown, True/False once probed.
        self._logprobs_supported: Optional[bool] = (
            False if self.model in self._KNOWN_NO_LOGPROBS else None
        )
        self._schema_supported: Optional[bool] = None

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
            except ImportError:
                print("google-generativeai package not installed. Run: pip install google-generativeai")

    def is_available(self) -> bool:
        return self.client is not None

    @staticmethod
    def _extract_token_data(response) -> List[Dict[str, Any]]:
        token_data: List[Dict[str, Any]] = []
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return token_data
        candidate = candidates[0]
        logprobs_result = getattr(candidate, "logprobs_result", None)
        top_candidates = getattr(logprobs_result, "top_candidates", None) if logprobs_result else None
        if not top_candidates:
            return token_data
        for token_candidates in top_candidates:
            inner = getattr(token_candidates, "candidates", None)
            if not inner:
                continue
            top_candidate = inner[0]
            logprob = getattr(top_candidate, "log_probability", 0.0)
            token_data.append({
                "token": top_candidate.token,
                "confidence": math.exp(logprob),
            })
        return token_data

    def _build_config(self, response_schema: Optional[Dict[str, Any]],
                      with_logprobs: bool, with_schema: bool) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"temperature": self.temperature}
        if with_logprobs:
            cfg["response_logprobs"] = True
            cfg["logprobs"] = 1
        if with_schema and response_schema:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = response_schema
        return cfg

    @staticmethod
    def _is_rate_limit(error_str: str) -> bool:
        lower = error_str.lower()
        return "429" in error_str or "resource exhausted" in lower \
            or "quota" in lower or "rate" in lower

    @staticmethod
    def _suggested_backoff(error_str: str, fallback: float) -> float:
        """Extract retry_delay { seconds: N } hint from Google error, else fallback."""
        m = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", error_str)
        if m:
            return float(m.group(1))
        return fallback

    def _call_once(self, prompt: str, response_schema: Optional[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Single attempt. Handles logprobs/schema capability fallbacks but not 429s."""
        want_logprobs = self._logprobs_supported is not False
        want_schema = response_schema is not None and self._schema_supported is not False

        try:
            cfg = self._build_config(response_schema, want_logprobs, want_schema)
            response = self.client.generate_content(prompt, generation_config=cfg)
            token_data = self._extract_token_data(response) if want_logprobs else []
            if self._logprobs_supported is None:
                self._logprobs_supported = bool(token_data) or want_logprobs
            if want_schema and self._schema_supported is None:
                self._schema_supported = True
            return response.text, token_data
        except Exception as e:
            error_str = str(e)

            if "Logprobs is not enabled" in error_str or "response_logprobs" in error_str:
                if self._logprobs_supported is not False:
                    print(f"Logprobs not supported for model '{self.model}'. "
                          f"Disabling for session. Tip: "
                          f"export GOOGLE_MODEL=gemini-2.0-flash to enable them.")
                self._logprobs_supported = False
                cfg = self._build_config(response_schema, False, want_schema)
                response = self.client.generate_content(prompt, generation_config=cfg)
                return response.text, []

            if response_schema and ("response_schema" in error_str or "Unknown field" in error_str):
                if self._schema_supported is not False:
                    print(f"Structured output not supported for model '{self.model}', "
                          f"falling back to free-form generation.")
                self._schema_supported = False
                cfg = self._build_config(None, want_logprobs, False)
                response = self.client.generate_content(prompt, generation_config=cfg)
                token_data = self._extract_token_data(response) if want_logprobs else []
                return response.text, token_data

            raise

    def generate(self, prompt: str, response_schema: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        if not self.is_available():
            raise RuntimeError("Google client not available")

        backoff = self.initial_backoff
        last_err: Optional[Exception] = None
        for attempt in range(self.max_429_retries + 1):
            try:
                return self._call_once(prompt, response_schema)
            except Exception as e:  # noqa: BLE001
                error_str = str(e)
                if self._is_rate_limit(error_str) and attempt < self.max_429_retries:
                    wait = self._suggested_backoff(error_str, backoff)
                    print(f"[Gemini 429] rate-limited, retrying in {wait:.1f}s "
                          f"(attempt {attempt + 1}/{self.max_429_retries})")
                    time.sleep(wait)
                    backoff = min(backoff * 2, 60.0)
                    last_err = e
                    continue
                if self._is_rate_limit(error_str):
                    raise RuntimeError(
                        f"Rate limit exceeded after {self.max_429_retries} retries. "
                        f"Error: {error_str[:200]}"
                    ) from e
                print(f"GoogleClient generate Exception: {error_str}")
                raise

        # Unreachable unless we exhausted retries with only rate-limit errors.
        raise RuntimeError(
            f"Rate limit exceeded after {self.max_429_retries} retries. "
            f"Last error: {str(last_err)[:200] if last_err else 'unknown'}"
        )



def create_llm_client(provider: str = "auto", **kwargs) -> BaseLLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        provider: "anthropic", "google", or "auto" (tries in order)
        **kwargs: Additional arguments passed to the client
        
    Returns:
        An LLM client instance
    """
    if provider == "anthropic":
        client = AnthropicClient(**kwargs)
        if client.is_available():
            return client
        raise RuntimeError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
    
    if provider == "google":
        client = GoogleClient(**kwargs)
        if client.is_available():
            return client
        raise RuntimeError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
    
    if provider == "auto":
        # Try Anthropic first
        anthropic_client = AnthropicClient(**kwargs)
        if anthropic_client.is_available():
            print("Using Anthropic Claude")
            return anthropic_client
        
        # Try Google
        google_client = GoogleClient(**kwargs)
        if google_client.is_available():
            print("Using Google Gemini")
            return google_client
        
        raise RuntimeError(
            "CRITICAL EXCEPTION: No LLM API keys found.\n"
            "This pipeline requires either ANTHROPIC_API_KEY or GOOGLE_API_KEY "
            "to be set in the environment variables.\n"
            "Example: export GOOGLE_API_KEY='your-key-here'"
        )
    
    raise ValueError(f"Unknown provider: {provider}")
