"""
Unified LLM Client supporting Anthropic Claude, Google Gemini, and local
Ollama models.

- Anthropic / Google: cloud, structured outputs supported.
- Ollama: local, native JSON-format mode, no API key, no rate limits.
  Best for fast iteration on the ambiguity benchmark when cloud quotas are
  exhausted. Recommended models: qwen2.5-coder:7b, llama3.1:8b, granite3-dense:8b.
"""
import json
import os
import math
import re
import time
import urllib.error
import urllib.request
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


class OllamaClient(BaseLLMClient):
    """
    Local LLM client backed by Ollama (https://ollama.com).

    No API key, no rate limits. Designed for fast iteration on the ambiguity
    benchmark. Uses Ollama's native /api/chat endpoint with JSON-mode for
    structured outputs (the same JSON Schema we send to Anthropic / Google
    is forwarded as Ollama's `format` parameter, which Ollama 0.5+ honours).

    Token-level log probabilities are NOT exposed by Ollama's standard API,
    so this client returns an empty token list. The pipeline's composite
    confidence still works via execution entropy + self-reported confidence
    + result clustering -- the logprob term simply contributes a neutral 0.

    Configuration (env vars):
      OLLAMA_HOST   - base URL (default http://localhost:11434)
      OLLAMA_MODEL  - model tag (default qwen2.5-coder:7b)
    """

    DEFAULT_MODEL = "qwen2.5-coder:7b"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(self, model: Optional[str] = None,
                 host: Optional[str] = None,
                 temperature: float = 0.1,
                 timeout: float = 180.0,
                 num_ctx: Optional[int] = None,
                 keep_alive: str = "30m"):
        self.model = model or os.environ.get("OLLAMA_MODEL") or self.DEFAULT_MODEL
        self.host = (host or os.environ.get("OLLAMA_HOST") or self.DEFAULT_HOST).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        # SQL prompts can be long; bump default context to 8192.
        self.num_ctx = num_ctx or int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
        self.keep_alive = keep_alive
        # Cached availability check to avoid hitting the server every call.
        self._available: Optional[bool] = None
        self._available_models: Optional[List[str]] = None
        # Tri-state: structured-output (`format=<schema>`) is Ollama 0.5+;
        # if the server rejects it we fall back to plain JSON mode.
        self._schema_supported: Optional[bool] = None

    def _http_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_get(self, path: str) -> Dict[str, Any]:
        with urllib.request.urlopen(self.host + path, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _probe(self) -> bool:
        try:
            data = self._http_get("/api/tags")
            self._available_models = [m.get("name", "") for m in data.get("models", [])]
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            return False
        except Exception:  # noqa: BLE001
            return False

    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._probe()
        return self._available

    def list_models(self) -> List[str]:
        if self._available_models is None:
            self._probe()
        return list(self._available_models or [])

    def _has_model(self) -> bool:
        models = self.list_models()
        if not models:
            return False
        # Require an exact tag match. Different sizes (e.g. :7b vs :14b) are
        # different models to Ollama; treating them as interchangeable caused
        # a false "available" pre-check and then HTTP 404 on /api/chat.
        return self.model in models

    def generate(self, prompt: str, response_schema: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        if not self.is_available():
            raise RuntimeError(
                f"Ollama not reachable at {self.host}. "
                f"Install from https://ollama.com and run `ollama serve`."
            )
        if not self._has_model():
            available = ", ".join(self.list_models()) or "<none>"
            raise RuntimeError(
                f"Ollama model '{self.model}' is not pulled locally.\n"
                f"Run: ollama pull {self.model}\n"
                f"Currently available: {available}"
            )

        # Build options: temperature + larger context for SQL prompts.
        options: Dict[str, Any] = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
        }

        # Structured outputs:
        #   - Ollama 0.5+ accepts a JSON-Schema dict as `format`.
        #   - Older versions only accept the literal string "json".
        # We try schema first, fall back on rejection, and remember the result.
        want_schema = response_schema is not None and self._schema_supported is not False
        format_value: Any
        if want_schema:
            format_value = response_schema
        elif response_schema is not None:
            format_value = "json"
        else:
            format_value = None

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": options,
            "keep_alive": self.keep_alive,
        }
        if format_value is not None:
            payload["format"] = format_value

        try:
            resp = self._http_post("/api/chat", payload)
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                pass
            # Schema unsupported: retry once with plain JSON mode and remember.
            if want_schema and ("format" in err_body.lower() or e.code in (400, 422)):
                if self._schema_supported is not False:
                    print(f"[Ollama] structured-output schema rejected by '{self.model}', "
                          f"falling back to JSON mode. Detail: {err_body[:200]}")
                self._schema_supported = False
                payload["format"] = "json" if response_schema is not None else None
                if payload["format"] is None:
                    payload.pop("format", None)
                resp = self._http_post("/api/chat", payload)
            else:
                hint = ""
                if e.code == 404 and "not found" in err_body.lower():
                    avail = ", ".join(self.list_models()) or "<none>"
                    hint = (
                        f"\nIf the model name is wrong, set OLLAMA_MODEL to a "
                        f"tag from `ollama list` (available: {avail})."
                    )
                raise RuntimeError(
                    f"Ollama HTTP {e.code} from /api/chat: {err_body[:300]}{hint}"
                ) from e

        # Confirm schema support on first success.
        if want_schema and self._schema_supported is None:
            self._schema_supported = True

        text = (resp.get("message") or {}).get("content", "") or resp.get("response", "")
        if not text:
            raise RuntimeError(f"Ollama returned empty response: {resp}")

        # Token logprobs not exposed by Ollama in a standardized way -> empty.
        return text, []



def create_llm_client(provider: str = "auto", **kwargs) -> BaseLLMClient:
    """
    Factory function to create an LLM client.

    Args:
        provider: "anthropic", "google", "ollama", or "auto"
                  (auto respects $LLM_PROVIDER, then tries cloud, then local).
        **kwargs: Additional arguments passed to the client

    Returns:
        An LLM client instance
    """
    # Allow env-var override even when caller passes "auto".
    if provider == "auto":
        env_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
        if env_provider in ("anthropic", "google", "ollama"):
            provider = env_provider

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

    if provider == "ollama":
        # Drop kwargs that don't apply to Ollama to avoid TypeErrors when the
        # caller passes shared cloud-style options.
        ollama_kwargs = {k: v for k, v in kwargs.items()
                         if k in {"model", "host", "temperature", "timeout",
                                  "num_ctx", "keep_alive"}}
        client = OllamaClient(**ollama_kwargs)
        if client.is_available():
            print(f"Using Ollama (model={client.model}, host={client.host})")
            return client
        raise RuntimeError(
            f"Ollama not reachable at {client.host}.\n"
            f"Install from https://ollama.com, run `ollama serve`, then "
            f"`ollama pull {client.model}`."
        )

    if provider == "auto":
        # Try Anthropic first.
        anthropic_client = AnthropicClient(**kwargs)
        if anthropic_client.is_available():
            print("Using Anthropic Claude")
            return anthropic_client

        # Then Google.
        google_client = GoogleClient(**kwargs)
        if google_client.is_available():
            print("Using Google Gemini")
            return google_client

        # Finally, fall back to a local Ollama server if one is running.
        ollama_kwargs = {k: v for k, v in kwargs.items()
                         if k in {"model", "host", "temperature", "timeout",
                                  "num_ctx", "keep_alive"}}
        ollama_client = OllamaClient(**ollama_kwargs)
        if ollama_client.is_available():
            print(f"Using Ollama (model={ollama_client.model}, host={ollama_client.host})")
            return ollama_client

        raise RuntimeError(
            "CRITICAL EXCEPTION: No LLM backend available.\n"
            "Set ANTHROPIC_API_KEY or GOOGLE_API_KEY, or run a local Ollama "
            "server (https://ollama.com) and `ollama pull qwen2.5-coder:7b`.\n"
            "You can also force a backend with LLM_PROVIDER=ollama."
        )

    raise ValueError(f"Unknown provider: {provider}")
