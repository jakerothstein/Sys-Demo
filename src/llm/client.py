"""
Unified LLM Client supporting Anthropic Claude and Google Gemini.
Supports native structured outputs for guaranteed valid JSON responses.
"""
import json
import os
import math
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
    """Google Gemini client."""
    
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
            except ImportError:
                print("google-generativeai package not installed. Run: pip install google-generativeai")
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def generate(self, prompt: str, response_schema: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        if not self.is_available():
            raise RuntimeError("Google client not available")
        
        try:
            # Build generation config
            generation_config = {"temperature": self.temperature, "response_logprobs": True, "logprobs": 1}
            
            # Use structured output when schema is provided
            # Note: Gemini uses 'response_schema' parameter, not 'response_json_schema'
            if response_schema:
                generation_config["response_mime_type"] = "application/json"
                generation_config["response_schema"] = response_schema
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            token_data = []
            if getattr(response, "candidates", None) and response.candidates:
                candidate = response.candidates[0]
                if getattr(candidate, "logprobs_result", None) and getattr(candidate.logprobs_result, "top_candidates", None):
                    for token_candidates in candidate.logprobs_result.top_candidates:
                        if getattr(token_candidates, "candidates", None) and token_candidates.candidates:
                            top_candidate = token_candidates.candidates[0]
                            token_str = top_candidate.token
                            logprob = getattr(top_candidate, "log_probability", 0.0)
                            linear_prob = math.exp(logprob)
                            token_data.append({"token": token_str, "confidence": linear_prob})
                            
            return response.text, token_data
        except Exception as e:
            error_str = str(e)
            
            # If logprobs is not enabled for this model
            if "Logprobs is not enabled" in error_str:
                print(f"Logprobs not supported for this model, falling back to standard generation")
                generation_config = {"temperature": self.temperature}
                if response_schema:
                    generation_config["response_mime_type"] = "application/json"
                    generation_config["response_schema"] = response_schema
                
                response = self.client.generate_content(prompt, generation_config=generation_config)
                return response.text, []
                
            # If structured output fails, retry without it
            if response_schema and ("response_schema" in error_str or "Unknown field" in error_str):
                print(f"Structured output not supported, falling back to regular generation")
                generation_config = {"temperature": self.temperature, "response_logprobs": True, "logprobs": 1}
                response = self.client.generate_content(prompt, generation_config=generation_config)
                
                token_data = []
                if getattr(response, "candidates", None) and response.candidates:
                    candidate = response.candidates[0]
                    if getattr(candidate, "logprobs_result", None) and getattr(candidate.logprobs_result, "top_candidates", None):
                        for token_candidates in candidate.logprobs_result.top_candidates:
                            if getattr(token_candidates, "candidates", None) and token_candidates.candidates:
                                top_candidate = token_candidates.candidates[0]
                                token_str = top_candidate.token
                                logprob = getattr(top_candidate, "log_probability", 0.0)
                                linear_prob = math.exp(logprob)
                                token_data.append({"token": token_str, "confidence": linear_prob})
                                
                return response.text, token_data
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                raise RuntimeError(f"Rate limit exceeded. Please wait 30 seconds and try again. Error: {error_str[:200]}")
            
            # Add print so we can debug other underlying failures
            print(f"GoogleClient generate Exception: {error_str}")
            raise



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
