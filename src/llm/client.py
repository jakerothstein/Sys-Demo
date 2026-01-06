"""
Unified LLM Client supporting Anthropic Claude and Google Gemini.
Supports native structured outputs for guaranteed valid JSON responses.
"""
import json
import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, response_schema: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The input prompt
            response_schema: Optional JSON schema dict to enforce structured output.
                             When provided, the response is guaranteed to be valid JSON.
            **kwargs: Additional arguments
            
        Returns:
            Raw response text (guaranteed valid JSON if response_schema provided)
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
    
    def generate(self, prompt: str, response_schema: Optional[Dict[str, Any]] = None, max_tokens: int = 2000, **kwargs) -> str:
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
        
        return message.content[0].text


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
    
    def generate(self, prompt: str, response_schema: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("Google client not available")
        
        try:
            # Build generation config
            generation_config = {"temperature": self.temperature}
            
            # Use structured output when schema is provided
            # Note: Gemini uses 'response_schema' parameter, not 'response_json_schema'
            if response_schema:
                generation_config["response_mime_type"] = "application/json"
                generation_config["response_schema"] = response_schema
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            # If structured output fails, retry without it
            if response_schema and ("response_schema" in error_str or "Unknown field" in error_str):
                print(f"Structured output not supported, falling back to regular generation")
                generation_config = {"temperature": self.temperature}
                response = self.client.generate_content(prompt, generation_config=generation_config)
                return response.text
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                raise RuntimeError(f"Rate limit exceeded. Please wait 30 seconds and try again. Error: {error_str[:200]}")
            raise


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing without API keys."""
    
    def __init__(self):
        pass
    
    def is_available(self) -> bool:
        return True
    
    def generate(self, prompt: str, response_schema: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Return mock responses based on prompt content."""
        prompt_lower = prompt.lower()
        
        # Mock disambiguation - detect the new LLM-first prompt format
        if "sql analyst" in prompt_lower and "analyze" in prompt_lower:
            # Extract just the user query to check for vagueness
            user_query = ""
            for line in prompt.split('\n'):
                if line.strip().lower().startswith('user query:'):
                    user_query = line.split(':', 1)[1].strip().strip('"').lower()
                    break
            
            # Check for vague markers only in the user query
            # Note: Removed "?" since questions naturally end with question marks
            is_vague = any(marker in user_query for marker in ["maybe", "probably", "stuff", "something like", "idk", "whatever"])
            tables = self._extract_tables_from_prompt(prompt)
            
            if is_vague:
                return json.dumps({
                    "confidence": 0.4,
                    "is_ambiguous": True,
                    "detected_tables": tables[:2] if tables else [],
                    "detected_intent": "select",
                    "ambiguity_reasons": ["Query is unclear or incomplete"],
                    "suggested_clarification": "Could you please clarify what you're looking for?"
                })
            else:
                # Detect intent from query
                intent = "select"
                if "count" in prompt_lower or "how many" in prompt_lower:
                    intent = "count"
                elif "total" in prompt_lower or "sum" in prompt_lower:
                    intent = "sum"
                elif "top" in prompt_lower or "highest" in prompt_lower:
                    intent = "ranking"
                
                return json.dumps({
                    "confidence": 0.85,
                    "is_ambiguous": False,
                    "detected_tables": tables[:2] if tables else ["customers"],
                    "detected_intent": intent,
                    "ambiguity_reasons": [],
                    "suggested_clarification": None
                })
        
        # Mock SQL generation - extract table names from schema in prompt
        if "sql developer" in prompt_lower or ("generate" in prompt_lower and "sql" in prompt_lower):
            tables = self._extract_tables_from_prompt(prompt)
            primary_table = tables[0] if tables else "customers"
            
            if "count" in prompt_lower or "how many" in prompt_lower:
                return json.dumps({
                    "sql_queries": [
                        f"SELECT COUNT(*) as count FROM {primary_table};",
                        f"SELECT COUNT(*) FROM {primary_table};",
                        f"SELECT COUNT(*) as total FROM {primary_table};"
                    ],
                    "reasoning": "Counting all rows in the table"
                })
            elif "highest" in prompt_lower or "most" in prompt_lower or "max" in prompt_lower or "top" in prompt_lower:
                return json.dumps({
                    "sql_queries": [
                        f"SELECT * FROM {primary_table} ORDER BY id DESC LIMIT 5;",
                        f"SELECT * FROM {primary_table} ORDER BY id DESC LIMIT 10;",
                        f"SELECT * FROM {primary_table} LIMIT 5;"
                    ],
                    "reasoning": "Getting top records from the table"
                })
            elif "show" in prompt_lower or "list" in prompt_lower:
                return json.dumps({
                    "sql_queries": [
                        f"SELECT * FROM {primary_table} LIMIT 20;",
                        f"SELECT * FROM {primary_table};",
                        f"SELECT * FROM {primary_table} ORDER BY 1 LIMIT 10;"
                    ],
                    "reasoning": "Listing records from the table"
                })
            else:
                return json.dumps({
                    "sql_queries": [
                        f"SELECT * FROM {primary_table} LIMIT 10;",
                        f"SELECT * FROM {primary_table};",
                        f"SELECT * FROM {primary_table} ORDER BY 1 LIMIT 10;"
                    ],
                    "reasoning": "General query on the table"
                })
        
        # Legacy format handling (backward compatibility)
        if "analyze" in prompt_lower and "ambiguity" in prompt_lower:
            return '{"confidence": 0.85, "is_ambiguous": false, "reasoning": "Query is clear."}'
        
        # Default response
        return "Mock response for: " + prompt[:100]
    
    def _extract_tables_from_prompt(self, prompt: str) -> list:
        """Extract table names from schema section of prompt."""
        tables = []
        lines = prompt.split('\n')
        for line in lines:
            # Look for "Table: tablename" or "### Table: tablename" pattern
            line_stripped = line.strip()
            if line_stripped.startswith('Table:') or line_stripped.startswith('### Table:'):
                table_name = line_stripped.split(':', 1)[1].strip()
                if table_name:
                    tables.append(table_name)
        return tables


def create_llm_client(provider: str = "auto", **kwargs) -> BaseLLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        provider: "anthropic", "google", "mock", or "auto" (tries in order)
        **kwargs: Additional arguments passed to the client
        
    Returns:
        An LLM client instance
    """
    if provider == "mock":
        return MockLLMClient()
    
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
        
        # Fall back to mock
        print("No LLM API keys found. Using mock client.")
        return MockLLMClient()
    
    raise ValueError(f"Unknown provider: {provider}")
