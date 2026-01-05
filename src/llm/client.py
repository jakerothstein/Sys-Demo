"""
Unified LLM Client supporting Anthropic Claude and Google Gemini.
"""
import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
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
    
    def generate(self, prompt: str, max_tokens: int = 2000, **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("Anthropic client not available")
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text


class GoogleClient(BaseLLMClient):
    """Google Gemini client."""
    
    def __init__(self, model: str = "gemini-2.0-flash", temperature: float = 0.1):
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
    
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("Google client not available")
        
        try:
            response = self.client.generate_content(
                prompt,
                generation_config={"temperature": self.temperature}
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                raise RuntimeError(f"Rate limit exceeded. Please wait 30 seconds and try again. Error: {error_str[:200]}")
            raise


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing without API keys."""
    
    def __init__(self):
        pass
    
    def is_available(self) -> bool:
        return True
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Return mock responses based on prompt content."""
        prompt_lower = prompt.lower()
        
        # Mock disambiguation
        if "analyze" in prompt_lower and "ambiguity" in prompt_lower:
            return '{"confidence": 0.85, "is_ambiguous": false, "reasoning": "Query is clear."}'
        
        # Mock SQL generation - extract table names from schema in prompt
        if "generate" in prompt_lower and "sql" in prompt_lower:
            # Try to extract table names from the schema in the prompt
            tables = self._extract_tables_from_prompt(prompt)
            primary_table = tables[0] if tables else "unknown_table"
            
            if "count" in prompt_lower or "how many" in prompt_lower:
                return f"""SQL_1: SELECT COUNT(*) as count FROM {primary_table};
SQL_2: SELECT COUNT(*) FROM {primary_table};
SQL_3: SELECT COUNT(*) as total FROM {primary_table};"""
            elif "highest" in prompt_lower or "most" in prompt_lower or "max" in prompt_lower:
                return f"""SQL_1: SELECT * FROM {primary_table} ORDER BY salary DESC LIMIT 1;
SQL_2: SELECT * FROM {primary_table} ORDER BY salary DESC LIMIT 5;
SQL_3: SELECT * FROM {primary_table} LIMIT 10;"""
            elif "show" in prompt_lower or "list" in prompt_lower:
                return f"""SQL_1: SELECT * FROM {primary_table} LIMIT 20;
SQL_2: SELECT * FROM {primary_table};
SQL_3: SELECT * FROM {primary_table} ORDER BY 1 LIMIT 10;"""
            else:
                return f"""SQL_1: SELECT * FROM {primary_table} LIMIT 10;
SQL_2: SELECT * FROM {primary_table};
SQL_3: SELECT * FROM {primary_table} ORDER BY 1 LIMIT 10;"""
        
        # Default response
        return "Mock response for: " + prompt[:100]
    
    def _extract_tables_from_prompt(self, prompt: str) -> list:
        """Extract table names from schema section of prompt."""
        tables = []
        lines = prompt.split('\n')
        for line in lines:
            # Look for "Table: tablename" pattern
            if line.strip().startswith('Table:'):
                table_name = line.split(':', 1)[1].strip()
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
