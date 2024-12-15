"""AI client implementations for different providers."""

from abc import ABC, abstractmethod
import os
from typing import Optional, Dict, Any
import httpx
import openai
from pydantic import BaseModel, Field

class OllamaRequest(BaseModel):
    """Request model for Ollama API."""
    model: str = Field(..., description="Name of the model to use")
    prompt: str = Field(..., description="The prompt to send to the model")
    stream: bool = Field(default=False, description="Whether to stream the response")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Additional model parameters")
    context: Optional[list[int]] = Field(default=None, description="The context parameter returned from a previous request")
    template: Optional[str] = Field(default=None, description="The prompt template to use")

class AIClient(ABC):
    """Abstract base class for AI clients."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the AI model.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters for the request
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    async def close(self):
        """Close any open connections."""
        pass

class OllamaClient(AIClient):
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(300.0)  # 5 minutes timeout
        )
        self.model = "llama3.2:latest"  # Default model
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from Ollama.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters for the request
            
        Returns:
            Generated text response
        """
        request = OllamaRequest(
            model=self.model,
            prompt=prompt,
            **kwargs
        )
        
        response = await self.client.post(
            "/api/generate",
            json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return response.json()["response"]
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

class OpenAIClient(AIClient):
    """Client for interacting with OpenAI API."""
    
    def __init__(self):
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = "gpt-4"  # Default model
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from OpenAI.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters for the request
            
        Returns:
            Generated text response
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
    
    async def close(self):
        """Close any open connections."""
        # OpenAI client doesn't require explicit cleanup
        pass

def get_ai_client() -> AIClient:
    """Factory function to get the appropriate AI client based on environment variables.
    
    Returns:
        An instance of AIClient (either OllamaClient or OpenAIClient)
    """
    ai_provider = os.getenv("AI_PROVIDER", "ollama").lower()
    
    if ai_provider == "ollama":
        return OllamaClient()
    elif ai_provider == "openai":
        return OpenAIClient()
    else:
        raise ValueError(f"Unknown AI provider: {ai_provider}. Must be 'ollama' or 'openai'.")
