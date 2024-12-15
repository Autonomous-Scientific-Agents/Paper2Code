"""Ollama client integration with Pydantic models.

This module provides a Pydantic-based interface for interacting with Ollama.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import httpx

class OllamaRequest(BaseModel):
    """Request model for Ollama API."""
    model: str = Field(..., description="Name of the model to use")
    prompt: str = Field(..., description="The prompt to send to the model")
    stream: bool = Field(default=False, description="Whether to stream the response")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Additional model parameters")
    context: Optional[List[int]] = Field(default=None, description="The context parameter returned from a previous request")
    template: Optional[str] = Field(default=None, description="The prompt template to use")

class OllamaResponse(BaseModel):
    """Response model from Ollama API."""
    model: str = Field(..., description="The model used for the response")
    created_at: str = Field(..., description="Timestamp of when the response was created")
    response: str = Field(..., description="The generated response text")
    context: Optional[List[int]] = Field(default=None, description="Context for use in follow-up requests")
    done: bool = Field(..., description="Whether the response is complete")

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        """Initialize the Ollama client.
        
        Args:
            base_url: Base URL for the Ollama API
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(300.0)  # 5 minutes timeout
        )
    
    async def generate(self, request: OllamaRequest) -> OllamaResponse:
        """Generate a response from Ollama.
        
        Args:
            request: The request model containing parameters for generation
            
        Returns:
            OllamaResponse containing the generated text and metadata
        """
        response = await self.client.post(
            "/api/generate",
            json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return OllamaResponse(**response.json())
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
