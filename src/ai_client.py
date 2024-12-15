"""AI client implementations for different providers."""

from abc import ABC, abstractmethod
import os
from typing import Optional, Dict, Any
import httpx
import openai
from pydantic import BaseModel, Field
import tiktoken
import logging

logger = logging.getLogger(__name__)

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
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(300.0)  # 5 minutes timeout
        )
        self.model = model  # Set model based on the parameter
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from Ollama.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters for the request
            
        Returns:
            Generated text response
        """
        logger.info(f"Generating response from Ollama with prompt: {prompt}")
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
        self.model = "gpt-3.5-turbo-16k"  # Using 16k model for larger context
        self.encoding = tiktoken.encoding_for_model(self.model)
        self.max_tokens = 14000  # Leave room for response
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.encoding.encode(text))
    
    def _split_text(self, text: str, max_tokens: int) -> list[str]:
        """Split text into chunks that fit within token limit."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split into sentences (rough approximation)
        sentences = text.split(". ")
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_length + sentence_tokens > max_tokens:
                # Current chunk is full, start a new one
                if current_chunk:
                    chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        if current_chunk:
            chunks.append(". ".join(current_chunk))
        
        return chunks
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from OpenAI.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters for the request
            
        Returns:
            Generated text response
        """
        # Split the prompt into manageable chunks
        chunks = self._split_text(prompt, self.max_tokens)
        responses = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                chunk_prompt = f"You are analyzing an academic paper. This is part 1 of {len(chunks)}. Extract the key implementation details: {chunk}"
            else:
                chunk_prompt = f"This is part {i+1} of {len(chunks)} of the same paper. Continue analyzing: {chunk}"
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": chunk_prompt}],
                **kwargs
            )
            responses.append(response.choices[0].message.content)
        
        # If we had multiple chunks, summarize them
        if len(responses) > 1:
            summary_prompt = "Combine and summarize all the implementation details from the paper parts into a cohesive response: " + " ".join(responses)
            final_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": summary_prompt}],
                **kwargs
            )
            return final_response.choices[0].message.content
        
        return responses[0]
    
    async def close(self):
        """Close any open connections."""
        # OpenAI client doesn't require explicit cleanup
        pass

def get_ai_client() -> AIClient:
    """Factory function to get the appropriate AI client based on environment variables.
    
    Returns:
        An instance of AIClient (either OllamaClient or OpenAIClient)
    """
    provider = os.getenv("AI_PROVIDER", "openai").lower()
    
    logger.info(f"Instantiating AI client for provider: {provider}")
    
    if provider == "ollama":
        return OllamaClient()
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required when using OpenAI provider")
        return OpenAIClient()
    else:
        raise ValueError(f"Unsupported AI provider: {provider}. Must be one of: ollama, openai")
