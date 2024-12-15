"""Main entry point for the Paper2Code project.

This module provides the main functionality to convert academic papers into
executable code using AI-powered agents.
"""

import asyncio
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import httpx
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from crew.workflows import PaperToCodeWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class OllamaRequest(BaseModel):
    """Request model for Ollama API."""
    model: str = Field(..., description="Name of the model to use")
    prompt: str = Field(..., description="The prompt to send to the model")
    stream: bool = Field(default=False, description="Whether to stream the response")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Additional model parameters")

class OllamaClient:
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

def read_paper(file_path: str) -> str:
    """Read paper content from a file, supporting both PDF and text formats.
    
    Args:
        file_path: Path to the paper file
        
    Returns:
        String content of the paper
    """
    file_path = Path(file_path)
    logger.info(f"Reading paper from: {file_path}")
    
    if file_path.suffix.lower() == '.pdf':
        # Handle PDF files
        logger.info("Detected PDF format, extracting text...")
        reader = PdfReader(file_path)
        text = ""
        total_pages = len(reader.pages)
        logger.info(f"Processing {total_pages} pages...")
        
        for i, page in enumerate(reader.pages, 1):
            logger.info(f"Extracting text from page {i}/{total_pages}")
            text += page.extract_text() + "\n"
        
        logger.info("PDF text extraction completed")
        return text
    else:
        # Handle text files
        logger.info("Reading text file...")
        with open(file_path, "r", encoding='utf-8') as f:
            text = f.read()
        logger.info("Text file reading completed")
        return text


async def main():
    """Main function to process papers and generate code."""
    logger.info("Starting Paper2Code")
    
    # Initialize Ollama client
    ollama = OllamaClient()
    
    try:
        # Example paper path - replace with actual path
        paper_path = "examples/paper.pdf"
        
        # Read paper content
        paper_content = read_paper(paper_path)
        
        # Generate code using Ollama
        prompt = f"Convert this academic paper into executable code:\n\n{paper_content}"
        response = await ollama.generate(prompt)
        
        logger.info("Generated code from paper:")
        print(response)
        
    except Exception as e:
        logger.error(f"Error processing paper: {e}")
        raise
    finally:
        await ollama.close()

if __name__ == "__main__":
    asyncio.run(main())