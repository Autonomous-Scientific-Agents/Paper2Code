"""Main entry point for the Paper2Code project.

This module provides the main functionality to convert academic papers into
executable code using AI-powered agents.
"""

import asyncio
import os
import logging
from pathlib import Path
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from crew.workflows import PaperToCodeWorkflow
from ai_client import get_ai_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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
            text += page.extract_text()
            
        logger.info("PDF text extraction completed")
        return text
    else:
        # Handle text files
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

async def main():
    """Main function to process papers and generate code."""
    logger.info("Starting Paper2Code")
    
    # Initialize AI client based on environment variable
    ai_client = get_ai_client()
    
    try:
        # Example paper path - replace with actual path
        paper_path = "examples/paper.pdf"
        
        # Read paper content
        paper_content = read_paper(paper_path)
        
        # Generate code using AI
        prompt = f"Convert this academic paper into executable code:\n\n{paper_content}"
        response = await ai_client.generate(prompt)
        
        logger.info("Generated code from paper:")
        print(response)
        
    except Exception as e:
        logger.error(f"Error processing paper: {e}")
        raise
    finally:
        await ai_client.close()

if __name__ == "__main__":
    asyncio.run(main())