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
import openai

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

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")


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


async def main() -> None:
    """Main function to process papers and generate code."""
    logger.info("Starting Paper2Code process...")
    
    # Initialize the workflow
    logger.info("Initializing workflow...")
    workflow = PaperToCodeWorkflow()

    # Get the paper path from environment or use default
    paper_path = os.getenv("PAPER_PATH", "paper.txt")
    logger.info(f"Using paper path: {paper_path}")
    
    try:
        # Read the paper content
        paper_text = read_paper(paper_path)
        logger.info("Successfully read paper content")

        # Process the paper and get code blocks and Docker configs
        logger.info("Processing paper content...")
        code_blocks, dockerfile, docker_compose = await workflow.process_paper(paper_text)
        logger.info("Paper processing completed")

        # Create output directory if it doesn't exist
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

        # Write Docker configurations
        logger.info("Writing Docker configurations...")
        dockerfile_path = output_dir / "Dockerfile"
        compose_path = output_dir / "docker-compose.yml"
        
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)
        logger.info(f"Dockerfile written to: {dockerfile_path}")
        
        with open(compose_path, "w") as f:
            f.write(docker_compose)
        logger.info(f"docker-compose.yml written to: {compose_path}")

        # Print the results
        logger.info("\nGenerated code blocks:")
        for i, block in enumerate(code_blocks, 1):
            logger.info(f"\nCode Block {i}:")
            logger.info(f"Language: {block.language}")
            logger.info(f"Description: {block.description}")
            logger.info("Code:")
            logger.info(block.code)

        logger.info("\nDocker configurations have been generated in the 'outputs' directory.")
        logger.info("You can now build and run the Docker container using the generated files.")

    except FileNotFoundError:
        logger.error(f"Error: Paper file not found at {paper_path}")
        logger.error("Please set the PAPER_PATH environment variable to point to your paper file.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())