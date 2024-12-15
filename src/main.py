"""Main entry point for Paper2Code."""

import asyncio
import logging
import os
from dotenv import load_dotenv
from pathlib import Path
import random
import string

from parsers.paper_parser import read_paper
from crew.workflows import PaperToCodeWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables from .env file."""
    env_path = Path(__file__).resolve().parent.parent / '.env'
    if not env_path.exists():
        logger.warning(f"No .env file found at {env_path}")
        return
    
    load_dotenv(env_path)
    logger.info(f"Loaded environment from {env_path}")
    
    # Read values from .env file directly
    ai_provider = os.getenv('AI_PROVIDER')
    paper_path = os.getenv('PAPER_PATH')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    logger.info(f"AI_PROVIDER is set to: {ai_provider}")
    logger.info(f"PAPER_PATH is set to: {paper_path}")
    logger.info(f"OPENAI_API_KEY is set to: {openai_api_key}")

# Function to generate a random alphanumeric string of fixed length
def generate_random_string(length=8):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

async def main():
    """Main function to process papers and generate code."""
    logger.info("Starting Paper2Code")
    
    try:
        # Load environment variables
        load_environment()
        
        # Get paper path from environment or use default
        paper_path = os.getenv("PAPER_PATH", "examples/paper.pdf")
        logger.info(f"Reading paper from: {paper_path}")
        
        # Read paper content
        paper_content = read_paper(paper_path)
        
        # Initialize and run the workflow
        workflow = PaperToCodeWorkflow()
        code_blocks, dockerfile, compose_file = await workflow.process_paper(paper_content)
        
        # Print the results
        if code_blocks:
            print("\nGenerated Code Blocks:")
            print("=" * 80)
            for i, block in enumerate(code_blocks, 1):
                print(f"\nCode Block {i}:")
                print(f"Language: {block.language}")
                print(f"Description: {block.description}")
                print("Code:")
                print("-" * 40)
                print(block.code)
                print("-" * 40)
        else:
            print("\nNo code blocks were generated.")
        
        if dockerfile:
            print("\nGenerated Dockerfile:")
            print("=" * 80)
            print(dockerfile)
        else:
            print("\nNo Dockerfile was generated.")
        
        if compose_file:
            print("\nGenerated docker-compose.yml:")
            print("=" * 80)
            print(compose_file)
        else:
            print("\nNo docker-compose.yml was generated.")
        
        # Create outputs directory if it doesn't exist
        outputs_dir = Path('outputs')
        outputs_dir.mkdir(exist_ok=True)

        # Create a random directory name and path
        random_dir_name = generate_random_string()
        random_dir_path = outputs_dir / random_dir_name
        random_dir_path.mkdir(exist_ok=True)

        # Save Dockerfile and docker-compose.yml in the random directory
        if dockerfile:
            with open(random_dir_path / 'Dockerfile', 'w') as dockerfile_file:
                dockerfile_file.write(dockerfile)

        if compose_file:
            with open(random_dir_path / 'docker-compose.yml', 'w') as composefile_file:
                composefile_file.write(compose_file)
        
    except Exception as e:
        logger.error(f"Error processing paper: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())