"""Main entry point for the Paper2Code project.

This module provides the main functionality to convert academic papers into
executable code using AI-powered agents.
"""

import asyncio
import os
from dotenv import load_dotenv
import openai

from crew.workflows import PaperToCodeWorkflow

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")


async def main() -> None:
    """Main function to process papers and generate code."""
    # Initialize the workflow
    workflow = PaperToCodeWorkflow()

    # Example paper text (in practice, this would come from a file or input)
    paper_text = """
    # Example Paper
    This is a sample academic paper with some equations and code.

    ## Methods
    We use the following equation to calculate the result:
    y = mx + b

    ## Implementation
    The implementation should use Python to calculate linear regression.
    """

    try:
        # Process the paper and get code blocks
        code_blocks = await workflow.process_paper(paper_text)

        # Print the results
        for i, block in enumerate(code_blocks, 1):
            print(f"\nCode Block {i}:")
            print(f"Language: {block.language}")
            print(f"Description: {block.description}")
            print("Code:")
            print(block.code)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Run the async main function
    asyncio.run(main())