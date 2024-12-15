"""Workflows module for the Paper2Code project.

This module contains the workflow classes that orchestrate the paper-to-code
conversion process using various agents.
"""

import logging
from typing import List, Tuple

from .agents import (
    CodeBlock,
    CodeGenerator,
    CodeTester,
    PaperParser,
    DockerConfigGenerator,
    SoftwareExtractor,
)

logger = logging.getLogger(__name__)


class PaperToCodeWorkflow:
    """Workflow for converting academic papers to code."""

    def __init__(self) -> None:
        """Initialize the workflow with required agents."""
        logger.info("Initializing PaperToCodeWorkflow agents...")
        self.parser = PaperParser()
        self.software_extractor = SoftwareExtractor()
        self.generator = CodeGenerator()
        self.tester = CodeTester()
        self.docker_generator = DockerConfigGenerator()
        logger.info("All agents initialized successfully")

    async def process_paper(self, paper_text: str) -> Tuple[List[CodeBlock], str, str]:
        """Process an academic paper and generate validated code with Docker configs.

        Args:
            paper_text: The text content of the academic paper.

        Returns:
            Tuple containing:
            - List of validated and tested code blocks
            - Dockerfile content
            - docker-compose.yml content
        """
        logger.info("Starting paper analysis...")
        
        # Extract sections from the paper
        logger.info("Extracting sections from paper...")
        sections = await self.parser.extract_sections(paper_text)
        logger.info(f"Found {len(sections)} sections in the paper")

        # Extract software mentions
        logger.info("Extracting software mentions...")
        software_list = await self.software_extractor.extract_software(paper_text)
        formatted_output = self.software_extractor.format_output(software_list)
        logger.info("\n" + formatted_output)

        code_blocks = []
        for i, section in enumerate(sections, 1):
            logger.info(f"\nProcessing section {i}/{len(sections)}: {section.title}")
            
            # Extract equations from each section
            logger.info("Identifying equations...")
            equations = await self.parser.identify_equations(section)
            section.equations = equations
            if equations:
                logger.info(f"Found {len(equations)} equations")

            # Generate code for the section
            logger.info("Generating code implementation...")
            code_block = await self.generator.generate_code(section)
            logger.info(f"Generated code in {code_block.language}")

            # Refactor and improve the code
            logger.info("Refactoring and improving code...")
            improved_code = await self.generator.refactor_code(code_block)
            logger.info("Code refactoring completed")

            # Validate the code
            logger.info("Validating generated code...")
            is_valid = await self.tester.validate_code(improved_code)
            if is_valid:
                logger.info("Code validation successful")
                # Generate test cases
                logger.info("Generating test cases...")
                test_block = await self.tester.generate_tests(improved_code)
                code_blocks.extend([improved_code, test_block])
                logger.info("Test cases generated")
            else:
                logger.warning("Code validation failed, skipping test generation")

        # Generate Docker configurations
        logger.info("\nGenerating Docker configurations...")
        logger.info("Creating Dockerfile...")
        dockerfile_content = await self.docker_generator.generate_dockerfile(code_blocks, software_list)
        logger.info("Dockerfile generation completed")
        
        logger.info("Creating docker-compose.yml...")
        compose_content = await self.docker_generator.generate_compose(code_blocks, software_list)
        logger.info("docker-compose.yml generation completed")

        return code_blocks, dockerfile_content, compose_content