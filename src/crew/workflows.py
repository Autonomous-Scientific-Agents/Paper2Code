"""Workflows module for the Paper2Code project.

This module contains the workflow classes that orchestrate the paper-to-code
conversion process using various agents.
"""

from typing import List

from .agents import (
    CodeBlock,
    CodeGenerator,
    CodeTester,
    PaperParser,
)


class PaperToCodeWorkflow:
    """Main workflow for converting academic papers to code."""

    def __init__(self) -> None:
        """Initialize the workflow with required agents."""
        self.parser = PaperParser()
        self.generator = CodeGenerator()
        self.tester = CodeTester()

    async def process_paper(self, paper_text: str) -> List[CodeBlock]:
        """Process an academic paper and generate validated code.

        Args:
            paper_text: The text content of the academic paper.

        Returns:
            List of validated and tested code blocks.
        """
        # Extract sections from the paper
        sections = await self.parser.extract_sections(paper_text)

        code_blocks = []
        for section in sections:
            # Extract equations from each section
            equations = await self.parser.identify_equations(section)
            section.equations = equations

            # Generate code for the section
            code_block = await self.generator.generate_code(section)

            # Refactor and improve the code
            improved_code = await self.generator.refactor_code(code_block)

            # Validate the code
            is_valid = await self.tester.validate_code(improved_code)
            if is_valid:
                # Generate tests for the code
                tested_code = await self.tester.generate_tests(improved_code)
                code_blocks.append(tested_code)

        return code_blocks