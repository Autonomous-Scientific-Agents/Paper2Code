"""Agents module for the Paper2Code project.

This module contains the agent classes responsible for parsing papers,
generating code, and testing the generated code.
"""

from typing import List
from pydantic import BaseModel, Field
from openai import AsyncClient
import json


class CodeBlock(BaseModel):
    """Represents a block of code with its metadata."""

    language: str = Field(
        ...,
        description="Programming language of the code block"
    )
    code: str = Field(
        ...,
        description="The actual code content"
    )
    description: str = Field(
        ...,
        description="Description of what the code does"
    )


class PaperSection(BaseModel):
    """Represents a section of an academic paper."""

    title: str = Field(
        ...,
        description="Title of the section"
    )
    content: str = Field(
        ...,
        description="Content of the section"
    )
    equations: List[str] = Field(
        default_factory=list,
        description="Mathematical equations found in the section"
    )
    code_blocks: List[CodeBlock] = Field(
        default_factory=list,
        description="Code blocks extracted from the section"
    )


class BaseAgent:
    """Base agent class with common OpenAI API functionality."""

    def __init__(self) -> None:
        """Initialize the agent with OpenAI client."""
        self.client = AsyncClient()
        self.model = "gpt-4-turbo-preview"

    async def _chat_completion(self, prompt: str) -> str:
        """Send a chat completion request to OpenAI API.

        Args:
            prompt: The prompt to send to the API.

        Returns:
            The response content from the API.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant that helps with "
                        "code generation and analysis."
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


class PaperParser(BaseAgent):
    """Agent responsible for parsing academic papers."""

    async def extract_sections(self, paper_text: str) -> List[PaperSection]:
        """Extract sections from the academic paper.

        Args:
            paper_text: The text content of the paper.

        Returns:
            List of PaperSection objects.
        """
        prompt = (
            "Extract sections from the following academic paper. Return the "
            "result as a JSON array where each object has 'title' and "
            "'content' fields.\n\nPaper:\n{paper_text}"
        ).format(paper_text=paper_text)

        response = await self._chat_completion(prompt)
        try:
            sections_data = json.loads(response)
            return [PaperSection(**section) for section in sections_data]
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return [
                PaperSection(
                    title="Main Section",
                    content=paper_text,
                    equations=[],
                    code_blocks=[]
                )
            ]

    async def identify_equations(self, section: PaperSection) -> List[str]:
        """Identify and extract mathematical equations from a paper section.

        Args:
            section: The paper section to analyze.

        Returns:
            List of equation strings.
        """
        prompt = (
            "Extract mathematical equations from the following section. "
            "Return the result as a JSON array of equation strings.\n\n"
            "Section content:\n{content}"
        ).format(content=section.content)

        response = await self._chat_completion(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return []


class CodeGenerator(BaseAgent):
    """Agent responsible for generating code from paper sections."""

    async def generate_code(self, section: PaperSection) -> CodeBlock:
        """Generate code implementation from a paper section.

        Args:
            section: The paper section to implement.

        Returns:
            A CodeBlock containing the implementation.
        """
        prompt = (
            "Generate code implementation for the following section. Return "
            "the result as a JSON object with 'language', 'code', and "
            "'description' fields.\n\n"
            "Title: {title}\nContent: {content}\nEquations: {equations}"
        ).format(
            title=section.title,
            content=section.content,
            equations=section.equations
        )

        response = await self._chat_completion(prompt)
        try:
            return CodeBlock(**json.loads(response))
        except json.JSONDecodeError:
            return CodeBlock(
                language="python",
                code="# Unable to generate code",
                description=f"Implementation for section: {section.title}"
            )

    async def refactor_code(self, code_block: CodeBlock) -> CodeBlock:
        """Refactor and improve generated code.

        Args:
            code_block: The code block to refactor.

        Returns:
            A refactored CodeBlock.
        """
        prompt = (
            "Refactor and improve the following code. Return the result as a "
            "JSON object with 'language', 'code', and 'description' fields.\n\n"
            "Language: {language}\nCode: {code}\n"
            "Description: {description}"
        ).format(
            language=code_block.language,
            code=code_block.code,
            description=code_block.description
        )

        response = await self._chat_completion(prompt)
        try:
            return CodeBlock(**json.loads(response))
        except json.JSONDecodeError:
            return code_block


class CodeTester(BaseAgent):
    """Agent responsible for testing and validating generated code."""

    async def validate_code(self, code_block: CodeBlock) -> bool:
        """Validate if the generated code meets the requirements.

        Args:
            code_block: The code block to validate.

        Returns:
            True if the code is valid, False otherwise.
        """
        prompt = (
            "Validate if the following code meets the requirements. Return "
            "'true' or 'false'.\n\n"
            "Language: {language}\nCode: {code}\n"
            "Description: {description}"
        ).format(
            language=code_block.language,
            code=code_block.code,
            description=code_block.description
        )

        response = await self._chat_completion(prompt)
        return response.lower().strip() == 'true'

    async def generate_tests(self, code_block: CodeBlock) -> CodeBlock:
        """Generate test cases for the code block.

        Args:
            code_block: The code block to generate tests for.

        Returns:
            A CodeBlock containing the test cases.
        """
        prompt = (
            "Generate test cases for the following code. Return the result "
            "as a JSON object with 'language', 'code', and 'description' "
            "fields.\n\n"
            "Language: {language}\nCode: {code}\n"
            "Description: {description}"
        ).format(
            language=code_block.language,
            code=code_block.code,
            description=code_block.description
        )

        response = await self._chat_completion(prompt)
        try:
            return CodeBlock(**json.loads(response))
        except json.JSONDecodeError:
            return CodeBlock(
                language=code_block.language,
                code=code_block.code + "\n\n# No tests generated",
                description=code_block.description + " (no tests)"
            )


class DockerConfigGenerator(BaseAgent):
    """Agent responsible for generating Docker configurations based on paper requirements."""

    async def generate_dockerfile(self, code_blocks: List[CodeBlock]) -> str:
        """Generate a Dockerfile based on the code requirements.

        Args:
            code_blocks: List of code blocks from the paper.

        Returns:
            String containing the Dockerfile content.
        """
        # Analyze code blocks to determine requirements
        languages = set(block.language.lower() for block in code_blocks)
        dependencies = self._extract_dependencies(code_blocks)
        
        # Generate Dockerfile content based on requirements
        dockerfile_content = await self._generate_docker_config(languages, dependencies)
        return dockerfile_content

    async def generate_compose(self, code_blocks: List[CodeBlock]) -> str:
        """Generate docker-compose.yml based on the code requirements.

        Args:
            code_blocks: List of code blocks from the paper.

        Returns:
            String containing the docker-compose.yml content.
        """
        # Analyze service requirements from code blocks
        services = self._identify_services(code_blocks)
        
        # Generate docker-compose content
        compose_content = await self._generate_compose_config(services)
        return compose_content

    def _extract_dependencies(self, code_blocks: List[CodeBlock]) -> List[str]:
        """Extract required dependencies from code blocks."""
        dependencies = set()
        for block in code_blocks:
            # Look for import statements and requirements
            if block.language.lower() == "python":
                for line in block.code.split("\n"):
                    if line.startswith(("import ", "from ")):
                        package = line.split()[1].split(".")[0]
                        if package not in ["os", "sys", "typing"]:
                            dependencies.add(package)
        return list(dependencies)

    def _identify_services(self, code_blocks: List[CodeBlock]) -> List[dict]:
        """Identify required services from code blocks."""
        services = []
        current_service = {}
        
        for block in code_blocks:
            # Analyze code to identify service requirements
            if "server" in block.description.lower() or "api" in block.description.lower():
                current_service = {
                    "name": block.description.split()[0].lower(),
                    "type": "web",
                    "language": block.language,
                    "code": block.code
                }
                services.append(current_service)
                
        return services

    async def _generate_docker_config(self, languages: set, dependencies: List[str]) -> str:
        """Generate Dockerfile content based on requirements."""
        prompt = f"""Generate a Dockerfile for a project with the following requirements:
        Languages: {', '.join(languages)}
        Dependencies: {', '.join(dependencies)}
        The Dockerfile should follow best practices and be production-ready."""
        
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def _generate_compose_config(self, services: List[dict]) -> str:
        """Generate docker-compose.yml content based on services."""
        prompt = f"""Generate a docker-compose.yml file for a project with the following services:
        {json.dumps(services, indent=2)}
        The configuration should follow best practices and be production-ready."""
        
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content