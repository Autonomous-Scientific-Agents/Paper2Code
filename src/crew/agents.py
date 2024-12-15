"""Agents module for the Paper2Code project.

This module contains the agent classes responsible for parsing papers,
generating code, and testing the generated code.
"""

from typing import List
from pydantic import BaseModel, Field
import openai
import json
import os
import logging
from ai_client import OllamaClient

logger = logging.getLogger(__name__)

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
        ai_provider = os.getenv("AI_PROVIDER")
        if ai_provider == "ollama":
            self.client = OllamaClient()  # Use Ollama client
            self.model = "llama3.2"  # Set model for Ollama
        else:
            self.client = openai.AsyncClient()  # Default to OpenAI client
            self.model = "gpt-4-turbo-preview"  # Set model for OpenAI

    async def _chat_completion(self, prompt: str) -> str:
        """Send a chat completion request to OpenAI API.

        Args:
            prompt: The prompt to send to the API.

        Returns:
            The response content from the API.
        """
        if isinstance(self.client, OllamaClient):
            response = await self.client.generate(prompt)
            return response  # Directly return the string response from Ollama
        else:
            try:
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
            except openai.RateLimitError:
                logger.error("Rate limit exceeded. Please check your OpenAI API usage and consider upgrading your plan.")
                return "Error: Rate limit exceeded."
            except openai.OpenAIError as e:
                logger.error(f"An OpenAI error occurred: {e}")
                return f"Error: {e}"
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                return f"Error: {e}"
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

    async def generate_dockerfile(self, code_blocks: List[CodeBlock], software_list: List[dict]) -> str:
        """Generate a Dockerfile based on the code requirements.

        Args:
            code_blocks: List of code blocks from the paper.
            software_list: List of software dependencies extracted from the paper.

        Returns:
            String containing the Dockerfile content.
        """
        # Analyze code blocks to determine requirements
        languages = set(block.language.lower() for block in code_blocks)
        dependencies = self._extract_dependencies(code_blocks)
        
        # Add software from software_list
        for sw in software_list:
            if sw['name'].lower() not in [d.lower() for d in dependencies]:
                dependencies.append(sw['name'])
                if sw.get('version'):
                    dependencies[-1] += f"=={sw['version']}"
        
        # Generate Dockerfile content based on requirements
        prompt = (
            "Generate a production-ready Dockerfile for a project with these requirements:\n"
            f"Languages: {', '.join(languages)}\n"
            f"Dependencies: {', '.join(dependencies)}\n\n"
            "Software details:\n"
        )
        
        # Add detailed software information
        for sw in software_list:
            version_str = f" v{sw['version']}" if sw.get('version') else ""
            prompt += f"- {sw['name']}{version_str}: {sw['purpose']}\n"
        
        prompt += "\nReturn only the Dockerfile content, without any explanations or additional text."
        
        response = await self._chat_completion(prompt)
        return response

    async def generate_compose(self, code_blocks: List[CodeBlock], software_list: List[dict]) -> str:
        """Generate docker-compose.yml based on the code requirements.

        Args:
            code_blocks: List of code blocks from the paper.
            software_list: List of software dependencies extracted from the paper.

        Returns:
            String containing the docker-compose.yml content.
        """
        # Analyze service requirements from code blocks
        services = self._identify_services(code_blocks)
        
        # Add services based on software requirements
        for sw in software_list:
            if self._is_service_software(sw['name']):
                service_name = sw['name'].lower().replace('-', '_')
                services.append({
                    'name': service_name,
                    'software': sw['name'],
                    'version': sw.get('version'),
                    'purpose': sw['purpose']
                })
        
        # Generate docker-compose content
        prompt = (
            "Generate a production-ready docker-compose.yml file for a project with these services:\n"
            f"{json.dumps(services, indent=2)}\n\n"
            "Software details:\n"
            "Return only the docker-compose.yml content, without any explanations or additional text."
        )
        
        # Add detailed software information
        for sw in software_list:
            version_str = f" v{sw['version']}" if sw.get('version') else ""
            prompt += f"- {sw['name']}{version_str}: {sw['purpose']}\n"
        
        response = await self._chat_completion(prompt)
        return response

    def _is_service_software(self, name: str) -> bool:
        """Check if the software typically runs as a service.
        
        Args:
            name: Name of the software
            
        Returns:
            True if the software typically runs as a service
        """
        service_keywords = {
            'db', 'database', 'redis', 'mongo', 'postgres', 'mysql', 'elasticsearch',
            'kafka', 'rabbitmq', 'queue', 'cache', 'server', 'api', 'service'
        }
        
        name_lower = name.lower()
        return (
            any(keyword in name_lower for keyword in service_keywords) or
            name_lower in {
                'postgresql', 'mongodb', 'mysql', 'mariadb', 'redis', 'elasticsearch',
                'kibana', 'kafka', 'zookeeper', 'rabbitmq', 'nginx', 'haproxy'
            }
        )

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


class SoftwareExtractor(BaseAgent):
    """Agent responsible for extracting software mentions from academic papers."""

    async def extract_software(self, paper_text: str) -> List[dict]:
        """Extract software mentions from the academic paper.

        Args:
            paper_text: The text content of the paper.

        Returns:
            List of dictionaries containing software information with fields:
            - name: Name of the software
            - version: Version if specified (optional)
            - purpose: Inferred purpose of the software
            - confidence: Confidence score of the inference
        """
        prompt = (
            "You are a software dependency analyzer. Your task is to extract all software mentions "
            "from this academic paper, including:\n"
            "1. Programming languages (e.g., Python, R, Java)\n"
            "2. Libraries and frameworks (e.g., TensorFlow, PyTorch, scikit-learn)\n"
            "3. Tools and platforms (e.g., Jupyter, Docker, AWS)\n"
            "4. Databases and data storage solutions (e.g., PostgreSQL, MongoDB)\n"
            "5. Development tools (e.g., Git, VS Code)\n\n"
            "Also infer any implicit software requirements based on the methods and techniques described. "
            "For example, if the paper mentions 'CNN' or 'deep learning', it likely requires deep learning frameworks.\n\n"
            "Return the result as a JSON array where each object has:\n"
            "- 'name': Software name\n"
            "- 'version': Version if specified (or null)\n"
            "- 'purpose': Brief description of how it's used\n"
            "- 'confidence': Score from 0.0 to 1.0 (use lower scores for inferred software)\n\n"
            "Example output format:\n"
            '[{"name": "Python", "version": "3.8", "purpose": "Implementation language", "confidence": 1.0},'
            '{"name": "TensorFlow", "version": null, "purpose": "Deep learning framework for CNN implementation", "confidence": 0.9}]\n\n'
            f"Paper text:\n{paper_text}"
        )

        response = await self._chat_completion(prompt)
        try:
            software_list = json.loads(response)
            # Filter out low confidence results (< 0.5)
            return [sw for sw in software_list if sw.get('confidence', 0) >= 0.5]
        except json.JSONDecodeError:
            return []

    def format_output(self, software_list: List[dict]) -> str:
        """Format the software list for console output.

        Args:
            software_list: List of software dictionaries.

        Returns:
            Formatted string for console output.
        """
        if not software_list:
            return "No software dependencies found in the paper."

        output = "Found Software Dependencies:\n"
        output += "-" * 50 + "\n"
        
        for sw in sorted(software_list, key=lambda x: x.get('confidence', 0), reverse=True):
            output += f"• {sw['name']}"
            if sw.get('version'):
                output += f" (v{sw['version']})"
            output += f"\n  Purpose: {sw['purpose']}"
            output += f"\n  Confidence: {sw['confidence']:.2f}\n"
        
        return output