"""Read paper content from a file, supporting both PDF and text formats."""

from PyPDF2 import PdfReader
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

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
            text += page.extract_text() or ""
            
        logger.info("PDF text extraction completed")
        return text
    else:
        # Handle text files
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()