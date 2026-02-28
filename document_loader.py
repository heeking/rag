"""Document loading utilities."""

from pathlib import Path
from typing import List

from langchain_core.documents import Document


def load_text_file(file_path: str | Path) -> List[Document]:
    """Load a single .txt file as a LangChain Document."""
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": str(file_path)})]


def load_documents_from_dir(dir_path: str | Path) -> List[Document]:
    """Load all .txt files from a directory."""
    dir_path = Path(dir_path)
    docs: List[Document] = []
    for f in sorted(dir_path.glob("*.txt")):
        docs.extend(load_text_file(f))
    return docs
