
from pathlib import Path
import csv

from pypdf import PdfReader
from docx import Document as DocxDocument
from bs4 import BeautifulSoup

from langchain_core.documents import Document


def read_text_file(file_path: Path) -> str:
    """Lire un fichier texte avec plusieurs encodages possibles."""
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="latin-1")


def load_pdf(file_path: Path) -> list[Document]:
    """Extraire le texte de chaque page d'un PDF."""
    reader = PdfReader(str(file_path))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "file_type": "pdf",
                        "page": page_number
                    }
                )
            )

    return documents


def load_docx(file_path: Path) -> list[Document]:
    """Extraire les paragraphes et les tableaux d'un document Word."""
    word_document = DocxDocument(str(file_path))

    contents = []

    for paragraph in word_document.paragraphs:
        if paragraph.text.strip():
            contents.append(paragraph.text)

    for table in word_document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            contents.append(" | ".join(cells))

    text = "\n".join(contents)

    return [
        Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "file_type": "docx"
            }
        )
    ] if text.strip() else []


def load_txt(file_path: Path) -> list[Document]:
    """Lire un fichier TXT ou Markdown."""
    text = read_text_file(file_path)

    return [
        Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "file_type": file_path.suffix.lower().replace(".", "")
            }
        )
    ] if text.strip() else []


def load_html(file_path: Path) -> list[Document]:
    """Extraire le texte d'une page HTML."""
    html_content = read_text_file(file_path)
    soup = BeautifulSoup(html_content, "html.parser")

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    text = soup.get_text(separator="\n", strip=True)

    return [
        Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "file_type": "html"
            }
        )
    ] if text.strip() else []


def load_csv(file_path: Path) -> list[Document]:
    """Transformer chaque ligne d'un CSV en document."""
    documents = []

    with file_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        for row_number, row in enumerate(reader, start=1):
            text = "\n".join(
                f"{key}: {value}"
                for key, value in row.items()
                if key and value is not None
            )

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "file_type": "csv",
                            "row": row_number
                        }
                    )
                )

    return documents


def load_document(file_path: str) -> list[Document]:
    """
    Charger un document selon son extension.

    Formats acceptés :
    PDF, DOCX, TXT, MD, HTML et CSV.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Le fichier n'existe pas : {file_path}"
        )

    if not path.is_file():
        raise ValueError("Le chemin indiqué n'est pas un fichier.")

    extension = path.suffix.lower()

    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_txt,
        ".md": load_txt,
        ".html": load_html,
        ".htm": load_html,
        ".csv": load_csv
    }

    if extension not in loaders:
        raise ValueError(
            f"Format non supporté : {extension}"
        )

    documents = loaders[extension](path)

    print(f"Document chargé : {path.name}")
    print(f"Nombre de documents extraits : {len(documents)}")

    return documents
