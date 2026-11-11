
from pathlib import Path
import sys

# Ajouter le dossier principal du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.document_loader import load_document
from app.ingestion.text_splitter import split_documents
from app.retrieval.vector_store import add_documents_to_store


# Dossier contenant les documents
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Formats acceptés par notre application
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".html",
    ".htm",
    ".md",
    ".csv"
}


def ingest_all_documents():

    print("\n===== INGESTION DE PLUSIEURS DOCUMENTS =====")

    # 1. Trouver tous les fichiers compatibles
    files = sorted([
        file_path
        for file_path in DATA_DIR.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    print("\nNombre de fichiers détectés :", len(files))

    if not files:
        print("Aucun document compatible trouvé.")
        return

    all_chunks = []
    total_documents = 0
    successful_files = 0

    # 2. Charger chaque document
    for file_path in files:

        print("\n--------------------------------")
        print("Fichier :", file_path.name)

        try:
            documents = load_document(str(file_path))

            # Garder un chemin relatif comme source
            relative_source = str(
                file_path.relative_to(PROJECT_ROOT)
            )

            for document in documents:
                document.metadata["source"] = relative_source

            print("Documents extraits :", len(documents))

            # 3. Découper les documents en chunks
            chunks = split_documents(documents)

            print("Chunks créés :", len(chunks))

            # Ajouter les chunks à la liste générale
            all_chunks.extend(chunks)

            total_documents += len(documents)
            successful_files += 1

        except Exception as error:
            print("Erreur lors du traitement :", error)

    # 4. Enregistrer tous les chunks dans ChromaDB
    if not all_chunks:
        print("\nAucun chunk à enregistrer.")
        return

    print("\n===== ENREGISTREMENT CHROMADB =====")

    add_documents_to_store(all_chunks)

    # 5. Afficher le bilan
    print("\n===== BILAN DE L'INGESTION =====")

    print("Fichiers détectés :", len(files))
    print("Fichiers traités :", successful_files)
    print("Documents extraits :", total_documents)
    print("Total des chunks :", len(all_chunks))

    print("\nIngestion terminée !")


if __name__ == "__main__":
    ingest_all_documents()
