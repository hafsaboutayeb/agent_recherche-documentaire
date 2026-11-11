
from pathlib import Path
import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.retrieval.embeddings import get_embedding_model


# Emplacement de la base vectorielle
CHROMA_PATH = "vectorstore/chroma_db"

# Nom de la collection
COLLECTION_NAME = "edu_research_onnx"


def get_vector_store():
    """
    Créer ou ouvrir la base vectorielle ChromaDB.
    """

    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vector_store


def add_documents_to_store(
    documents: list[Document]
):
    """
    Ajouter les chunks dans ChromaDB.
    """

    if not documents:
        print("Aucun document à enregistrer.")
        return

    vector_store = get_vector_store()

    # Créer des identifiants stables pour éviter
    # d'ajouter plusieurs fois les mêmes chunks.
    ids = []

    for document in documents:
        unique_content = (
            str(document.metadata)
            + document.page_content
        )

        document_id = hashlib.sha256(
            unique_content.encode("utf-8")
        ).hexdigest()

        ids.append(document_id)

    vector_store.add_documents(
        documents=documents,
        ids=ids
    )

    print("Enregistrement terminé !")
    print(f"Nombre de chunks traités : {len(documents)}")

    return vector_store


def search_documents(
    query: str,
    k: int = 5
):
    """
    Rechercher les chunks les plus proches
    d'une question.
    """

    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results
