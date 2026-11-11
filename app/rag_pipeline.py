
from langchain_core.documents import Document

from app.retrieval.hybrid_search import hybrid_search
from app.generation.llm import (
    generate_answer,
    ABSTENTION_MESSAGE
)


def ask_question(query: str, top_k: int = 5) -> dict:
    """
    Exécute le pipeline RAG complet.

    Étapes :
    1. Recherche hybride.
    2. Reranking.
    3. Génération de la réponse avec OpenAI.
    4. Récupération des sources.

    Retourne un dictionnaire contenant :
    - question
    - answer
    - sources
    """

    # 1. Vérifier la question
    if not query or not query.strip():
        raise ValueError("La question ne doit pas être vide.")

    if top_k < 1:
        raise ValueError("top_k doit être supérieur ou égal à 1.")

    print("\n===== DEMARRAGE DU PIPELINE RAG =====")

    # 2. Recherche hybride + Reranking
    print("\n1. Recherche des documents pertinents...")

    retrieved_documents = hybrid_search(
        query=query,
        k=top_k
    )

    print(
        f"Nombre de passages retenus : "
        f"{len(retrieved_documents)}"
    )

    # 3. Vérifier si des documents ont été retrouvés
    if not retrieved_documents:
        return {
            "question": query,
            "answer": ABSTENTION_MESSAGE,
            "sources": []
        }

    # 4. Générer la réponse avec OpenAI
    print("\n2. Generation de la reponse avec OpenAI...")

    answer = generate_answer(
        query=query,
        documents=retrieved_documents
    )

    # 5. Préparer les sources utilisées
    sources = []

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):
        metadata = document.metadata or {}

        source_information = {
            "citation": f"[{index}]",
            "source": metadata.get(
                "source",
                "Source inconnue"
            ),
            "page": metadata.get("page"),
            "chunk_index": metadata.get("chunk_index"),
            "rerank_score": metadata.get("rerank_score")
        }

        sources.append(source_information)

    print("\n3. Pipeline RAG termine !")

    # 6. Retourner le résultat final
    return {
        "question": query,
        "answer": answer,
        "sources": sources
    }
