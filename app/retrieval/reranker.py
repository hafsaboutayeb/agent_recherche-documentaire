
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_core.documents import Document


# Modèle multilingue BGE
MODEL_NAME = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def get_reranker() -> TextCrossEncoder:
    """
    Charger le Reranker ONNX une seule fois.

    FastEmbed utilise ONNX Runtime au lieu de PyTorch.
    """

    print("Chargement du modèle BGE Reranker ONNX...")

    reranker = TextCrossEncoder(
        model_name=MODEL_NAME
    )

    print("Modèle BGE Reranker ONNX chargé avec succès !")

    return reranker


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = 5
) -> list[Document]:
    """
    Réorganiser les documents selon leur pertinence
    par rapport à la question.

    Le classement est effectué avec ONNX Runtime.
    """

    if not query or not query.strip():
        raise ValueError("La question ne doit pas être vide.")

    if top_k < 1:
        raise ValueError("top_k doit être supérieur ou égal à 1.")

    if not documents:
        print("Aucun document à réordonner.")
        return []

    # 1. Préparer les textes des documents
    passages = [
        document.page_content
        for document in documents
    ]

    # 2. Charger le modèle
    reranker = get_reranker()

    # 3. Calculer les scores de pertinence
    scores = list(
        reranker.rerank(
            query=query,
            documents=passages
        )
    )

    if len(scores) != len(documents):
        raise RuntimeError(
            "Le nombre de scores ne correspond pas "
            "au nombre de documents."
        )

    # 4. Trier les documents par score décroissant
    ranked_indices = sorted(
        range(len(documents)),
        key=lambda index: float(scores[index]),
        reverse=True
    )

    # 5. Sélectionner les meilleurs documents
    reranked_documents = []

    for rank, index in enumerate(
        ranked_indices[:top_k],
        start=1
    ):

        original_document = documents[index]

        # Conserver les métadonnées
        metadata = dict(original_document.metadata)

        metadata["rerank_score"] = round(
            float(scores[index]), 6
        )

        metadata["rerank_rank"] = rank

        reranked_document = Document(
            page_content=original_document.page_content,
            metadata=metadata
        )

        reranked_documents.append(reranked_document)

    print("Reranking ONNX terminé !")
    print("Documents reçus :", len(documents))
    print(
        "Documents sélectionnés :",
        len(reranked_documents)
    )

    return reranked_documents
