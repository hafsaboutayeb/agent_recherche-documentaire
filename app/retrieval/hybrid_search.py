
import re
import json
import hashlib

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from app.retrieval.reranker import rerank_documents

from app.retrieval.vector_store import get_vector_store


# Nombre de résultats à récupérer pour chaque méthode
CANDIDATE_K = 20

# Constante utilisée dans la formule RRF
RRF_CONSTANT = 60


def tokenize_text(text: str) -> list[str]:
    """
    Découpe un texte en mots.
    Exemple :
    'Le RAG utilise des documents'
    devient :
    ['le', 'rag', 'utilise', 'des', 'documents']
    """
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def get_document_id(document: Document) -> str:
    """
    Crée un identifiant unique à partir du contenu
    et des métadonnées du document.
    """
    metadata = json.dumps(
        document.metadata,
        sort_keys=True,
        ensure_ascii=False,
        default=str
    )

    content = metadata + document.page_content

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def reciprocal_rank_fusion(
    dense_results: list[Document],
    bm25_results: list[Document],
    k: int = 5
) -> list[Document]:
    """
    Fusionne les résultats Dense et BM25 avec la méthode RRF.
    """

    scores = {}
    documents = {}
    methods = {}

    # Parcourir les deux listes de résultats
    rankings = {
        "Dense": dense_results,
        "BM25": bm25_results
    }

    for method_name, results in rankings.items():

        for rank, document in enumerate(results, start=1):

            document_id = get_document_id(document)

            # Ajouter le document s'il n'existe pas encore
            documents[document_id] = document

            # Calculer le score RRF
            score = 1 / (RRF_CONSTANT + rank)

            scores[document_id] = (
                scores.get(document_id, 0) + score
            )

            # Garder une trace des méthodes qui ont trouvé le document
            methods.setdefault(document_id, set()).add(method_name)

    # Trier les documents par score RRF décroissant
    sorted_ids = sorted(
        scores,
        key=lambda doc_id: scores[doc_id],
        reverse=True
    )

    # Construire la liste finale
    fused_results = []

    for document_id in sorted_ids[:k]:

        original_document = documents[document_id]

        metadata = dict(original_document.metadata)

        metadata["rrf_score"] = round(
            scores[document_id], 6
        )

        metadata["retrieval_methods"] = sorted(
            methods[document_id]
        )

        fused_document = Document(
            page_content=original_document.page_content,
            metadata=metadata
        )

        fused_results.append(fused_document)

    return fused_results


def hybrid_search(query: str, k: int = 5) -> list[Document]:
    """
    Effectue une recherche hybride :
    1. Recherche Dense avec ChromaDB.
    2. Recherche lexicale avec BM25.
    3. Fusion des résultats avec RRF.
    """

    if not query or not query.strip():
        raise ValueError("La question ne doit pas être vide.")

    # 1. Récupérer la base vectorielle
    vector_store = get_vector_store()

    # 2. Effectuer la recherche Dense
    dense_results = vector_store.similarity_search(
        query,
        k=CANDIDATE_K
    )

    # 3. Récupérer tous les documents enregistrés dans ChromaDB
    stored_data = vector_store.get(
        include=["documents", "metadatas"]
    )

    contents = stored_data.get("documents") or []
    metadatas = stored_data.get("metadatas") or []

    all_documents = []

    for content, metadata in zip(contents, metadatas):

        if content is not None:
            document = Document(
                page_content=content,
                metadata=metadata or {}
            )

            all_documents.append(document)

    if not all_documents:
        print("Aucun document disponible pour la recherche.")
        return []

    # 4. Construire le moteur BM25
    bm25_retriever = BM25Retriever.from_documents(
        all_documents,
        preprocess_func=tokenize_text,
        k=CANDIDATE_K
    )

    # 5. Effectuer la recherche BM25
    bm25_results = bm25_retriever.invoke(query)

    # 6. Fusionner les résultats avec RRF
    fused_candidates = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        k=CANDIDATE_K
    )

    # 7. Réorganiser les résultats avec le Reranker BGE
    final_results = rerank_documents(
        query=query,
        documents=fused_candidates,
        top_k=k
    )

    return final_results
