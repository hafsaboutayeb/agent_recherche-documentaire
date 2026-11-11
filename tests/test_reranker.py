
from pathlib import Path
import sys

# Permet à Python de retrouver le dossier principal du projet
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from app.retrieval.reranker import rerank_documents


print("\n===== TEST DU RERANKER BGE =====")

# Question à poser au système
question = "Qu'est-ce que le RAG ?"

# Création de trois passages de test
documents = [
    Document(
        page_content=(
            "L'intelligence artificielle permet aux machines "
            "d'effectuer des tâches nécessitant normalement "
            "l'intelligence humaine."
        ),
        metadata={"source": "document_ia.pdf"}
    ),

    Document(
        page_content=(
            "Le RAG (Retrieval-Augmented Generation) est une "
            "technique qui permet de rechercher des informations "
            "dans des documents pour générer des réponses "
            "plus pertinentes."
        ),
        metadata={"source": "document_rag.pdf"}
    ),

    Document(
        page_content=(
            "Python est un langage de programmation utilisé "
            "pour développer des applications, des sites web "
            "et des projets d'intelligence artificielle."
        ),
        metadata={"source": "document_python.pdf"}
    )
]

print("\nQuestion :", question)
print("\nNombre de passages avant Reranking :", len(documents))

# Effectuer le Reranking
results = rerank_documents(
    query=question,
    documents=documents,
    top_k=3
)

# Afficher les résultats classés
print("\n===== RESULTATS APRES RERANKING =====")

for index, document in enumerate(results, start=1):

    print(f"\n--- Résultat {index} ---")

    print("Source :", document.metadata["source"])

    print("Score :", document.metadata["rerank_score"])

    print("Contenu :", document.page_content)

# Vérifier le classement des scores
scores = [
    document.metadata["rerank_score"]
    for document in results
]

assert scores == sorted(scores, reverse=True), (
    "Erreur : les résultats ne sont pas triés correctement."
)

print("\nTest du Reranker réussi !")
