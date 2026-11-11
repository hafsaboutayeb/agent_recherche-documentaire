from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval.hybrid_search import hybrid_search


print("\n===== TEST DE LA RECHERCHE HYBRIDE =====")

question = "Qu'est-ce que le RAG ?"

print("\nQuestion :", question)

results = hybrid_search(question, k=5)

print("\n===== RESULTATS FINAUX =====")
print("Nombre de résultats :", len(results))

for index, document in enumerate(results, start=1):

    print(f"\n--- Résultat {index} ---")

    print("Contenu :", document.page_content)

    print("Source :", document.metadata.get("source"))

    print("Score RRF :", document.metadata.get("rrf_score"))

    print(
        "Méthodes :",
        document.metadata.get("retrieval_methods")
    )
print("Source :", document.metadata.get("source"))

print("Score RRF :", document.metadata.get("rrf_score"))

print("Score Reranking :", document.metadata.get("rerank_score"))

print("Rang Reranking :", document.metadata.get("rerank_rank"))

print("Méthodes :", document.metadata.get("retrieval_methods"))