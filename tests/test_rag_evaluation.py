
from pathlib import Path
import sys
import time

# Ajouter le dossier principal du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.hybrid_search import hybrid_search


# =====================================================
# 1. CONFIGURATION DE L'EVALUATION
# =====================================================

TOP_K = 5

# Question et documents pertinents connus
EVALUATION_CASES = [
    {
        "question": (
            "Qu'est-ce que le RAG et comment combine-t-il "
            "la recherche documentaire et la génération de texte ?"
        ),
        "expected_sources": [
            "data/raw/test.docx",
            "data/raw/test.txt",
            "data/raw/test.md",
            "data/raw/test.pdf",
            "data/raw/test.csv"
        ]
    }
]


def normalize_source(source: str) -> str:
    """Normalise les chemins pour Windows."""
    return str(source).replace("/", "\\").lower()


# =====================================================
# 2. EVALUATION DU RECALL@5
# =====================================================

def evaluate_recall():

    print("\n===== EVALUATION RECALL@5 =====")

    total_relevant = 0
    total_retrieved = 0
    total_time = 0

    for case_number, case in enumerate(
        EVALUATION_CASES,
        start=1
    ):

        question = case["question"]

        expected_sources = {
            normalize_source(source)
            for source in case["expected_sources"]
        }

        print(f"\n--- Question {case_number} ---")
        print("Question :", question)

        # Mesurer le temps de recherche
        start_time = time.perf_counter()

        results = hybrid_search(
            query=question,
            k=TOP_K
        )

        elapsed_time = time.perf_counter() - start_time

        total_time += elapsed_time

        # Récupérer les sources retrouvées
        retrieved_sources = {
            normalize_source(
                document.metadata.get("source", "")
            )
            for document in results
        }

        # Calculer les documents pertinents retrouvés
        relevant_found = (
            expected_sources.intersection(retrieved_sources)
        )

        recall = (
            len(relevant_found) / len(expected_sources)
            if expected_sources
            else 0
        )

        total_relevant += len(expected_sources)
        total_retrieved += len(relevant_found)

        print("\nSources attendues :", len(expected_sources))
        print("Sources retrouvées :", len(relevant_found))

        print(
            "Recall@5 :",
            f"{recall * 100:.2f}%"
        )

        print(
            "Temps de recherche :",
            f"{elapsed_time:.2f} secondes"
        )

        print("\nSources retrouvées :")

        for source in sorted(retrieved_sources):
            print("-", source)

        print("\nSources pertinentes retrouvées :")

        for source in sorted(relevant_found):
            print("-", source)

    # =================================================
    # 3. RESULTATS GLOBAUX
    # =================================================

    global_recall = (
        total_retrieved / total_relevant
        if total_relevant
        else 0
    )

    print("\n===== RESULTATS GLOBAUX =====")

    print("Recall@5 global :", f"{global_recall * 100:.2f}%")

    print("Temps moyen de recherche :",
          f"{total_time / len(EVALUATION_CASES):.2f} secondes")

    print("\n===== FIN DE L'EVALUATION =====")


if __name__ == "__main__":
    evaluate_recall()
