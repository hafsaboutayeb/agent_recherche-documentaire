
from pathlib import Path
import sys
import time

# Ajouter le dossier principal du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag_pipeline import ask_question


# Question présente dans nos documents
QUESTION = (
    "Qu'est-ce que le RAG et comment combine-t-il "
    "la recherche documentaire et la génération de texte ?"
)

print("\n===== BENCHMARK PIPELINE RAG COMPLET =====")

times = []

for run in range(1, 3):

    print(f"\n===== EXECUTION {run} =====")

    start_time = time.perf_counter()

    # Exécuter tout le pipeline :
    # Recherche + Reranking + Génération OpenAI
    result = ask_question(
        query=QUESTION,
        top_k=5
    )

    elapsed = time.perf_counter() - start_time

    times.append(elapsed)

    print("\nRéponse générée :", bool(result["answer"]))
    print("Nombre de sources :", len(result["sources"]))

    print(f"Temps total : {elapsed:.2f} secondes")


# Calculer les statistiques
first_time = times[0]
warm_time = times[1]

print("\n===== RESULTATS DU BENCHMARK =====")

print(f"Premier lancement : {first_time:.2f} secondes")
print(f"Deuxième lancement : {warm_time:.2f} secondes")

print("\nObjectif : moins de 8 secondes")

if warm_time < 8:
    print("OBJECTIF ATTEINT pour le pipeline après chargement.")
else:
    print("OBJECTIF NON ATTEINT : optimisation nécessaire.")

print("\n===== FIN DU BENCHMARK =====")
