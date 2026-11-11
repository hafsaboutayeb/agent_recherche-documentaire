
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.hybrid_search import hybrid_search


question = "Qu'est-ce que le RAG ?"

print("\n===== BENCHMARK DE RECHERCHE =====")

times = []

for iteration in range(1, 4):

    print(f"\n===== RECHERCHE {iteration} =====")

    start = time.perf_counter()

    results = hybrid_search(
        query=question,
        k=5
    )

    elapsed = time.perf_counter() - start

    times.append(elapsed)

    print("Nombre de résultats :", len(results))
    print(f"Temps : {elapsed:.2f} secondes")


# Première recherche : chargement des modèles inclus
cold_time = times[0]

# Recherches suivantes : modèles déjà chargés
warm_times = times[1:]

average_warm_time = sum(warm_times) / len(warm_times)

print("\n===== RESULTATS DU BENCHMARK =====")

print(f"Premier lancement : {cold_time:.2f} secondes")
print(f"Temps moyen après chargement : {average_warm_time:.2f} secondes")

print("\nObjectif : moins de 8 secondes")

if average_warm_time < 8:
    print("OBJECTIF ATTEINT pour la recherche après chargement.")
else:
    print("OBJECTIF NON ATTEINT : des optimisations supplémentaires seront nécessaires.")

print("\n===== FIN DU BENCHMARK =====")
