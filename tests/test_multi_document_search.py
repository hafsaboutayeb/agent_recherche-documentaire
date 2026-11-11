
from pathlib import Path
import sys

# Ajouter le dossier principal du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.vector_store import get_vector_store
from app.retrieval.hybrid_search import hybrid_search


print("\n===== TEST MULTI-DOCUMENTS =====")

# =========================================
# 1. Vérifier le contenu de ChromaDB
# =========================================

print("\n===== 1. CONTENU DE CHROMADB =====")

vector_store = get_vector_store()

stored_data = vector_store.get(
    include=["documents", "metadatas"]
)

documents = stored_data.get("documents") or []
metadatas = stored_data.get("metadatas") or []

print("Nombre total de chunks enregistrés :", len(documents))

# Afficher les différentes sources
sources = sorted({
    metadata.get("source", "Source inconnue")
    for metadata in metadatas
    if metadata
})

print("\nSources présentes dans ChromaDB :")

for source in sources:
    print("-", source)

print("\nNombre de sources différentes :", len(sources))


# =========================================
# 2. Tester la recherche hybride
# =========================================

print("\n===== 2. RECHERCHE HYBRIDE =====")

question = "Qu'est-ce que le RAG ?"

print("\nQuestion :", question)

results = hybrid_search(
    query=question,
    k=5
)

print("\n===== 3. RESULTATS DE RECHERCHE =====")

print("Nombre de résultats :", len(results))

for index, document in enumerate(results, start=1):

    print(f"\n--- Résultat {index} ---")

    print("Source :", document.metadata.get("source"))

    print("Score RRF :", document.metadata.get("rrf_score"))

    print("Score Reranking :", document.metadata.get("rerank_score"))

    print("Contenu :", document.page_content[:300])


# =========================================
# 3. Bilan
# =========================================

print("\n===== 4. BILAN =====")

if documents and results:
    print("TEST REUSSI : ChromaDB et la recherche hybride fonctionnent.")
else:
    print("ATTENTION : aucun document ou résultat trouvé.")

print("\n===== FIN DU TEST =====")
