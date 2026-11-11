
from pathlib import Path
import sys

# Ajouter le dossier principal du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag_pipeline import ask_question


print("\n===== TEST RAG MULTI-DOCUMENTS =====")

# 1. Poser une question
question = (
    "Qu'est-ce que le RAG et comment combine-t-il "
    "la recherche documentaire et la generation de texte ?"
)

print("\nQuestion :", question)

# 2. Exécuter le pipeline RAG
result = ask_question(
    query=question,
    top_k=5
)

# 3. Afficher la réponse
print("\n===== REPONSE DE EDU RESEARCH AI =====")

print(result["answer"])

# 4. Afficher les sources
print("\n===== SOURCES UTILISEES =====")

if result["sources"]:

    for source in result["sources"]:

        print("\nCitation :", source["citation"])
        print("Document :", source["source"])
        print("Page :", source["page"])
        print("Chunk :", source["chunk_index"])
        print("Score Reranking :", source["rerank_score"])

else:
    print("Aucune source trouvée.")

# 5. Afficher le bilan
print("\n===== BILAN DU TEST =====")

print("Question envoyée :", result["question"])
print("Nombre de sources utilisées :", len(result["sources"]))

if result["sources"]:
    print("TEST TERMINE : réponse générée avec des sources.")
else:
    print("Aucune source disponible pour cette question.")

print("\n===== FIN DU TEST =====")
