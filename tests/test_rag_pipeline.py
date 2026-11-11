
from pathlib import Path
import sys

# Ajouter le dossier principal du projet
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from app.rag_pipeline import ask_question


print("\n===== TEST DU PIPELINE RAG COMPLET =====")

# 1. Question de l'étudiant
question = "Qu'est-ce que le RAG ?"

print("\nQuestion :", question)

# 2. Exécuter le pipeline RAG
result = ask_question(
    query=question,
    top_k=5
)

# 3. Afficher la réponse
print("\n===== REPONSE DE EDU RESEARCH AI =====")

print(result["answer"])

# 4. Afficher les sources utilisées
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

print("\n===== TEST TERMINE =====")
