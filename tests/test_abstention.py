
from pathlib import Path
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from app.rag_pipeline import ask_question
from app.generation.llm import ABSTENTION_MESSAGE


print("\n===== TEST ANTI-HALLUCINATION =====")

# Question dont la réponse n'est pas dans le PDF
question = "Quelle est la capitale du Japon ?"

print("\nQuestion :", question)

# Exécuter le pipeline RAG
result = ask_question(
    query=question,
    top_k=5
)

answer = result["answer"].strip()

print("\n===== REPONSE DU SYSTEME =====")
print(answer)

# Vérifier le comportement du système
print("\n===== VERIFICATION =====")

if answer == ABSTENTION_MESSAGE:
    print("TEST REUSSI : le système a correctement refuse de répondre.")
else:
    print("ATTENTION : le système n'a pas utilisé la phrase d'abstention exacte.")

print("\n===== FIN DU TEST =====")
