
from pathlib import Path
import sys

# Ajouter le dossier principal du projet au chemin Python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from app.generation.llm import generate_answer, MODEL_NAME


print("\n===== TEST DE GENERATION LLM =====")

# 1. Définir une question
question = "Qu'est-ce que le RAG et à quoi sert-il ?"

print("\nModèle utilisé :", MODEL_NAME)
print("Question :", question)

# 2. Préparer un document de test
documents = [
    Document(
        page_content=(
            "Le RAG (Retrieval-Augmented Generation) est une "
            "technique qui combine la recherche d'informations "
            "dans des documents avec la génération de texte "
            "par un modèle de langage. Il permet de fournir "
            "des réponses basées sur des sources documentaires."
        ),
        metadata={
            "source": "test_rag.pdf",
            "page": 1
        }
    )
]

# 3. Générer la réponse avec OpenAI
print("\nEnvoi de la question à OpenAI...")

answer = generate_answer(
    query=question,
    documents=documents
)

# 4. Afficher la réponse
print("\n===== REPONSE DE EDU RESEARCH AI =====")
print(answer)

print("\n===== TEST TERMINE =====")
