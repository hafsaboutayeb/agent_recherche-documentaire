
from app.ingestion.document_loader import load_document
from app.ingestion.text_splitter import split_documents

from app.retrieval.vector_store import (
    add_documents_to_store,
    search_documents
)


# 1. Charger le document PDF
print("\n===== 1. CHARGEMENT DU DOCUMENT =====")

documents = load_document("data/raw/test.pdf")


# 2. Découper le document
print("\n===== 2. DECOUPAGE =====")

chunks = split_documents(documents)


# 3. Enregistrer les chunks dans ChromaDB
print("\n===== 3. ENREGISTREMENT CHROMADB =====")

add_documents_to_store(chunks)


# 4. Effectuer une recherche
print("\n===== 4. RECHERCHE DOCUMENTAIRE =====")

question = "Qu'est-ce que le RAG ?"

results = search_documents(question, k=3)


# 5. Afficher les résultats
print("\nQuestion :", question)

for index, result in enumerate(results, start=1):

    print(f"\nRésultat {index}")
    print("Contenu :", result.page_content)
    print("Source :", result.metadata)
