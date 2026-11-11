
from langchain_core.documents import Document
from app.ingestion.text_splitter import split_documents
import tiktoken

# Créer un texte assez long pour produire plusieurs chunks
paragraph = """
L'intelligence artificielle est un domaine informatique
qui permet de développer des systèmes capables d'effectuer
des tâches complexes. Le RAG combine la recherche
documentaire avec les modèles de langage. Il permet
de retrouver des informations pertinentes dans une base
documentaire et de produire des réponses contextualisées.
"""

long_text = paragraph * 100

# Créer un document LangChain
document = Document(
    page_content=long_text,
    metadata={
        "source": "test_long.txt",
        "file_type": "txt"
    }
)

# Effectuer le découpage
chunks = split_documents([document])

# Compter les tokens
encoding = tiktoken.get_encoding("cl100k_base")

print("\n===== VERIFICATION DES CHUNKS =====")

for i, chunk in enumerate(chunks):
    token_count = len(encoding.encode(chunk.page_content))

    print(f"\nChunk {i + 1}")
    print("Nombre de tokens :", token_count)
    print("Métadonnées :", chunk.metadata)
    print("Début du contenu :", chunk.page_content[:150])

print("\n===== BILAN =====")
print("Nombre total de chunks :", len(chunks))
