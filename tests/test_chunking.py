
from app.ingestion.document_loader import load_document
from app.ingestion.text_splitter import split_documents

# Charger le PDF
documents = load_document("data/raw/test.pdf")

# Découper le contenu
chunks = split_documents(documents)

print("\n===== RESULTATS DU DECOUPAGE =====")

for index, chunk in enumerate(chunks):
    print(f"\nCHUNK {index + 1}")
    print("Contenu :", chunk.page_content)
    print("Métadonnées :", chunk.metadata)
