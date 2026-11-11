
from app.ingestion.document_loader import load_document

file_path = "data/raw/test.pdf"

print("===== TEST DU PDF =====")

documents = load_document(file_path)

for document in documents:
    print("\nCONTENU EXTRAIT :")
    print(document.page_content)

    print("\nMETADONNEES :")
    print(document.metadata)
