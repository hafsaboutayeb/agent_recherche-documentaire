
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Paramètres de découpage
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def split_documents(documents):
    """
    Découper les documents en petits morceaux.
    
    Taille : 800 tokens
    Chevauchement : 100 tokens
    """

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Ajouter un numéro à chaque morceau
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    print("Découpage terminé !")
    print(f"Nombre de documents initiaux : {len(documents)}")
    print(f"Nombre de chunks créés : {len(chunks)}")

    return chunks
