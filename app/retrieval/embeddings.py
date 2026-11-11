
from functools import lru_cache

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings


# Modèle multilingue utilisé dans EduResearch AI
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FastEmbedEmbeddingAdapter(Embeddings):
    """
    Adaptateur qui permet d'utiliser FastEmbed
    avec LangChain et ChromaDB.

    Le calcul des embeddings utilise ONNX Runtime,
    sans nécessiter PyTorch.
    """

    def __init__(self):
        self.model = TextEmbedding(
            model_name=MODEL_NAME
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Créer les embeddings des documents."""

        vectors = self.model.embed(texts)

        return [
            vector.tolist()
            for vector in vectors
        ]

    def embed_query(self, text: str) -> list[float]:
        """Créer l'embedding d'une question."""

        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedding_model() -> FastEmbedEmbeddingAdapter:
    """
    Charger le modèle une seule fois
    et le réutiliser dans toute l'application.
    """

    print("Chargement du modèle Embeddings ONNX...")

    embeddings = FastEmbedEmbeddingAdapter()

    print("Modèle Embeddings ONNX chargé avec succès !")

    return embeddings
