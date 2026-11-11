
from app.retrieval.embeddings import get_embedding_model

print("Chargement du modèle d'embeddings...")

# Charger le modèle
embeddings = get_embedding_model()

# Transformer une phrase en vecteur
text = "L'intelligence artificielle permet aux machines d'apprendre."

vector = embeddings.embed_query(text)

print("\n===== RESULTAT =====")
print("Texte :", text)
print("Nombre de dimensions :", len(vector))
print("Premiers nombres :", vector[:10])

print("\nModèle d'embeddings opérationnel !")
