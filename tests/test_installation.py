
import langchain_core
import chromadb
import streamlit
import fastapi
import pandas
import fastembed
import onnxruntime

print("========== VERIFICATION DE L'INSTALLATION ==========")

print("LangChain Core :", langchain_core.__version__)
print("ChromaDB :", chromadb.__version__)
print("Pandas :", pandas.__version__)
print("FastEmbed :", fastembed.__version__)
print("ONNX Runtime :", onnxruntime.__version__)

print("\nToutes les bibliotheques necessaires sont installees !")
print("Installation verifiee avec succes.")
