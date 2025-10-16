# ===============================================
# BUILD QDRANT DB POUR VISION NOUVELLE
# ===============================================
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

# --- Configuration ---
PDF_FOLDER = "C:\\Users\\farya\\Downloads\\Vision Nouvelle\\Assistant-Vision-Nouvelle\\formulaire\\PDF_Folder"

# Remplace ces valeurs par celles de ton Qdrant Cloud
QDRANT_URL = "https://f1e83af3-4124-477e-aea0-758aa1a90037.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.cfXICPDA1MesK8UgN9MBdg07IhliD3uochOGitDptTI"
COLLECTION_NAME = "rag_visionAI"

# --- Étape 1 : Charger les fichiers PDF ---
documents = []
for filename in os.listdir(PDF_FOLDER):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, filename)
        print(f"Chargement de {filename}...")
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Erreur lors du chargement de {filename} : {e}")

print(f"\n{len(documents)} pages chargées depuis {PDF_FOLDER}")

# --- Étape 2 : Découper le texte en chunks ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=850,
    chunk_overlap=120,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"{len(chunks)} chunks générés")

# --- Étape 3 : Créer les embeddings ---
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
print("Embeddings créés avec le modèle BAAI/bge-m3")

# --- Étape 4 : Connexion à Qdrant Cloud ---
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# (Optionnel) recrée la collection proprement
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "size": 1024,  # dépend du modèle d'embedding (BAAI/bge-m3 = 1024)
        "distance": "Cosine"
    }
)
print("Collection Qdrant initialisée ✅")
# --- Étape 5 : Envoyer les embeddings dans Qdrant Cloud ---
print("Envoi des embeddings dans Qdrant Cloud...")
vector_db = Qdrant.from_documents(
    documents=chunks,
    embedding=embedding,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=COLLECTION_NAME,
    batch_size=16 
)

print("✅ Base vectorielle Qdrant créée avec succès !")
