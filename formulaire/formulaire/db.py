
# ===============================================
# BUILD CHROMA DB POUR VISION NOUVELLE
# ===============================================
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
PDF_FOLDER = "C:/Users/farya/Downloads/Vision Nouvelle/formulaire/PDF_Folder"  
PERSIST_DIR = "./chroma_vision_nouvelle_db"    
         
# --- Étape 1 : Charger tous les fichiers PDF ---
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
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)
print(f"{len(chunks)} chunks générés")

# --- Étape 3 : Créer les embeddings ---
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
print("Embeddings créés avec le modèle BAAI/bge-m3")

# --- Étape 4 : Créer et persister la base Chroma ---
print("Création de la base vectorielle Chroma...")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="vision_nouvelle_docs",
    persist_directory=PERSIST_DIR
)