
# --- Imports LangChain & Gemini ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os

# =====================================================
# CONFIGURATION INITIALE DU PIPELINE RAG
# =====================================================
os.environ["GOOGLE_API_KEY"] = "AIzaSyBEY7fZj7fHBcYCsuB3_OIz22G61-gc91E"
# Base Chroma persistante
PERSIST_DIR = "./chroma_vision_nouvelle_db"

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vector_db = Chroma(
    collection_name="vision_nouvelle_docs",
    persist_directory=PERSIST_DIR,
    embedding_function=embedding
)
# Modèle Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# =====================================================
# PROMPT : Assistant de campagne persuasif
# =====================================================
prompt_template = """
Tu es un **assistant virtuel de la campagne Vision Nouvelle**, charismatique et convaincant.  
Rôle : répondre **clairement et brièvement (~120 caractères)**, de manière **optimiste et humaine**, en inspirant confiance et engagement.  
- Mets en avant la **vision positive, la transparence, l’innovation, la solidarité et l’espoir**.  
- Sois concret, accessible, jamais agressif.  
- Engage l’utilisateur : pose des questions, adapte-toi à ses réponses, encourage le dialogue.  
- Termine souvent par un **appel à l’action ou à la réflexion positive**.  

Informations de référence :
{context}

Message du citoyen :
{question}

Commence directement la réponse :
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)


# =====================================================
#  RAG PIPELINE : recherche + génération
# =====================================================
def retrieve_context(query: str, k: int = 3):
    """Recherche les k passages les plus pertinents dans la base Chroma."""
    results = vector_db.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

def rag_pipeline(query: str):
    """Pipeline RAG complet pour Vision Nouvelle."""
    context = retrieve_context(query)
    final_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(final_prompt)
    return response.content
