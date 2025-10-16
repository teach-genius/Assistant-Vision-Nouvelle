# =====================================================
# PIPELINE RAG – Vision Nouvelle (Gemini + Qdrant Cloud)
# =====================================================
import os
import tiktoken
from asgiref.sync import sync_to_async
from dotenv import load_dotenv
# --- LangChain & Qdrant ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import Qdrant

from qdrant_client import QdrantClient
from langchain.prompts import PromptTemplate

# =====================================================
# CONFIGURATION INITIALE DU PIPELINE RAG
# =====================================================
load_dotenv()

# Clés d'API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY or "AIzaSyBEY7fZj7fHBcYCsuB3_OIz22G61-gc91E"

QDRANT_URL = os.getenv("QDRANT_URL") or "https://f1e83af3-4124-477e-aea0-758aa1a90037.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.cfXICPDA1MesK8UgN9MBdg07IhliD3uochOGitDptTI"
COLLECTION_NAME = "rag_visionAI"


# --- Embeddings ---
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# --- Connexion à Qdrant Cloud ---
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

vector_db = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding,
)

# --- Modèle de génération ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# =====================================================
# GESTION DES TOKENS
# =====================================================
MAX_TOKENS_CONTEXT = 6000
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Compte le nombre de tokens dans un texte."""
    return len(encoding.encode(text))

def build_context_with_limit(docs, max_tokens=MAX_TOKENS_CONTEXT):
    """
    Concatène les chunks de texte tant que la limite de tokens n’est pas atteinte.
    """
    context = ""
    total_tokens = 0
    for doc in docs:
        chunk = doc.page_content
        chunk_tokens = count_tokens(chunk)
        if total_tokens + chunk_tokens > max_tokens:
            break
        context += "\n\n" + chunk
        total_tokens += chunk_tokens
    return context

# =====================================================
# PROMPT : Assistant de campagne persuasif
# =====================================================
prompt_template = """
Tu es VisionIA, l’assistant virtuel officiel et porte-voix du Bureau Exécutif Vision Nouvelle (Croire. Innover. Agir), une équipe jeune, dynamique et engagée dans les élections de l’Union des Étudiants et Stagiaires Gabonais au Maroc (UESGM), dont la devise est : Unité. Excellence. Réussite.

Ta mission :
En t'appuyant sur les Informations de référence ci-dessous, réponds clairement, brièvement (environ 200 caractères) et avec chaleur humaine, en inspirant confiance, engagement et optimisme.  

Règle importante :
- Si la réponse est présente dans les informations de référence, cite-la explicitement (noms, rôles, actions, projets, etc.).
- Si l’information est absente ou incomplète, dis-le avec transparence et invite l’utilisateur à consulter nos canaux officiels.

Style attendu :
- Positif, humain, inspirant et crédible.  
- Jamais agressif ni polémique.  
- Encourage le dialogue : pose des questions, relance l’utilisateur, crée une connexion émotionnelle.  
- Valorise toujours les valeurs : Vision, Transparence, Solidarité, Innovation et Espoir.  
- Termine souvent par un appel à l’action ou une phrase inspirante.

Contexte :
C’est la période électorale du Bureau Exécutif de l’UESGM.  
Tu représentes Vision Nouvelle, un mouvement d’unité et de renouveau pour la communauté gabonaise au Maroc.  
Ta mission est d’informer, de motiver et d’inspirer sans jamais t’écarter du contexte de la campagne.

Informations de référence :
{context}

Message du citoyen :
{question}

Commence directement ta réponse, avec un ton engageant et sincère :
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# =====================================================
#  FONCTIONS RAG ASYNCHRONES
# =====================================================
@sync_to_async
def retrieve_context(query: str, k: int = 5):
    """Recherche les k passages les plus pertinents dans la base Qdrant."""
    results = vector_db.similarity_search(query, k=k)
    limited_context = build_context_with_limit(results)
    return limited_context

@sync_to_async
def generate_response(final_prompt: str):
    """Appelle le modèle Gemini de manière non bloquante."""
    response = llm.invoke(final_prompt)
    return response.content

async def rag_pipeline(query: str):
    """Pipeline RAG complet asynchrone pour Vision Nouvelle."""
    context = await retrieve_context(query)
    final_prompt = prompt.format(context=context, question=query)
    response_content = await generate_response(final_prompt)
    return response_content
