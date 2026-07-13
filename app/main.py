"""
Travel Africa RAG Assistant - FastAPI Backend
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import pandas as pd
import os

load_dotenv()

app = FastAPI(title="Travel Africa RAG Assistant", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded")

CHROMA_PATH = os.path.join(BASE, 'chroma_db')
print(f"Connecting to ChromaDB at {CHROMA_PATH}...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_or_build_collection():
    existing = [c.name for c in chroma_client.list_collections()]
    if "travel_africa" in existing:
        print("ChromaDB collection found")
        return chroma_client.get_collection("travel_africa")
    
    print("Collection not found — building from CSV files...")
    collection = chroma_client.get_or_create_collection(
        name="travel_africa",
        metadata={"hnsw:space": "cosine"}
    )
    
    datasets = [
        ('hotels', ['hotel_name','location','country','category','price_range','rating','source_url']),
        ('attractions', ['name','location','country','type','price_range']),
        ('transport', ['name','city','country','transport_type','approximate_cost']),
        ('scams', ['scam_name','location','country']),
        ('malls', ['name','location','country','opening_hours']),
    ]
    
    for name, meta_cols in datasets:
        path = os.path.join(BASE, 'data', 'cleaned', f'{name}_cleaned.csv')
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        texts = df['combined_text'].tolist()
        ids = [f"{name}_{i}" for i in range(len(texts))]
        available_cols = [c for c in meta_cols if c in df.columns]
        metadatas = df[available_cols].fillna('').to_dict('records')
        embeddings = embedding_model.encode(texts, show_progress_bar=False).tolist()
        collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
        print(f"Added {len(texts)} {name}")
    
    return collection

collection = get_or_build_collection()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("All systems ready")


class Question(BaseModel):
    question: str

class TripRequest(BaseModel):
    destination: str
    duration_days: int
    interests: str
    budget: str


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health():
    return {"status": "healthy", "documents": collection.count()}


@app.post("/ask")
def ask(payload: Question):
    query_embedding = embedding_model.encode([payload.question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5)
    context = "\n\n".join([f"- {doc}" for doc in results['documents'][0]])
    metadatas = results['metadatas'][0]

    prompt = f"""You are a warm, knowledgeable and friendly Travel Africa assistant helping users discover hotels, attractions, transport and travel tips across Kenya and East Africa.

Use the retrieved information below to answer the question. Be specific, helpful and practical. Write in a friendly conversational tone.

Retrieved Information:
{context}

User Question: {payload.question}

Give a clear helpful answer. Mention prices, tips and practical details where relevant."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )

    sources = []
    for meta in metadatas:
        source = {}
        source['name'] = meta.get('hotel_name') or meta.get('name') or meta.get('scam_name', 'Info')
        source['location'] = meta.get('location') or meta.get('city', '')
        source['country'] = meta.get('country', '')
        source['type'] = meta.get('category') or meta.get('type') or meta.get('transport_type', 'Travel Info')
        sources.append(source)

    return {"answer": response.choices[0].message.content.strip(), "sources": sources}


@app.get("/hotels")
def get_hotels():
    path = os.path.join(BASE, 'data', 'cleaned', 'hotels_cleaned.csv')
    hotels = pd.read_csv(path)
    return hotels[['hotel_name','location','country','category','price_range','rating']].to_dict('records')


@app.get("/hotels/{location}")
def get_hotels_by_location(location: str):
    path = os.path.join(BASE, 'data', 'cleaned', 'hotels_cleaned.csv')
    hotels = pd.read_csv(path)
    filtered = hotels[hotels['location'].str.lower() == location.lower()]
    return filtered[['hotel_name','location','country','category','price_range','rating']].to_dict('records')


@app.post("/plan-trip")
def plan_trip(payload: TripRequest):
    query = f"hotels attractions things to do in {payload.destination} for {payload.interests}"
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=8)
    context = "\n\n".join([f"- {doc}" for doc in results['documents'][0]])

    prompt = f"""You are an expert East Africa travel planner with deep knowledge of Kenya, Tanzania, Uganda and Zanzibar.

Create a detailed {payload.duration_days}-day itinerary for {payload.destination}.
Traveller interests: {payload.interests}
Budget level: {payload.budget}

Use this destination information:
{context}

Write a day by day itinerary with morning, afternoon and evening activities. Include recommended hotels with price range, transport options, practical tips and estimated daily budget. Be specific and enthusiastic."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
    )

    return {"itinerary": response.choices[0].message.content.strip()}