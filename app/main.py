"""
Travel Africa RAG Assistant - FastAPI Backend
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

load_dotenv()

app = FastAPI(title="Travel Africa RAG Assistant", version="1.0.0")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

BASE = r'C:\Users\User\Desktop\travel-africa-rag'

print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=f'{BASE}/chroma_db')
collection = chroma_client.get_collection("travel_africa")

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
    return {
        "status": "healthy",
        "documents_in_db": collection.count(),
        "version": "1.0.0"
    }


@app.post("/ask")
def ask(payload: Question):
    query_embedding = embedding_model.encode([payload.question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )

    context_docs = results['documents'][0]
    metadatas = results['metadatas'][0]

    context = "\n\n".join([f"- {doc}" for doc in context_docs])

    prompt = f"""You are a knowledgeable and friendly Travel Africa assistant helping users discover hotels, attractions, transport options and travel tips across Kenya and East Africa.

Use the following retrieved information to answer the user's question. Be helpful, specific and practical.

Retrieved Information:
{context}

User Question: {payload.question}

Provide a clear, helpful answer based on the retrieved information. If recommending hotels mention the price range and key amenities. If discussing transport mention the cost and how to use it. If warning about scams be specific about how to avoid them."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )

    answer = response.choices[0].message.content.strip()

    sources = []
    for meta in metadatas:
        source = {}
        if 'hotel_name' in meta:
            source['name'] = meta['hotel_name']
            source['type'] = 'Hotel'
        elif 'name' in meta:
            source['name'] = meta['name']
            source['type'] = meta.get('type', meta.get('transport_type', 'Information'))
        source['location'] = meta.get('location', meta.get('city', ''))
        source['country'] = meta.get('country', '')
        sources.append(source)

    return {"answer": answer, "sources": sources}


@app.get("/hotels")
def get_hotels():
    import pandas as pd
    hotels = pd.read_csv(f'{BASE}/data/cleaned/hotels_cleaned.csv')
    return hotels[['hotel_name', 'location', 'country', 'category', 'price_range', 'rating']].to_dict('records')


@app.get("/hotels/{location}")
def get_hotels_by_location(location: str):
    import pandas as pd
    hotels = pd.read_csv(f'{BASE}/data/cleaned/hotels_cleaned.csv')
    filtered = hotels[hotels['location'].str.lower() == location.lower()]
    return filtered[['hotel_name', 'location', 'country', 'category', 'price_range', 'rating']].to_dict('records')


@app.post("/plan-trip")
def plan_trip(payload: TripRequest):
    query = f"hotels and attractions in {payload.destination} for {payload.interests} travel"
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=8
    )

    context = "\n\n".join([f"- {doc}" for doc in results['documents'][0]])

    prompt = f"""You are an expert East Africa travel planner.

Create a detailed {payload.duration_days}-day itinerary for {payload.destination}.
Traveller interests: {payload.interests}
Budget level: {payload.budget}

Use this information about the destination:
{context}

Create a day by day itinerary with:
- Morning afternoon and evening activities
- Recommended hotels with price range
- Transport options between locations
- Practical tips and warnings
- Estimated daily budget"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
    )

    return {"itinerary": response.choices[0].message.content.strip()}

