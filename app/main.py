from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
import os
import json
import re

load_dotenv()

app = FastAPI(title="Travel Africa RAG Assistant", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_all_data():
    dfs = {}
    files = ['hotels', 'attractions', 'transport', 'scams', 'malls']
    for name in files:
        path = os.path.join(BASE, 'data', 'cleaned', f'{name}_cleaned.csv')
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path)
            print(f"Loaded {name}: {len(dfs[name])} records")
    return dfs

print("Loading data...")
ALL_DATA = load_all_data()
print("Data loaded. Ready.")

def search_data(query: str, top_k: int = 6) -> list:
    query_lower = query.lower()
    keywords = re.findall(r'\b\w{3,}\b', query_lower)
    results = []
    for name, df in ALL_DATA.items():
        if 'combined_text' not in df.columns:
            continue
        for _, row in df.iterrows():
            text = str(row['combined_text']).lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                results.append((score, str(row['combined_text']), row.to_dict()))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


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
    total = sum(len(df) for df in ALL_DATA.values())
    return {"status": "healthy", "total_records": total}

@app.post("/ask")
def ask(payload: Question):
    results = search_data(payload.question)
    context = "\n\n".join([f"- {r[1]}" for r in results])
    metadatas = [r[2] for r in results]

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
    for meta in metadatas[:5]:
        source = {}
        source['name'] = meta.get('hotel_name') or meta.get('name') or meta.get('scam_name', 'Info')
        source['location'] = meta.get('location') or meta.get('city', '')
        source['country'] = meta.get('country', '')
        sources.append(source)

    return {"answer": response.choices[0].message.content.strip(), "sources": sources}

@app.get("/hotels")
def get_hotels():
    if 'hotels' not in ALL_DATA:
        return []
    hotels = ALL_DATA['hotels']
    return hotels[['hotel_name','location','country','category','price_range','rating']].to_dict('records')

@app.get("/hotels/{location}")
def get_hotels_by_location(location: str):
    if 'hotels' not in ALL_DATA:
        return []
    hotels = ALL_DATA['hotels']
    filtered = hotels[hotels['location'].str.lower() == location.lower()]
    return filtered[['hotel_name','location','country','category','price_range','rating']].to_dict('records')

@app.post("/plan-trip")
def plan_trip(payload: TripRequest):
    query = f"hotels attractions things to do in {payload.destination} {payload.interests}"
    results = search_data(query, top_k=8)
    context = "\n\n".join([f"- {r[1]}" for r in results])

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