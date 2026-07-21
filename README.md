# WanderWithNicole — AI Travel Assistant

> Your AI travel bestie for Kenya & East Africa. Helps you experience East Africa stress free, especially when it comes to avoiding scams and getting robbed which can tank your vacation experience in general.
> Guiding thought for me was what would I want from an assistant if I was to travel to a whole new area, and I came up with the kind of data that would help build that. Below is a little more about what went into it and how you can also follow the same to build your own assistant.
> Built by **Nicole Onyango** · Powered by Groq AI & FastAPI

**Live demo:** https://wanderwithnicole-mr4y.onrender.com

---

## What it does

WanderWithNicole is a RAG-powered travel assistant that helps you discover hotels, plan trips, navigate cities and avoid scams across Kenya and East Africa.

Ask it anything:
- *"Best luxury safari lodges in Maasai Mara"*
- *"How do I get around Nairobi safely?"*
- *"Plan me 5 days in Zanzibar on a mid-range budget"*
- *"What scams should I watch out for in Diani?"*

---

## Tech stack

| Layer | Tool |
|-------|------|
| Backend | FastAPI + Python |
| AI | Groq (LLaMA 3.3 70B) |
| Search | Keyword search over embedded CSV data |
| Frontend | HTML + CSS + Vanilla JS |
| Deployment | Render |

---

## Data

95+ hotels, 25 attractions, 29 transport options, 20 scam warnings and 18 malls across Kenya, Tanzania, Uganda and Zanzibar — all stored as structured CSV files in `data/`.

---

## Run locally

```bash
git clone https://github.com/HERNIQNESS/travel-Africa-rag.git
cd travel-Africa-rag
python -m venv venv
source venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.example .env           # add your GROQ_API_KEY
uvicorn app.main:app --reload
```

Open `http://localhost:8000`

---

## Project structure

travel-africa-rag/
├── app/
│ ├── main.py # FastAPI backend
│ ├── templates/ # HTML frontend
│ └── static/ # CSS
├── data/
│ ├── raw/ # Source CSVs
│ └── cleaned/ # Processed CSVs
├── notebooks/ # Data cleaning & RAG pipeline
└── requirements.txt
