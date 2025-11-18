# Ukweli_Lens

Truth Verifier (Fact-Checking Engine).

## Overview

This repository contains the Ukweli_Lens project: a multilingual AI fact-checking engine built for Kenya’s unique linguistic and cultural landscape. It detects and verifies misinformation in English, Kiswahili, and Sheng, with a special focus on Technology-Facilitated Gender-Based Violence (TFGBV) and high-risk political/ethnic incitement.

🎯 The Problem
Digital Harm

The rapid spread of Technology-Facilitated Gender-Based Violence (TFGBV) and politically motivated hate speech is fueling real-world harm, trauma, and distrust.

The Language Gap

Most global AI fact-checkers fail to understand local dialects like Sheng or the high-stakes cultural nuance of Swahili and code-switched phrases.

The Context Gap

Standard models cannot differentiate between a simple false claim and a dangerous, inflammatory statement (e.g., related to PEV 2007) that requires immediate, cautious handling.

✨ Core Features 

1. Grounded RAG Verification

The engine’s verdicts are grounded in a trusted “Source of Truth.”
A RAG pipeline indexes and queries only verified Kenyan government publications (KNBS, NGEC) and academic/legal reports on GBV.
The system never hallucinates an answer.

2. Socio-Cultural Nuance Engine

This is the ethical core.

A spaCy PhraseMatcher scans claims for a curated list of high-risk keywords, including:

Multi-word Sheng/Swahili:
mtu wetu, watu wa mlima, kamatakamata

PEV 2007 Incitement:
madoadoa, kabila, ukabila

TFGBV & Misogyny:
asking for it, malaya, slay queen

3. The Ethical Risk Penalty ($R_T$)

A weighted penalty is applied to high-risk claims.
This allows the system to identify claims that are not just false, but dangerous, and adjust its confidence.

High-Risk (0.75 weight): GBV terms & PEV-related incitement

Low-Risk (0.25 weight): General political disinformation

4. Explainable AI (XAI) Verdict

The API returns a complete JSON rationale, including:

final_verdict (e.g., FALSE (REFUTED))

explainable_confidence_score

top_evidence_snippet from trusted documents

xai_rationale (the penalty math)

🧠 How the XAI Works
Final Score = Total_Evidence_Confidence * (1.0 - Ethical_Risk_Penalty)


Total_Evidence_Confidence: Raw NLI stance score (−1.0 to +1.0)

Ethical_Risk_Penalty ($R_T$): Normalized risk score (0.0 → 1.0)

This lets Ukweli-Lens be both accurate and responsible.

Example:
A claim like “She was asking for it” becomes:

INCONCLUSIVE (0.0) confidence

High risk penalty (e.g., 0.25)

Showing the system understands the claim is both unverifiable and toxic.

🛠 Tech Stack & Models
Backend

Django

Django REST Framework

django-cors-headers

Vector DB

ChromaDB

Data Ingestion

LangChain (PyPDFLoader, RecursiveCharacterTextSplitter)

Core Models

Retrieval: multi-qa-mpnet-base-dot-v1

Stance (NLI): roberta-large-mnli

Nuance: spaCy (en_core_web_sm) + PhraseMatcher

🚀 How to Run (Backend API)
1. Setup
# Clone the repository
git clone https://github.com/[your-username]/ukweli_lens.git
cd ukweli_lens

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install django djangorestframework django-cors-headers chromadb sentence-transformers langchain-community pypdf spacy transformers torch "numpy<2.0.0" "blis==1.0.1" "thinc==8.2.5" "spacy==3.7.5"

2. Download AI Models
python -m spacy download en_core_web_sm


(Transformers models download automatically on first run.)

3. Ingest Your Data

Create a folder: source_documents/

Add verified Kenyan PDFs (KNBS, NGEC, etc.)

Run ingestion:

python ingest_data.py


This builds the chroma_db/ vector store.

4. Configure Django

Add corsheaders and api to INSTALLED_APPS

Add corsheaders.middleware.CorsMiddleware to MIDDLEWARE

Add CORS_ALLOWED_ORIGINS

Run migrations:

python manage.py migrate

5. Run the Server
python manage.py runserver

🧪 Testing the API
Test 1: Factual Claim
Invoke-WebRequest -Uri http://127.0.0.1:8000/api/verify/ -Method POST -ContentType "application/json" -Body '{"claim": "What is the cost of GBV in Kenya?"}'


Expected: VERIFIED (SUPPORTED)

Test 2: High-Risk (GBV) Claim
Invoke-WebRequest -Uri http://127.0.0.1:8000/api/verify/ -Method POST -ContentType "application/json" -Body '{"claim": "She was asking for it."}'


Expected:
INCONCLUSIVE + high ethical_risk_penalty_R_T (e.g., 0.25)

Test 3: High-Risk (Incitement) Claim
Invoke-WebRequest -Uri http://127.0.0.1:8000/api/verify/ -Method POST -ContentType "application/json" -Body '{"claim": "watu madoadoa"}'


Expected:
FALSE (REFUTED) + high ethical_risk_penalty_R_T
