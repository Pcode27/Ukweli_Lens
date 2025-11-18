Ukweli_Lens

Truth Verifier (Fact-Checking Engine)

Overview

Ukweli_Lens is a multilingual AI fact-checking engine tailored for Kenya’s linguistic and cultural realities. It detects and verifies misinformation in English, Kiswahili, and Sheng, with a focus on:

Technology-Facilitated Gender-Based Violence (TFGBV)

High-risk political and ethnic incitement

🎯 The Problem
Digital Harm

The rise of TFGBV and politically motivated hate speech is leading to real-world harm, trauma, and growing distrust.

The Language Gap

Most global fact-checkers struggle to interpret dialects like Sheng or culturally nuanced Swahili expressions.

The Context Gap

General-purpose models cannot distinguish simple falsehoods from dangerous statements that need urgent scrutiny.

✨ Core Features
1. Grounded RAG Verification

Verdicts grounded in a trusted “Source of Truth.”

RAG pipeline indexes verified Kenyan publications (KNBS, NGEC) and academic/legal GBV reports.

System avoids hallucination by design.

2. Socio-Cultural Nuance Engine

A spaCy PhraseMatcher scans for high-risk multi-word terms.

Examples:

Sheng/Swahili:

mtu wetu, watu wa mlima, kamatakamata

PEV 2007-related incitement:

madoadoa, kabila, ukabila

TFGBV & misogyny:

“asking for it”, malaya, “slay queen”

3. Ethical Risk Penalty ($R_T$)

A weighted penalty adjusts confidence for harmful or inflammatory claims.

High-Risk (0.75): GBV terms & PEV-related incitement

Low-Risk (0.25): General political misinformation

4. Explainable AI (XAI) Verdict

API returns a structured JSON packet containing:

final_verdict

explainable_confidence_score

top_evidence_snippet

xai_rationale (penalty math included)

🧠 How the XAI Works
Final Score = Total_Evidence_Confidence * (1.0 - Ethical_Risk_Penalty)


Total_Evidence_Confidence: NLI stance score (−1.0 → +1.0)

Ethical_Risk_Penalty: Normalized risk value (0.0 → 1.0)

Example

Claim: “She was asking for it”

Confidence: 0.0 (inconclusive)

High risk penalty applied → system detects toxicity even when unverifiable

🛠 Tech Stack & Models
Backend

Django

Django REST Framework

django-cors-headers

Vector DB

ChromaDB

Data Ingestion

LangChain (PyPDFLoader, RecursiveCharacterTextSplitter)

Models

Retrieval: multi-qa-mpnet-base-dot-v1

Stance (NLI): roberta-large-mnli

Nuance: spaCy (en_core_web_sm) + PhraseMatcher

🚀 How to Run (Backend API)
1. Setup
# Clone the repository
git clone https://github.com/[your-username]/ukweli_lens.git
cd ukweli_lens

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install django djangorestframework django-cors-headers chromadb sentence-transformers langchain-community pypdf spacy transformers torch "numpy<2.0.0" "blis==1.0.1" "thinc==8.2.5" "spacy==3.7.5"

2. Download AI Models
python -m spacy download en_core_web_sm


(Transformers models download on first run.)

3. Ingest Your Data

Create folder: source_documents/

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
