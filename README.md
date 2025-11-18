# Ukweli_Lens
Truth Verifier (Fact-Checking Engine)

## Overview
Ukweli_Lens is a multilingual AI fact-checking engine tailored for Kenya’s linguistic and cultural realities. It detects and verifies misinformation in English, Kiswahili, and Sheng, with a focus on:

- Technology-Facilitated Gender-Based Violence (TFGBV)
- High-risk political and ethnic incitement

## 🎯 The Problem

### Digital Harm
The rise of TFGBV and politically motivated hate speech is leading to real-world harm, trauma, and growing distrust.

### The Language Gap
Most global fact-checkers struggle to interpret dialects like Sheng or culturally nuanced Swahili expressions.

### The Context Gap
General-purpose models cannot distinguish simple falsehoods from dangerous statements that need urgent scrutiny.

## ✨ Core Features

1. **Grounded RAG Verification**  
   Verdicts grounded in a trusted “Source of Truth.”  
   RAG pipeline indexes verified Kenyan publications (KNBS, NGEC) and academic/legal GBV reports.  
   System avoids hallucination by design.

2. **Socio-Cultural Nuance Engine**  
   A spaCy PhraseMatcher scans for high-risk multi-word terms.  
   Examples:  
   - Sheng/Swahili: *mtu wetu*, *watu wa mlima*, *kamatakamata*  
   - PEV 2007-related incitement: *madoadoa*, *kabila*, *ukabila*  
   - TFGBV & misogyny: “asking for it”, *malaya*, “slay queen”

3. **Ethical Risk Penalty ($R_T$)**  
   A weighted penalty adjusts confidence for harmful or inflammatory claims.  
   - High-Risk (0.75): GBV terms & PEV-related incitement  
   - Low-Risk (0.25): General political misinformation

4. **Explainable AI (XAI) Verdict**  
   API returns a structured JSON packet containing:  
   - final_verdict  
   - explainable_confidence_score  
   - top_evidence_snippet  
   - xai_rationale (penalty math included)

## 🧠 How the XAI Works
\[
\text{Final Score} = \text{Total\_Evidence\_Confidence} \times (1.0 - \text{Ethical\_Risk\_Penalty})
\]

- **Total_Evidence_Confidence:** NLI stance score (−1.0 → +1.0)  
- **Ethical_Risk_Penalty:** Normalized risk value (0.0 → 1.0)  

**Example:**  
Claim: “She was asking for it”  
Confidence: 0.0 (inconclusive)  
High risk penalty applied → system detects toxicity even when unverifiable

## 🛠 Tech Stack & Models

| Component       | Details                                            |
|-----------------|----------------------------------------------------|
| Backend         | Django, Django REST Framework, django-cors-headers |
| Vector DB       | ChromaDB                                           |
| Data Ingestion  | LangChain (PyPDFLoader, RecursiveCharacterTextSplitter) |
| Models          | Retrieval: multi-qa-mpnet-base-dot-v1             |
|                 | Stance (NLI): roberta-large-mnli                   |
|                 | Nuance: spaCy (en_core_web_sm) + PhraseMatcher    |

## 🚀 How to Run (Backend API)

### 1. Setup
