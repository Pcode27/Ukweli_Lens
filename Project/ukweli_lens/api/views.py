from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import chromadb
from sentence_transformers import SentenceTransformer 
import spacy
from spacy.matcher import PhraseMatcher 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from googleapiclient.discovery import build 
from deep_translator import GoogleTranslator

# --- Import your keyword lists ---
from .nuance_config import GBV_MYTH_RISK_WORDS, HIGH_RISK_INCITEMENT_WORDS, LOW_RISK_DISINFO_WORDS

# ==============================================================================
# CONFIGURATION & RESOURCES
# ==============================================================================
DOMAIN_RESOURCES = {
    "GBV_CRITICAL": {
        "alert": "SAFETY ALERT: This content contains potential gender-based violence triggers.",
        "actions": [
            {"name": "National GBV Helpline (HAK)", "contact": "1195 (Toll Free 24/7)"},
            {"name": "FIDA Kenya (Legal Aid)", "contact": "+254 722 509760"},
            {"name": "Gender Violence Recovery Centre", "contact": "+254 709 667 000"}
        ]
    },
    "HATE_SPEECH": {
        "alert": "INCITEMENT ALERT: This content may violate the NCI Act on hate speech.",
        "actions": [
            {"name": "Report to NCIC (Toll Free SMS)", "contact": "1547"},
            {"name": "Report to NCIC (Call)", "contact": "0702 777 000"}
        ]
    },
    "GENERAL": {
        "alert": "Standard Verification.",
        "actions": [
            {"name": "Verify with PesaCheck", "contact": "pesacheck.org"},
            {"name": "Independent Search", "contact": "Use trusted news sources."}
        ]
    }
}

# ==============================================================================
# WEB SEARCH MODULE (The "Freshness" Layer)
# ==============================================================================
def search_web_authoritative(query, nlp_model=None):
    """
    Searches verified Kenyan sources via Google PSE.
    Includes a 'Keyword Fallback' if the full sentence fails.
    """
    print(f"Web Agent: Searching for '{query}'...")
    
    trusted_sites = "site:go.ke OR site:pesacheck.org OR site:africacheck.org OR site:knbs.or.ke OR site:kbc.co.ke OR site:nation.africa OR site:standardmedia.co.ke OR site:ghettoradio.co.ke OR site:radiomaisha.co.ke"
    
    try:        
        api_key = os.getenv("GOOGLE_API_KEY")
        cse_id = os.getenv("GOOGLE_CSE_ID")
        
        if not api_key or not cse_id:
            print("Missing API Keys in Environment")
            return []

        service = build("customsearch", "v1", developerKey=api_key)
        
        # --- ATTEMPT 1: Full Sentence Search ---
        full_query = f"{query} ({trusted_sites})"
        result = service.cse().list(q=full_query, cx=cse_id, num=3).execute()
        
        # --- ATTEMPT 2: Keyword Fallback (If Attempt 1 failed) ---
        if 'items' not in result:
            print("No results for full sentence. Attempting Keyword Fallback...")
            
            keywords = []
            
            # Strategy A: Try spaCy extraction first
            if nlp_model:
                doc = nlp_model(query)
                keywords = [
                    token.text for token in doc 
                    if token.pos_ in ["PROPN", "NUM"] and not token.is_stop 
                ]
            
            # Strategy B: Heuristic Fallback (If spaCy failed to reduce the query)
            if not keywords or len(" ".join(keywords)) > len(query) * 0.8:
                print("spaCy extraction ineffective (likely Swahili). Using heuristic fallback.")
                words = query.split()
                keywords = [w for w in words if len(w) > 4][:5]

            if keywords:
                keyword_query_text = " ".join(keywords)
                print(f"Retrying with keywords: '{keyword_query_text}'")
                
                fallback_query = f"{keyword_query_text} ({trusted_sites})"
                result = service.cse().list(q=fallback_query, cx=cse_id, num=3).execute()
            else:
                print("No valid keywords found. Giving up.")

        # --- ATTEMPT 3: Translation Fallback ---
        if 'items' not in result:
            print("Keywords failed. Attempting Translation to English...")
            try:
                translated_query = GoogleTranslator(source='auto', target='en').translate(query)
                if translated_query and translated_query.lower() != query.lower():
                    print(f"Translated Claim: '{translated_query}'")
                    trans_query_full = f"{translated_query} ({trusted_sites})"
                    result = service.cse().list(q=trans_query_full, cx=cse_id, num=3).execute()
                else:
                    print("Translation yielded same text.")
            except Exception as trans_error:
                print(f"Translation API failed: {trans_error}")

        # --- Process Results ---
        web_evidence = []
        if 'items' in result:
            for item in result['items']:
                clean_snippet = item['snippet'].replace("\n", " ")
                evidence_text = f"{item['title']}. {clean_snippet}"
                web_evidence.append(f"{evidence_text} (Source: {item['link']})")
        
        print(f"Web Agent: Found {len(web_evidence)} results.")
        return web_evidence
        
    except Exception as e:
        print(f"Web Search Failed: {e}")
        return []

# ==============================================================================
# GLOBAL INITIALIZATION
# ==============================================================================
print("Ukweli-Lens: Initializing Hybrid Pipeline...")
try:
    CHROMA_PATH = "./chroma_db"
    RETRIEVAL_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" 
    COLLECTION_NAME = "uhakiki_docs"
    
    retrieval_model = SentenceTransformer(RETRIEVAL_MODEL_NAME)
    db_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db_client.get_collection(name=COLLECTION_NAME)

    nlp = spacy.load("en_core_web_sm")
    nuance_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

    gbv_patterns = [nlp.make_doc(text) for text in GBV_MYTH_RISK_WORDS]
    high_risk_pol_patterns = [nlp.make_doc(text) for text in HIGH_RISK_INCITEMENT_WORDS]
    low_risk_pol_patterns = [nlp.make_doc(text) for text in LOW_RISK_DISINFO_WORDS]

    nuance_matcher.add("DOMAIN_GBV", gbv_patterns)
    nuance_matcher.add("DOMAIN_INCITEMENT", high_risk_pol_patterns)
    nuance_matcher.add("DOMAIN_DISINFO", low_risk_pol_patterns)
    
    STANCE_MODEL_NAME = "roberta-large-mnli" 
    stance_tokenizer = AutoTokenizer.from_pretrained(STANCE_MODEL_NAME)
    stance_model = AutoModelForSequenceClassification.from_pretrained(STANCE_MODEL_NAME)
    
    stance_id_to_label = {
        stance_model.config.label2id['CONTRADICTION']: "REFUTE",
        stance_model.config.label2id['NEUTRAL']: "NEUTRAL",
        stance_model.config.label2id['ENTAILMENT']: "SUPPORT"
    }
    
    print("System Ready.")

except Exception as e:
    print(f"!!! CRITICAL INIT ERROR: {e}")
    collection = None; retrieval_model = None; nlp = None; stance_model = None; nuance_matcher = None


class VerifyAPIView(APIView):
    def post(self, request, *args, **kwargs):
        claim_text = request.data.get('claim')

        if not claim_text:
            return Response({"error": "No 'claim' provided in POST data"}, status=400)
        
        if not collection or not stance_model:
             return Response({"error": "Verification system offline. Models not initialized."}, status=503)

        # --- STAGE 1: CLAIM ANALYSIS ---
        doc = nlp(claim_text) 
        matches = nuance_matcher(doc)
        
        risk_score_gbv = 0
        risk_score_incitement = 0
        risk_score_disinfo = 0
        detected_domains = []

        for match_id, start, end in matches:
            rule_id = nlp.vocab.strings[match_id]
            if rule_id == "DOMAIN_GBV":
                risk_score_gbv += 1
                if "GBV" not in detected_domains: detected_domains.append("GBV")
            elif rule_id == "DOMAIN_INCITEMENT":
                risk_score_incitement += 1
                if "Incitement" not in detected_domains: detected_domains.append("Incitement")
            elif rule_id == "DOMAIN_DISINFO":
                risk_score_disinfo += 1
        
        total_penalty = min(1.0, ((risk_score_gbv * 0.75) + (risk_score_incitement * 0.75) + (risk_score_disinfo * 0.25)) / 3.0)
        identified_entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        # --- STAGE 2: HYBRID RETRIEVAL ---
        claim_embedding = retrieval_model.encode([claim_text])
        results = collection.query(
            query_embeddings=claim_embedding.tolist(),
            n_results=3, 
            include=['documents', 'metadatas', 'distances'] 
        )
        evidence_chunks = results.get('documents', [[]])[0]
        evidence_metadatas = results.get('metadatas', [[]])[0]
        evidence_distances = results.get('distances', [[]])[0]
        
        best_distance = evidence_distances[0] if evidence_distances else 1.0
        is_web_source = False
        
        if best_distance > 0.3:
            print(f"Local evidence weak (Dist: {round(best_distance, 2)}). Triggering Web Search...")
            web_chunks = search_web_authoritative(claim_text, nlp_model=nlp)
            
            if web_chunks:
                evidence_chunks = web_chunks
                evidence_metadatas = [{"source": "Live Web Search", "page": "External"} for _ in web_chunks]
                evidence_distances = [0.1] * len(web_chunks) 
                is_web_source = True
            else:
                print("Web search yielded no results. Falling back to local evidence.")

        # --- STAGE 3: VERDICT ---
        stance_verdicts = []
        total_confidence = 0.0
        top_evidence = None
        
        if evidence_chunks:
            for i, evidence in enumerate(evidence_chunks):
                if is_web_source:
                    evidence_quality = 0.9
                else:
                    raw_quality = 1.0 - evidence_distances[i]
                    evidence_quality = max(0.0, raw_quality)

                if evidence_quality < 0.4:
                    stance_verdicts.append({"verdict": "NEUTRAL (IRRELEVANT)", "confidence": 0.0, "evidence": evidence})
                    continue

                inputs = stance_tokenizer(evidence, claim_text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    logits = stance_model(**inputs).logits
                
                pred_id = logits.argmax().item()
                verdict = stance_id_to_label[pred_id]
                
                stance_val = 1.0 if verdict == "SUPPORT" else -1.0 if verdict == "REFUTE" else 0.0
                chunk_conf = stance_val * evidence_quality
                total_confidence += chunk_conf
                
                source_info = evidence_metadatas[i].get('source', 'Unknown')
                if is_web_source: source_info = "Live Web Search (Google PSE)"

                stance_verdicts.append({
                    "verdict": verdict,
                    "evidence": evidence[:200] + "...", 
                    "source": source_info,
                    "confidence": round(chunk_conf, 3)
                })

            if stance_verdicts:
                top_evidence = max(stance_verdicts, key=lambda x: abs(x['confidence']))

        if total_confidence > 0.25: final_verdict = "VERIFIED (SUPPORTED)"
        elif total_confidence < -0.25: final_verdict = "FALSE (REFUTED)"
        else: final_verdict = "INCONCLUSIVE (NEUTRAL)"

        final_score = total_confidence * (1.0 - total_penalty)

        resources = DOMAIN_RESOURCES["GENERAL"]
        if "GBV" in detected_domains: resources = DOMAIN_RESOURCES["GBV_CRITICAL"]
        elif "Incitement" in detected_domains: resources = DOMAIN_RESOURCES["HATE_SPEECH"]

        return Response({
            "input_claim": claim_text,
            "final_verdict": final_verdict,
            "explainable_confidence_score": round(final_score, 4),
            "source_type": "LIVE_WEB" if is_web_source else "LOCAL_DB",
            
            "xai_rationale": {
                "total_evidence_confidence": round(total_confidence, 4),
                "ethical_risk_penalty_R_T": round(total_penalty, 4),
                "final_score_calculation": f"{round(total_confidence,4)} * (1.0 - {round(total_penalty,4)}) = {round(final_score,4)}"
            },
            "top_evidence_snippet": top_evidence,
            "nuance_analysis": {
                "risk_triggers": f"GBV: {risk_score_gbv}, Incitement: {risk_score_incitement}, Disinfo: {risk_score_disinfo}",
                "identified_entities": identified_entities
            },
            "stage_3_actions": resources,
            "all_stance_verdicts": stance_verdicts
        }, status=status.HTTP_200_OK)