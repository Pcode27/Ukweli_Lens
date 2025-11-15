from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import chromadb
from sentence_transformers import SentenceTransformer 
import spacy
# --- NEW: Import Transformers for NLI (Stance Detection) ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Import your keyword lists ---
from .nuance_config import POLITICAL_ETHNIC_RISK_WORDS, GBV_MYTH_RISK_WORDS

# --------------------------------------------------------------------------
# I. GLOBAL INITIALIZATION (Loads models once when Django starts)
# --------------------------------------------------------------------------
print("Uhakiki System: Initializing RAG & AI Components...")
try:
    # --- Phase 1: RAG Retrieval Components ---
    CHROMA_PATH = "./chroma_db"
    RETRIEVAL_MODEL_NAME = "multi-qa-mpnet-base-dot-v1" 
    COLLECTION_NAME = "uhakiki_docs"
    
    retrieval_model = SentenceTransformer(RETRIEVAL_MODEL_NAME)
    db_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db_client.get_collection(name=COLLECTION_NAME)

    # --- Phase 2, Step 5: Nuance Engine ---
    nlp = spacy.load("en_core_web_sm")

    # --- Phase 2, Step 6: Stance Detection Model 
    
    # OLD (DeBERTa model with tokenizer issues): "microsoft/deberta-v2-xlarge-mnli"
    # NEW STABLE MODEL: A RoBERTa model fine-tuned on the same NLI (MNLI) task
    STANCE_MODEL_NAME = "roberta-large-mnli" 
    
    # We don't need use_fast=False for this one, it's more stable
    stance_tokenizer = AutoTokenizer.from_pretrained(STANCE_MODEL_NAME)
    stance_model = AutoModelForSequenceClassification.from_pretrained(STANCE_MODEL_NAME)
    
    # NLI Labels for this model:
    # 0 -> contradiction (REFUTE)
    # 1 -> neutral (NEUTRAL)
    # 2 -> entailment (SUPPORT)
    stance_id_to_label = {
        stance_model.config.label2id['CONTRADICTION']: "REFUTE",
        stance_model.config.label2id['NEUTRAL']: "NEUTRAL",
        stance_model.config.label2id['ENTAILMENT']: "SUPPORT"
    }
    
    print("Uhakiki System: All components loaded successfully.")

except Exception as e:
    print(f"!!! CRITICAL ERROR LOADING MODELS: {e}")
    collection = None
    retrieval_model = None
    nlp = None
    stance_model = None
# --------------------------------------------------------------------------

class VerifyAPIView(APIView):
    """
    API endpoint to verify a claim. (Final Version)
    """
    def post(self, request, *args, **kwargs):
        claim_text = request.data.get('claim') # This is our "hypothesis"

        if not claim_text:
            return Response(
                {"error": "No 'claim' provided in POST data"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not collection or not stance_model or not nlp:
             return Response(
                {"error": "Verification system offline. Models not initialized."}, 
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        # --- 1. PHASE 1: RETRIEVAL ---
        claim_embedding = retrieval_model.encode([claim_text])
        results = collection.query(
            query_embeddings=claim_embedding.tolist(),
            n_results=3, 
            include=['documents', 'metadatas', 'distances'] 
        )
        evidence_chunks = results.get('documents', [[]])[0] # These are our "premises"
        evidence_metadatas = results.get('metadatas', [[]])[0]
        evidence_distances = results.get('distances', [[]])[0]

        # --- 2. PHASE 2, STEP 5: NUANCE ENGINE (FIXED) ---
        claim_lower = claim_text.lower()
        doc = nlp(claim_lower) # Process text with spaCy
        
        # Use spaCy tokens (token.text) to correctly handle punctuation
        risk_score_politics = sum(1 for token in doc if token.text in POLITICAL_ETHNIC_RISK_WORDS)
        risk_score_gbv = sum(1 for token in doc if token.text in GBV_MYTH_RISK_WORDS)
        
        # This is our R_T penalty (0.0 to 1.0)
        ethical_risk_penalty = min(1.0, (risk_score_politics * 0.25 + risk_score_gbv * 0.75) / 3.0)
        
        ner_doc = nlp(claim_text)
        identified_entities = [{"text": ent.text, "label": ent.label_} for ent in ner_doc.ents]

        # --- 3. PHASE 2, STEP 6 & 7: STANCE & XAI SCORE ---
        stance_verdicts = []
        final_verdict = "UNVERIFIABLE"
        total_confidence = 0.0
        top_evidence = None

        if evidence_chunks:
            for i, evidence in enumerate(evidence_chunks): # evidence is the "premise"
                # --- Stance Detection (NLI format) ---
                # We check if the evidence (premise) entails the claim (hypothesis)
                inputs = stance_tokenizer(evidence, claim_text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    logits = stance_model(**inputs).logits
                
                predicted_class_id = logits.argmax().item()
                verdict = stance_id_to_label[predicted_class_id]
                
                # --- XAI Score Calculation ---
                stance_score = 0.0
                if verdict == "SUPPORT":
                    stance_score = 1.0
                elif verdict == "REFUTE":
                    stance_score = -1.0
                
                evidence_quality = 1.0 - evidence_distances[i] # Similarity score
                
                chunk_confidence = stance_score * evidence_quality
                total_confidence += chunk_confidence

                stance_verdicts.append({
                    "verdict": verdict,
                    "evidence": evidence,
                    "source": evidence_metadatas[i].get('source', 'Unknown'),
                    "page": evidence_metadatas[i].get('page', 'N/A'),
                    "confidence": round(chunk_confidence, 3)
                })

            if stance_verdicts:
                top_evidence = max(stance_verdicts, key=lambda x: abs(x['confidence']))

            # Determine final verdict based on total confidence
            if total_confidence > 0.3: # Threshold for support
                final_verdict = "VERIFIED (SUPPORTED)"
            elif total_confidence < -0.3: # Threshold for refute
                final_verdict = "FALSE (REFUTED)"
            else:
                final_verdict = "INCONCLUSIVE (NEUTRAL)"

        # Apply the Ethical Risk Penalty ($R_T$)
        final_score = total_confidence * (1.0 - ethical_risk_penalty)

        # --- 4. RETURN FINAL XAI RESPONSE ---
        return Response(
            {
                "input_claim": claim_text,
                "final_verdict": final_verdict,
                "explainable_confidence_score": round(final_score, 4),
                
                "xai_rationale": {
                    "total_evidence_confidence": round(total_confidence, 4),
                    "ethical_risk_penalty_R_T": round(ethical_risk_penalty, 4),
                    "final_score_calculation": f"{total_confidence} * (1.0 - {ethical_risk_penalty}) = {final_score}"
                },

                "top_evidence_snippet": top_evidence,
                
                "nuance_analysis": {
                    "risk_triggers": f"Politics: {risk_score_politics}, GBV: {risk_score_gbv}",
                    "identified_entities": identified_entities
                },
                
                "all_stance_verdicts": stance_verdicts
            },
            status=status.HTTP_200_OK
        )