from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import chromadb
from sentence_transformers import SentenceTransformer 
import spacy
from spacy.matcher import PhraseMatcher 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Import your THREE new keyword lists ---
from .nuance_config import GBV_MYTH_RISK_WORDS, HIGH_RISK_INCITEMENT_WORDS, LOW_RISK_DISINFO_WORDS

# --------------------------------------------------------------------------
# I. GLOBAL INITIALIZATION (Loads models once when Django starts)
# --------------------------------------------------------------------------
print("Ukweli Lens: Initializing RAG & AI Components...")
try:
    # --- Phase 1: RAG Retrieval Components ---
    CHROMA_PATH = "./chroma_db"
    RETRIEVAL_MODEL_NAME = "multi-qa-mpnet-base-dot-v1" 
    COLLECTION_NAME = "uhakiki_docs"
    
    retrieval_model = SentenceTransformer(RETRIEVAL_MODEL_NAME)
    db_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db_client.get_collection(name=COLLECTION_NAME)

    # --- Phase 2, Step 5: Nuance Engine (UPGRADED) ---
    nlp = spacy.load("en_core_web_sm")
    
    # --- Build the PhraseMatcher ---
    nuance_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

    # Convert our keyword lists into spaCy "Doc" objects
    gbv_patterns = [nlp.make_doc(text) for text in GBV_MYTH_RISK_WORDS]
    high_risk_pol_patterns = [nlp.make_doc(text) for text in HIGH_RISK_INCITEMENT_WORDS]
    low_risk_pol_patterns = [nlp.make_doc(text) for text in LOW_RISK_DISINFO_WORDS]

    # Add the patterns to the matcher with new Rule IDs
    nuance_matcher.add("GBV_RISK", gbv_patterns)
    nuance_matcher.add("HIGH_POLITICAL_RISK", high_risk_pol_patterns)
    nuance_matcher.add("LOW_POLITICAL_RISK", low_risk_pol_patterns)
    
    print("spaCy and Nuance PhraseMatcher loaded successfully.")
    
    # --- Phase 2, Step 6: Stance Detection Model ---
    STANCE_MODEL_NAME = "roberta-large-mnli" 
    
    stance_tokenizer = AutoTokenizer.from_pretrained(STANCE_MODEL_NAME)
    stance_model = AutoModelForSequenceClassification.from_pretrained(STANCE_MODEL_NAME)
    
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
    nuance_matcher = None
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
        
        if not collection or not stance_model or not nlp or not nuance_matcher:
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

        # --- 2. PHASE 2, STEP 5: NUANCE ENGINE (UPGRADED) ---
        doc = nlp(claim_text) 
        matches = nuance_matcher(doc)
        
        # Initialize all three risk scores
        risk_score_gbv = 0
        risk_score_politics_high = 0
        risk_score_politics_low = 0
        
        # Iterate over the matches found
        for match_id, start, end in matches:
            rule_id = nlp.vocab.strings[match_id] # Get the match ID string
            if rule_id == "GBV_RISK":
                risk_score_gbv += 1
            elif rule_id == "HIGH_POLITICAL_RISK":
                risk_score_politics_high += 1
            elif rule_id == "LOW_POLITICAL_RISK":
                risk_score_politics_low += 1
        
        # --- NEW R_T Calculation ---
        # Apply the 0.75 weight to both high-risk categories
        high_risk_score = (risk_score_gbv * 0.75) + (risk_score_politics_high * 0.75)
        # Apply the 0.25 weight to the low-risk category
        low_risk_score = (risk_score_politics_low * 0.25)
        
        # Normalize the penalty (e.g., divide by 3.0 as a simple normalizer)
        ethical_risk_penalty = min(1.0, (high_risk_score + low_risk_score) / 3.0)
        
        identified_entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        # --- 3. PHASE 2, STEP 6 & 7: STANCE & XAI SCORE ---       
        stance_verdicts = []
        final_verdict = "UNVERIFIABLE"
        total_confidence = 0.0
        top_evidence = None

        if evidence_chunks:
            for i, evidence in enumerate(evidence_chunks): 
                inputs = stance_tokenizer(evidence, claim_text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    logits = stance_model(**inputs).logits
                
                predicted_class_id = logits.argmax().item()
                verdict = stance_id_to_label[predicted_class_id]
                
                stance_score = 0.0
                if verdict == "SUPPORT":
                    stance_score = 1.0
                elif verdict == "REFUTE":
                    stance_score = -1.0
                
                evidence_quality = 1.0 - evidence_distances[i]
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

            if total_confidence > 0.3:
                final_verdict = "VERIFIED (SUPPORTED)"
            elif total_confidence < -0.3:
                final_verdict = "FALSE (REFUTED)"
            else:
                final_verdict = "INCONCLUSIVE (NEUTRAL)"

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
                    # Show the new, more detailed trigger list
                    "risk_triggers": f"GBV (High): {risk_score_gbv}, Incitement (High): {risk_score_politics_high}, Disinfo (Low): {risk_score_politics_low}",
                    "identified_entities": identified_entities
                },
                
                "all_stance_verdicts": stance_verdicts
            },
            status=status.HTTP_200_OK
        )