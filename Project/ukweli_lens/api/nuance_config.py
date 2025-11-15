# api/nuance_config.py

# High-risk keywords derived from PolitiKwel insights (political, ethnic, social conflict)
POLITICAL_ETHNIC_RISK_WORDS = [
    # --- Add at least 20-30 terms here ---
    "azimio", "kenya kwanza", "iebc", "deep state", "kplc", "hustler", 
    "kimani", "raila", "ruto", "kalenjin", "kikuyu", "luo", "nyanza",
    "madoadoa", "wizi", "cartel", "kudumbisha", "njaa"
]

# Critical words related to Gender-Based Violence (GBV) myths or threats
GBV_MYTH_RISK_WORDS = [
    # --- Add at least 15-20 terms here ---
    "she deserved", "blame her", "dress code", "asking for it", "doxing", 
    "expose", "must obey", "wifey", "sidelined", "prostitute", "sgbv",
    "femicide", "rape culture", "victim blaming"
]