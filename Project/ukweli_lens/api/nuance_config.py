# api/nuance_config.py
#
# This file contains the "Kenyan Nuance Engine" keywords.
# These lists are used to calculate the Ethical Risk Penalty ($R_T$).
# All keywords should be in LOWERCASE for easier matching.

# --------------------------------------------------------------------------
# 1. HIGH-RISK: GBV, MYTH & MISOGYNY WORDS
# --------------------------------------------------------------------------
# Weight: 0.75 (Immediate, direct personal harm)
# --------------------------------------------------------------------------
GBV_MYTH_RISK_WORDS = [
    # --- Direct GBV Terms ---
    "gbv", "sgbv", "femicide", "doxing", "revenge porn",
    # --- Victim-Blaming (English) ---
    "asking for it", "dress code", "blame her", "she deserved",
    "gold digger", "slay queen", "side chick",
    # --- Victim-Blaming (Sheng/Swahili) ---
    "malaya", "chang'aa", "dem", "kuchapwa",
    "adabu", "matako",
    # --- Abusive/Controlling Language ---
    "expose", "must obey", "wifey"
]

# --------------------------------------------------------------------------
# 2. HIGH-RISK: POLITICAL & ETHNIC INCITEMENT WORDS
# --------------------------------------------------------------------------
# Weight: 0.75 (Immediate, direct societal harm - PEV 2007 context)
# --------------------------------------------------------------------------
HIGH_RISK_INCITEMENT_WORDS = [
    # --- High-Risk Inflammatory Slang (Sheng/Swahili) ---
    "madoadoa",         # "stains" (a highly inflammatory ethnic slur)
    "kabila",           # "tribe"
    "ukabila",          # "tribalism"
    "watu wa mlima",    # "people of the mountain" (regional code)
    "watu wa bonde",    # "people of the valley" (regional code)
    "mganga",           # "witchdoctor" (used as a political slur)
    "mchawi",           # "wizard" (slur)
    "watu wasiojulikana", # "unknown people" (threat)
    "kamatakamata",     # Slang for politically-motivated arrests
    "mtu wetu",         # "our person" (nepotism/tribalism)

    # --- Politicians (Matching names is high-risk) ---
    "raila",
    "ruto",
    "odinga",
    "kenyatta",
    "uhuru",
    "gachagua",
    "karua"
]

# --------------------------------------------------------------------------
# 3. LOW-RISK: INSTITUTIONAL DISINFORMATION WORDS
# --------------------------------------------------------------------------
# Weight: 0.25 (Slower-moving, institutional harm)
# --------------------------------------------------------------------------
LOW_RISK_DISINFO_WORDS = [
    # --- Political Entities & Slang ---
    "azimio", 
    "kenya kwanza",
    "uda",
    "odm",
    "iebc",
    "deep state",
    "dynasty",
    "hustler",
    "system",           # "the system" (conspiracy)
    
    # --- Public Institutions (for Disinformation) ---
    "kplc",
    "safaricom",
    "nys",              # National Youth Service
    "kemsa",
    "kra",
    "dci",
    "wizi",             # "theft"
    "mwizi",            # "thief"
]