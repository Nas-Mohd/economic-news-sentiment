
import re
import torch
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from snorkel.labeling import labeling_function
from snorkel.labeling.lf.nlp import nlp_labeling_function
from labeling.keywords import KEYWORDS
from labeling.anchors import ANCHORS, SEMANTIC_THRESHOLD
from labeling.labeling_functions.lf_factories import make_preprocessor
from typing import Dict, List

PRESENT = 1
ABSENT = 0
ABSTAIN = -1


lemmatizer = WordNetLemmatizer()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
economic_anchor_embeddings = []
for anchor in ANCHORS['economic_news']:
    economic_anchor_embeddings.append(embedder.encode(anchor, convert_to_tensor=True))

ECONOMIC_ASPECTS = ["monetary_policy", "inflation", "economic_growth", "labor_market", "consumer_activity", "business_activity", "financial_markets", "trade_external", "fiscal_policy", "energy_commodities", "banking_credit", "corporate_climate"]

# --- Convert to dict of list of tensors ---
ANCHORS_EMBEDDING: Dict[str, List] = {} # Value type will be list[tensor]
for category in ECONOMIC_ASPECTS:
    ANCHORS_EMBEDDING[category] = [
    embedder.encode(phrase, convert_to_tensor=True) for phrase in ANCHORS[category]
    ]


cache_embeddings = make_preprocessor(embedder)

# Pre-compile the regex pattern outside the function loop for maximum speed
COMPILED_CONTEXT_REGEX_REAL = re.compile("|".join(KEYWORDS["advanced_context_real"]), re.IGNORECASE)
COMPILED_CONTEXT_REGEX_TRADE = re.compile("|".join(KEYWORDS["advanced_context_trade"]), re.IGNORECASE)

# Helper function for fast regex matching
def contains_keywords(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)

@labeling_function()
def lf_lemmatized_cost_push(x):
    """
    Tokenizes and lemmatizes the sentence before applying regex matching
    to catch all morphological variants (e.g., pricing -> price, adjustments -> adjustment).
    """
    # 1. Tokenize and Lemmatize the text
    tokens = word_tokenize(x.text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Reconstruct the text into a clean string of base words
    processed_text = " ".join(lemmatized_tokens)

    # 2. Direct Fallback Keywords (now in base form)
    if re.search(r"(inflation|inflationary|cost of living|purchasing power)", processed_text):
        return PRESENT

    # 3. Base-word Component Patterns
    # Notice we use the root form here (e.g., 'utility' instead of 'utilities')
    energy_inputs = r"(diesel|fuel|petrol|oil|energy|utility|electricity|pump price)"
    price_adjustments = r"(adjustment|hike|spike|rise|increase|soaring|escalate|tariff|price|charge)"
    systemic_impact = r"(consumer|business|enterprise|industry|household|supply chain|economic effect|burden|margin)"

    # NEW: Macro economic shock indicators for classic cost-push inflation drivers
    macro_shocks = r"(supply shock|commodity shock|import cost|raw material shortage|wage spiral)"

    # 4. Multi-component Matching
    has_energy = re.search(energy_inputs, processed_text)
    has_adjustment = re.search(price_adjustments, processed_text)
    has_impact = re.search(systemic_impact, processed_text)

    # NEW: Check if the text explicitly cites a macro supply shock alongside pricing terms
    has_macro_shock = re.search(macro_shocks, processed_text)

    # Trigger if it hits the original multi-component energy/utility impact matrix
    if has_energy and has_adjustment and has_impact:
        return PRESENT

    # FIX TRIGGER: Trigger if a explicit supply/macro shock is directly linked
    # to a price adjustment/term (e.g., "supply shock fits into domestic price")
    if has_macro_shock and has_adjustment:
        return PRESENT

    return ABSTAIN


# Check if sentence is actually related to economics
@labeling_function(pre=[cache_embeddings])
def lf_is_not_economic_news(x):
    """
    Acts as a master filter. Votes ABSENT (0) if the text's embedding fails 
    to clear the SEMANTIC_THRESHOLD against every single anchor in our collection.
    """
    # 1. Start with your baseline economic news anchors
    all_tensors = list(economic_anchor_embeddings)
    
    # 2. Dynamically gather and flatten all tensors from your ECONOMIC_ASPECTS registry
    for category in ECONOMIC_ASPECTS:
        if category in ANCHORS_EMBEDDING:
            all_tensors.extend(ANCHORS_EMBEDDING[category])
            
    # 3. Stack individual 1D tensors into a single, high-performance 2D matrix
    # (Assuming embeddings are PyTorch tensors based on convert_to_tensor=True)
    master_anchor_matrix = torch.stack(all_tensors)
    
    # 4. Compute cosine similarity across the entire matrix in one parallel pass
    similarities = util.cos_sim(x.embedding, master_anchor_matrix)[0]
    
    # 5. Gatekeeper Logic: If it doesn't match a single anchor, rule it out.
    # We use torch.any() for optimized GPU/CPU tensor evaluation
    if not torch.any(similarities >= SEMANTIC_THRESHOLD-0.05):
        return ABSENT
        
    return ABSTAIN


# Complex Conditional Rule Tuning
@labeling_function()
def lf_monetary_policy_inflation_exclusion(x):
    """
    Disambiguates institutional monetary policy from household cost-of-living metrics.
    """
    has_inflation = contains_keywords(x.text, KEYWORDS['inflation'])
    has_monetary = contains_keywords(x.text, KEYWORDS['monetary_policy'])
    has_consumer = contains_keywords(x.text, KEYWORDS['consumer_activity'])

    if has_inflation and has_consumer:
        return ABSTAIN
    return PRESENT if (has_monetary and has_inflation) else ABSTAIN

# Checks for central banks
@nlp_labeling_function()
def lf_central_banks_nlp(x):
    """
    Isolates Central Bank institution mentions.
    Target Aspects: Monetary_Financial, Fiscal_Government
    """
    for ent in x.doc.ents:
        if ent.label_ == "ORG" and ent.text.lower() in KEYWORDS['central_banks']:
            return PRESENT
    return ABSTAIN

# Checks if there is a mention of a rate change
@nlp_labeling_function()
def lf_rate_actions_nlp(x):
    """
    Isolates precise grammar structures where interest rates are being altered/held.
    Target Aspects: Monetary_Financial, Inflation_Prices, Fiscal_Government
    """
    for token in x.doc:
        if token.text.lower() == "rate" and token.dep_ in ("dobj", "nsubj", "pobj"):
            if token.head.lemma_ in KEYWORDS['rate_actions']:
                return PRESENT
    return ABSTAIN

@nlp_labeling_function()
def lf_labor_market_action_nlp(x):
    """
    Isolates corporate or market-driven labor movements (e.g., "Companies lay off..."),
    state-driven worker protection adjustments, and labor statistics shifts (e.g., population headcount changes).
    Target Aspects: Labor_Consumption
    """
    # 1. Loop through ALL tokens
    for token in x.doc:

        # 2. Focus on subjects, passive subjects, and prepositional objects
        if token.dep_ in ("nsubj", "nsubjpass", "pobj"):

            # Core Check: If the token is a labor market element
            if token.lemma_ in KEYWORDS['labor_market']:

                # Check the direct governing head first
                if token.head.lemma_ in KEYWORDS['labor_actions']:
                    return PRESENT

                # FALLBACK FIX: Climb the dependency tree (Ancestors)
                # This handles statistical layouts (e.g., "number [of] workers [recorded] a [rise]")
                for i, ancestor in enumerate(token.ancestors):
                    if i >= 3: # Keep it efficient and bounded to 3 levels
                        break
                    if ancestor.lemma_ in KEYWORDS['labor_actions'] or ancestor.lemma_ in ("rise", "increase", "drop", "growth", "decline", "fall"):
                        return PRESENT

            # Alternative Check: If the governor is a labor term, look at the token itself
            elif token.head.lemma_ in KEYWORDS['labor_market']:
                if token.lemma_ in KEYWORDS['labor_actions'] or token.lemma_ in ("rise", "increase", "drop", "growth", "decline", "fall"):
                    return PRESENT

    return ABSTAIN


@labeling_function()
def lf_central_bank_fiscal_exclusion(x):
    """
    Abstains or rejects if a central bank is speaking without clear fiscal context,
    preventing institutional bleed.
    """
    has_central_bank = contains_keywords(x.text, KEYWORDS['central_banks'])
    has_fiscal_keywords = contains_keywords(x.text, KEYWORDS['fiscal_policy'])

    # If a central bank is talking, but there are no government spending/tax keywords,
    # actively force an ABSTAIN (or return 0 if you want it to vote ABSENT)
    if has_central_bank and not has_fiscal_keywords:
        return ABSENT

    return ABSTAIN

@labeling_function()
def lf_fiscal_sector_clash_exclusion(x):
    """
    If a sentence is overwhelmingly tracking heavy monetary or external macro metrics
    WITHOUT fiscal keywords, explicitly vote 0 (Absent) instead of abstaining.
    """
    KEYWORDS['fiscal_policy_clash'] = [
    # === 1. Taxation & Government Revenue (Direct & Indirect) ===
    "tax", "taxes", "taxation", "corporate tax", "income tax", "windfall tax",
    "sst", "gst", "sales tax", "service tax", "value-added tax", "vat",
    "tariff", "tariffs", "customs duty", "excise duty", "tax holiday", 
    "tax incentive", "tax incentives", "tax exemption", "tax revenue",
    "bracket", "tax bracket", "capital gains tax", "cgt",

    # === 2. National Budgeting & Expenditures ===
    "budget", "national budget", "belanjawan", "supply bill", "fiscal budget",
    "operating expenditure", "opex", "development expenditure", "depex",
    "government allocation", "government spending", "public spending",
    "state coffers", "treasury allocation", "public expenditure",

    # === 3. Subsidies & Social Assistance (Crucial for Malaysia/SEA context) ===
    "subsidy", "subsidies", "targeted subsidy", "subsidy rationalisation",
    "subsidy restructuring", "fuel subsidy", "ron95 subsidy", "diesel subsidy",
    "cash aid", "cash assistance", "fiscal assistance", "financial assistance",
    "br1m", "bsh", "bpr", "str", "sumbangan tunai rahmah", "social safety net",
    "welfare payout", "government handout",

    # === 4. Sovereign Debt & Fiscal Deficits ===
    "fiscal deficit", "budget deficit", "fiscal surplus", "budget surplus",
    "sovereign debt", "government debt", "national debt", "public debt",
    "debt-to-gdp ratio", "fiscal consolidation", "statutory debt ceiling",
    "debt ceiling", "treasury bonds", "mgs", "malaysian government securities",

    # === 5. Institutional Fiscal Frameworks & Legistlation ===
    "ministry of finance", "mof", "treasury", "fiscal policy committee", "fpc",
    "inland revenue board", "lhdn", "customs department", "royal malaysian customs",
    "fiscal responsibility act", "fra", "public finance", "fiscal space"
    ]


    has_monetary = contains_keywords(x.text, KEYWORDS['monetary_policy'])
    has_external = contains_keywords(x.text, KEYWORDS['trade_external']) or contains_keywords(x.text, KEYWORDS['energy_commodities'])
    has_fiscal = contains_keywords(x.text, KEYWORDS['fiscal_policy_clash'])

    # If it's heavily monetary/external but lacks fiscal words, vote ABSENT (0)
    if (has_monetary or has_external) and not has_fiscal:
        return ABSENT

    return ABSTAIN

@labeling_function()
def lf_monetary_exclusion_clash(x):
    """
    Votes ABSENT (0) if the sentence is heavily focused on pure fiscal budgets
    or corporate labor actions without any monetary indicator.
    """
    has_monetary = contains_keywords(x.text, KEYWORDS['monetary_policy']) or contains_keywords(x.text, KEYWORDS['financial_markets'])
    has_fiscal_heavy = contains_keywords(x.text, ["budget deficit", "austerity", "tax base", "tax revenue"])
    has_labor_heavy = contains_keywords(x.text, ["layoffs", "unemployment", "hiring", "minimum wage"])

    if (has_fiscal_heavy or has_labor_heavy) and not has_monetary:
        return ABSENT
    return ABSTAIN

@labeling_function()
def lf_inflation_exclusion_clash(x):
    """
    Votes ABSENT (0) if text is strictly about equity market adjustments, corporate
    automation, or trade treaties without direct price pressure vocabulary.
    """
    has_inflation = contains_keywords(x.text, KEYWORDS['inflation'])
    has_market_heavy = contains_keywords(x.text, ["bursa", "equity", "stock index", "yield curve"])
    has_corp_heavy = contains_keywords(x.text, ["automation", "property rights", "red tape", "free trade"])

    if (has_market_heavy or has_corp_heavy) and not has_inflation:
        return ABSENT
    return ABSTAIN

@labeling_function()
def lf_activity_exclusion_clash(x):
    """
    Votes ABSENT (0) if the text is confined to central bank rate holding syntax
    or consumer debt metrics with no production or growth keywords.
    """
    has_activity = contains_keywords(x.text, KEYWORDS['economic_growth']) or contains_keywords(x.text, KEYWORDS['business_activity'])
    has_rate_heavy = contains_keywords(x.text, ["rate cut", "rate hike", "opr", "overnight policy rate"])
    has_consumer_debt = contains_keywords(x.text, ["credit card usage", "household debt"])

    if (has_rate_heavy or has_consumer_debt) and not has_activity:
        return ABSENT
    return ABSTAIN

@labeling_function()
def lf_labor_exclusion_clash(x):
    """
    Votes ABSENT (0) if the text details cross-border energy/commodity markets
    or sovereign public debt dynamics with no mention of workers or consumers.
    """
    has_labor_cons = contains_keywords(x.text, KEYWORDS['labor_market']) or contains_keywords(x.text, KEYWORDS['consumer_activity'])
    has_energy_heavy = contains_keywords(x.text, KEYWORDS['energy_commodities'])
    has_sovereign_debt = contains_keywords(x.text, ["public debt", "budget deficit", "sovereign bond"])

    if (has_energy_heavy or has_sovereign_debt) and not has_labor_cons:
        return ABSENT
    return ABSTAIN

@labeling_function()
def lf_fiscal_exclusion_clash(x):
    """
    Votes ABSENT (0) if a sentence is overwhelmingly focused on monetary policy
    mechanisms or cross-border trade balances without any legislative government anchors.
    """
    has_fiscal = contains_keywords(x.text, KEYWORDS['fiscal_policy'])
    has_monetary_heavy = contains_keywords(x.text, KEYWORDS['monetary_policy']) or contains_keywords(x.text, KEYWORDS['central_banks'])
    has_trade_heavy = contains_keywords(x.text, KEYWORDS['trade_external'])

    if (has_monetary_heavy or has_trade_heavy) and not has_fiscal:
        return ABSENT
    return ABSTAIN

@labeling_function()
def lf_external_exclusion_clash(x):
    """
    Votes ABSENT (0) if the text handles hyper-localized economic variables
    like domestic minimum wages, national retail sales, or localized infrastructure.
    """
    has_external = contains_keywords(x.text, KEYWORDS['trade_external']) or contains_keywords(x.text, KEYWORDS['energy_commodities'])
    has_local_heavy = contains_keywords(x.text, ["minimum wage", "retail sales", "payroll tax", "infrastructure"])

    if has_local_heavy and not has_external:
        return ABSENT
    return ABSTAIN

@labeling_function()
def lf_deny_fiscal_if_pure_labor(x):
    labor_indicators = {"employers", "wages", "dismissals", "workers", "leave", "strike"}
    fiscal_indicators = {"government", "tax", "budget", "subsidy", "ministry", "fiscal"}

    text_lower = x.text.lower()

    # If it is explicitly about workplace rights and mentions NO government apparatus
    if any(word in text_lower for word in labor_indicators) and not any(word in text_lower for word in fiscal_indicators):
        return ABSENT  # Explicitly vote ABSENT to break the passive voting pattern
    return ABSTAIN

@labeling_function()
def lf_advanced_context_regex_trade(x):
    """
    Scans for complex structural phrases representing external sector impacts
    and real economic activity shocks in a single high-speed pass.
    """
    return PRESENT if COMPILED_CONTEXT_REGEX_TRADE.search(x.text) else ABSTAIN

@labeling_function()
def lf_advanced_context_regex_real(x):
    """
    Scans for complex structural phrases representing external sector impacts
    and real economic activity shocks in a single high-speed pass.
    """
    return PRESENT if COMPILED_CONTEXT_REGEX_REAL.search(x.text) else ABSTAIN


@labeling_function(pre=[cache_embeddings])
def lf_fiscal_policy_semantic_true(x):
    # Fires PRESENT (1) only if similarity is very high
    if any(util.cos_sim(x.embedding, anchor) > SEMANTIC_THRESHOLD for anchor in ANCHORS['fiscal_policy']):
        return PRESENT
    return ABSTAIN

@labeling_function(pre=[cache_embeddings])
def lf_fiscal_policy_semantic_false(x):
    # Explicitly flags ABSENT (0) if it matches non-fiscal anchors
    if all(util.cos_sim(x.embedding, anchor) > SEMANTIC_THRESHOLD - 0.1 for anchor in ANCHORS['fiscal_policy']):
        return ABSENT
    return ABSTAIN

@nlp_labeling_function()
def lf_fiscal_actions_nlp(x):
    """
    Isolates precise grammar structures where fiscal instruments (taxes, subsidies, budgets, packages)
    are being acted upon by a governing verb, or where government policy is explicitly directed.
    Target Aspects: Fiscal_Government
    """
    for token in x.doc:
        # Check if the current noun is a known fiscal instrument
        if token.lemma_ in KEYWORDS['fiscal_instruments']:

            # FIX: Added 'ccomp' to handle reported-speech news layout patterns
            if token.dep_ in ("dobj", "nsubj", "nsubjpass", "pobj", "ccomp"):

                # Check the direct governing verb
                if token.head.lemma_ in KEYWORDS['fiscal_actions']:
                    return PRESENT

                # FALLBACK FIX 1: Look up the tree (Ancestors)
                # Handles nested prepositional phrases and split hyphens (e.g., "re-evaluation -> of -> subsidies")
                # Restricts search to a maximum depth of 3 to avoid false positive triggers elsewhere in long sentences
                for i, ancestor in enumerate(token.ancestors):
                    if i >= 3:
                        break
                    if ancestor.lemma_ in KEYWORDS['fiscal_actions']:
                        return PRESENT

                # FALLBACK FIX 2: Check if the token's children/advcl contain the action verb
                # (Handles when a currency number splits the token dependency tree)
                for child in token.children:
                    if child.lemma_ in KEYWORDS['fiscal_actions']:
                        return PRESENT

        # Rule 2: Expanded to include "minister" and "package"
        if token.lemma_ in ("government", "ministry", "policy", "measure", "minister", "package"):
            if token.dep_ in ("dobj", "nsubj", "nsubjpass", "pobj", "ccomp") and token.head.lemma_ in KEYWORDS['fiscal_actions']:
                return PRESENT

            # Check ancestors for Rule 2 as well to capture multi-word structural policy mentions
            for i, ancestor in enumerate(token.ancestors):
                if i >= 3:
                    break
                if token.dep_ in ("dobj", "nsubj", "nsubjpass", "pobj", "ccomp") and ancestor.lemma_ in KEYWORDS['fiscal_actions']:
                    return PRESENT

    return ABSTAIN