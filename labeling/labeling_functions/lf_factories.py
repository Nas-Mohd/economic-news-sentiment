# labeling/factories.py

import spacy
from spacy.matcher import PhraseMatcher
from sentence_transformers import util
from snorkel.labeling import labeling_function
from snorkel.labeling.lf.nlp import nlp_labeling_function
from snorkel.preprocess import preprocessor

PRESENT = 1
ABSENT = 0
ABSTAIN = -1

nlp = spacy.load("en_core_web_sm")


def make_preprocessor(embedder):
    """Returns a memoized preprocessor bound to the given embedder."""
    @preprocessor(memoize=True)
    def cache_embeddings(x):
        if not hasattr(x, "embedding"):
            x.embedding = embedder.encode(x.text, convert_to_tensor=True)
        return x
    return cache_embeddings

def make_keyword_lf(category_name, keywords):
    lower_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    lemma_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    
    keyword_list = keywords[category_name] if isinstance(keywords, dict) else keywords
    
    # Compile patterns
    lower_patterns = [nlp(kw.lower()) for kw in keyword_list]
    lemma_patterns = [nlp(kw) for kw in keyword_list]
    
    lower_matcher.add(f"{category_name}_lower", lower_patterns)
    lemma_matcher.add(f"{category_name}_lemma", lemma_patterns)

    @nlp_labeling_function()
    def lf_keyword(x):
        # 1. Use the pre-computed Snorkel doc for the heavy linguistic matching
        native_doc = x.doc 
        
        # 2. OPTIMIZATION: Use make_doc for the lowercase fallback.
        # This only tokenizes and skips the heavy tagger/parser/lemmatizer components.
        lower_doc = nlp.make_doc(native_doc.text.lower())
        
        # 3. Check LOWER first
        if lower_matcher(lower_doc):
            return PRESENT
            
        # 4. Check LEMMA second
        if lemma_matcher(native_doc):
            return PRESENT
            
        return ABSTAIN

    lf_keyword.name = f"lf_{category_name}_nlp_matcher"
    return lf_keyword


def make_semantic_lf(category_name, anchors, embedder, threshold=0.5):
    cache_embeddings = make_preprocessor(embedder)

    @labeling_function(pre=[cache_embeddings])
    def lf_semantic(x):
        if any(util.cos_sim(x.embedding, anchor) > threshold for anchor in anchors[category_name]):
            return PRESENT
        elif all(util.cos_sim(x.embedding, anchor) < 0.05 for anchor in anchors[category_name]):
            return ABSENT
        return ABSTAIN

    lf_semantic.name = f"lf_{category_name}_semantic"
    return lf_semantic

def make_strict_semantic_lf(category_name, anchors, embedder, threshold=0.5):
    cache_embeddings = make_preprocessor(embedder)

    @labeling_function(pre=[cache_embeddings])
    def lf_semantic(x):
        if any(util.cos_sim(x.embedding, anchor) > (threshold*1.25) for anchor in anchors[category_name]):
            return PRESENT
        elif all(util.cos_sim(x.embedding, anchor) < 0.05 for anchor in anchors[category_name]):
            return ABSENT
        return ABSTAIN

    lf_semantic.name = f"lf_{category_name}_semantic"
    return lf_semantic