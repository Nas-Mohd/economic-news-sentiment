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
    # 1. Use LEMMA attribute so SpaCy natively handles variations (run, running, runs)
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    
    # 2. Process keywords through the full pipeline. 
    # Pre-lowercasing the keyword string keeps acronyms safe from aggressive lemmatization.
    patterns = [nlp(kw.lower()) for kw in keywords[category_name]]
    matcher.add(category_name, patterns)

    @nlp_labeling_function()
    def lf_keyword(x):
        # 3. x.doc is already provided by Snorkel, but to apply our lowercase fix,
        # we pull the raw text, lowercase it, and re-process it through the full NLP pipeline.
        normalized_doc = nlp(x.doc.text.lower())
        
        matches = matcher(normalized_doc)
        if matches:
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