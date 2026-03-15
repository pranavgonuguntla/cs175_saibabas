import os
import json
import hashlib
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from keybert import KeyBERT

from config import USE_RAKE, CACHE_DIR, KEYWORD_CACHE_PATH
from data.preprocessing import preprocess_review

os.makedirs(CACHE_DIR, exist_ok=True)

kw_model = KeyBERT(model="all-MiniLM-L6-v2")

if USE_RAKE:
    from rake_nltk import Rake
    rake_model = Rake()

keyword_cache = {}


def load_keyword_cache(path=KEYWORD_CACHE_PATH):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_keyword_cache(cache, path=KEYWORD_CACHE_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def extract_keywords_keybert(text: str, top_k: int = 8):
    """
    Returns a list of keyword strings using KeyBERT.
    We remove very short tokens and keep unigrams/bigrams for simplicity.
    """
    text = preprocess_review(text)
    if not text:
        return []
    kws = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_k
    )
    return [k for (k, _) in kws]


def extract_keywords_rake(text: str, top_k: int = 8):
    """
    Optional RAKE keyword extraction.
    Returns a list of keyword strings.
    """
    text = preprocess_review(text)
    if not text:
        return []
    rake_model.extract_keywords_from_text(text)
    return rake_model.get_ranked_phrases()[:top_k]


def get_cached_keywords(text: str, top_k: int = 8, use_rake: bool = False):
    text = preprocess_review(text)
    hid = text_hash(text)

    if hid in keyword_cache:
        return keyword_cache[hid]

    if use_rake:
        keywords = extract_keywords_rake(text, top_k=top_k)
    else:
        keywords = extract_keywords_keybert(text, top_k=top_k)

    keyword_cache[hid] = keywords
    return keywords


def flush_keyword_cache():
    save_keyword_cache(keyword_cache)
    print(f"Saved {len(keyword_cache)} keyword entries to {KEYWORD_CACHE_PATH}")


# Load cache on module import
keyword_cache.update(load_keyword_cache())
print(f"Loaded {len(keyword_cache)} cached keyword entries")
