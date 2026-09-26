"""Text normalization for noisy business names and addresses across US, India, and France.

Handles:
- Unicode accent normalization (e.g. é -> e, ô -> o for French test set)
- Stripping legal entity suffixes (US, India, France)
- Normalizing common address abbreviations (US, India, France)
- Stripping domain noise (e.g. .com, .in, www.)
- Number / PIN / postal code extraction for precise geo-matching
"""
import re
import unicodedata
from typing import List

LEGAL_SUFFIXES = [
    # Multi-word legal forms
    "private limited", "pvt ltd", "pvt limited", "private ltd",
    "limited liability company", "limited liability partnership",
    "societe anonyme", "societe par actions simplifiee",
    "societe a responsabilite limitee", "et fils", "and fils",
    # Single-word / short acronyms
    "incorporated", "corporation", "limited", "private",
    "company", "assoc", "associates", "industries", "enterprises",
    "pvt", "ltd", "corp", "inc", "co", "llc", "llp", "plc", "pllc",
    "gmbh", "sarl", "sas", "sasu", "eurl", "sci", "snc", "gie",
    "sca", "scs", "ets", "proprietor", "prop", "m s",
]

# Compile legal suffix regex once sorted by longest first
_LEGAL_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

ADDRESS_ABBREVIATIONS = {
    # US / General road & building terms
    r"\brd\b": "road", r"\bst\b": "street", r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard", r"\bapt\b": "apartment", r"\bfl\b": "floor",
    r"\bflr\b": "floor", r"\bste\b": "suite", r"\bdr\b": "drive",
    r"\bln\b": "lane", r"\bct\b": "court", r"\bcir\b": "circle",
    r"\bpkwy\b": "parkway", r"\bhwy\b": "highway", r"\bbldg\b": "building",
    r"\bno\b": "number", r"\bnum\b": "number", r"\bsq\b": "square",
    r"\bpl\b": "place",
    # India specific landmarks and address terms
    r"\bopp\b": "opposite", r"\bnr\b": "near", r"\bb/h\b": "behind",
    r"\bh[\.\s]*no\b": "house number", r"\bplot[\.\s]*no\b": "plot number",
    r"\bflat[\.\s]*no\b": "flat number", r"\bflt[\.\s]*no\b": "flat number",
    r"\bsec\b": "sector", r"\bmrg\b": "marg", r"\bcol\b": "colony",
    r"\bstn\b": "station", r"\bpo\b": "post office",
    # France specific road & address terms
    r"\bbd\b": "boulevard", r"\bbvd\b": "boulevard", r"\bav\b": "avenue",
    r"\br\b": "rue", r"\bimp\b": "impasse", r"\bch\b": "chemin",
    r"\bchem\b": "chemin", r"\brte\b": "route", r"\bpass\b": "passage",
    r"\ball\b": "allee",
}

_ADDRESS_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), repl)
    for pattern, repl in ADDRESS_ABBREVIATIONS.items()
]

_DOMAIN_PATTERN = re.compile(
    r"(https?://(?:www\.)?|\bwww\.|\.(com|in|org|net|fr|co\.in|co|gov|edu)\b)",
    re.IGNORECASE,
)


def _basic_clean(text: str) -> str:
    """Strips accents, URL domains, punctuation, and excess whitespace."""
    if text is None or not isinstance(text, str):
        return ""
    # Ultra-fast Unicode NFKD normalization to remove accents (é -> e, ô -> o, etc.)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower().strip()
    # Strip URL domains and prefixes
    text = _DOMAIN_PATTERN.sub(" ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def normalize_name(name: str) -> str:
    """Cleans business name and strips legal suffixes."""
    text = _basic_clean(name)
    text = _LEGAL_SUFFIX_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_address(address: str) -> str:
    """Cleans business address and standardizes common abbreviations."""
    text = _basic_clean(address)
    for pattern, repl in _ADDRESS_PATTERNS:
        text = pattern.sub(repl, text)
    return re.sub(r"\s+", " ", text).strip()


def extract_numbers(text: str) -> List[str]:
    """Extracts numerical sequences (house numbers, postal codes, PIN codes)."""
    if not text or not isinstance(text, str):
        return []
    return re.findall(r"\b\d+\b", text)


def tokenize(text: str) -> List[str]:
    return text.split() if text else []

