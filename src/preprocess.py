"""Text normalization for noisy business names and addresses.

These are deliberately simple, regex-based rules. Once you've looked at the
real data (Step 1), extend LEGAL_SUFFIXES / ADDRESS_ABBREVIATIONS with
whatever patterns you actually see — this is meant as a starting point, not
a finished normalizer.
"""
import re

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt", "private",
    "limited", "ltd", "corporation", "corp", "incorporated", "inc",
    "company", "co", "llc", "llp", "plc", "gmbh", "sarl",
]

ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road", r"\bst\b": "street", r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard", r"\bapt\b": "apartment", r"\bfl\b": "floor",
    r"\bste\b": "suite", r"\bdr\b": "drive", r"\bln\b": "lane",
    r"\bhwy\b": "highway", r"\bnr\b": "near", r"\bopp\b": "opposite",
    r"\bbldg\b": "building", r"\bno\b": "number",
}


def _basic_clean(text) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(name) -> str:
    text = _basic_clean(name)
    for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(suf)}\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_address(address) -> str:
    text = _basic_clean(address)
    for pattern, repl in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(pattern, repl, text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str):
    return text.split() if text else []
