#k1helper
def extract_ein_number(text: str) -> str | None:
    """
    Extract EIN from text — tolerant to OCR dash variants and missing dashes.
    Example matches:
      12-3456789, 12–3456789, 12—3456789, 123456789
    """
    import re

    # Normalize all dash variants to a normal hyphen
    normalized = re.sub(r"[–—−‐‒﹘﹣]", "-", text)

    # Try standard EIN pattern (with dash)
    m = re.search(r"\b\d{2}-\d{7}\b", normalized)
    if m:
        return m.group(0)

    # Try continuous 9-digit pattern, and reformat
    m = re.search(r"\b(\d{9})\b", normalized)
    if m:
        raw = m.group(1)
        return f"{raw[:2]}-{raw[2:]}"  # insert dash

    return None

