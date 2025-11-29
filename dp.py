
# Tracks merged PDF page numbers for each real (path, idx) K-1 page
k1_page_map = {}
def reorder_k1_pages(pages):
    """
    Sort K-1 pages in correct IRS order:
    1. Main K-1 form page
    2. Partner/shareholder information
    3. Supplemental info
    4. Statement A (QBI)
    5. Section 199A
    6. Rental activity
    7. Basis worksheets
    8. Continuation statements
    """

    def get_score(text):
        t = text.lower()

        # 1. Main federal K-1 page
        if "schedule k-1" in t and ("form 1065" in t or "form 1120-s" in t or "form 1041" in t):
            return 1

        # 2. Personal info sections
        if "information about the partner" in t:
            return 2
        if "information about the shareholder" in t:
            return 2
        if "partner's share" in t:
            return 3
        if "shareholder's share" in t:
            return 3

        # 3. Supplement pages
        if "supplemental information" in t:
            return 4
        if "additional information" in t:
            return 4

        # 4. QBI pages
        if "statement a" in t:
            return 5
        if "section 199a" in t:
            return 5
        if "qbi" in t:
            return 5

        # 5. Rental real estate pages
        if "rental real estate activity" in t:
            return 6

        # 6. Basis worksheets
        if "basis worksheet" in t:
            return 7
        if "at-risk basis" in t:
            return 7

        # 7. Continuation statements
        if "continuation" in t:
            return 8

        # 8. Fallback → leave last
        return 99

    # Score every page
    scored = []
    for (p, idx, _) in pages:
        text = extract_text(p, idx)
        score = get_score(text)
        scored.append((score, p, idx))

    # Sort by score only (KEEP original order if score ties)
    scored.sort(key=lambda x: x[0])

    # Return sorted as original format
    return [(p, idx, "K-1") for (_, p, idx) in scored]
def k1_page_priority(text):
    t = text.lower()

    # --- TRUE MAIN K-1 FORM PAGE ---
    if (
        "schedule k-1" in t
        and ("form 1065" in t or "form 1120-s" in t or "form 1041" in t)
        and (
            "information about the partner" in t
            or "information about the shareholder" in t
            or ("part i" in t and "information about" in t)
        )
    ):
        return 1

    # Supplemental info (NOT main)
    if "supplemental information" in t:
        return 3

    # Rental real estate pages
    if "rental real estate" in t:
        return 4

    # QBI / Section 199A
    if (
        "statement a" in t
        or "section 199a" in t
        or "qualified business income" in t
        or "qbi" in t
    ):
        return 5

    # Other continuation
    if "continuation" in t:
        return 6

    # Fallback last
    return 99
def extract_k1_company(text: str) -> str | None:
    # Normalize line breaks and spaces
    clean = re.sub(r"[^\w\s,&.'\-]", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()

    # 1. Strongest pattern: any line ending with LLC/LP/INC/CORP/etc.
    multi = re.findall(
        r"([A-Z][A-Z\s,&.'\-]{2,80}?(?:LLC|L\.L\.C\.|INC|CORP|LP|L\.P\.|LLP|FUND))",
        text,
        flags=re.IGNORECASE,
    )
    if multi:
        # choose longest = most complete name
        return max(multi, key=len).strip()

    # 2. Multi-line OCR-broken names: join consecutive lines
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        joined = (lines[i] + " " + lines[i+1]).strip()
        m = re.search(
            r"([A-Z][A-Z\s,&.'\-]{2,80}?(?:LLC|INC|CORP|LP|LLP|FUND))",
            joined,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()

    # 3. fallback for things like “Desire Homes North Salem LLC”
    words = clean.split()
    for i in range(len(words)):
        for j in range(i+2, min(i+8, len(words))):
            segment = " ".join(words[i:j])
            if re.search(r"(LLC|INC|CORP|LP|LLP)$", segment, flags=re.IGNORECASE):
                return segment

    return None

def clean_k1_company_name(raw: str) -> str:
    """
    Clean K-1 company name by:
    - Removing multi-line noise ('Name of Partnership', 'Partnership EIN', etc.)
    - Keeping only the longest block ending with LLC/LP/INC/CORP
    - Removing leading/trailing garbage
    """

    if not raw:
        return None

    import re

    # Collapse newlines → spaces
    raw = raw.replace("\n", " ")
    raw = re.sub(r"\s+", " ", raw).strip()

    # Remove known garbage phrases
    garbage = [
        "name of partnership",
        "partnership ein",
        "information about the partner",
        "information about the shareholder",
        "schedule k-1",
        "form 1065",
        "form 1120-s",
        "form 1041"
    ]

    lower = raw.lower()
    for g in garbage:
        if g in lower:
            # Remove the entire garbage phrase + anything BEFORE it
            idx = lower.index(g)
            raw = raw[idx + len(g):].strip()
            lower = raw.lower()

    # Strong match: pick the longest LLC/INC/CORP/LP ending
    m = re.findall(
        r"[A-Za-z0-9& ,.'\-]{3,100}?(?:LLC|L\.L\.C\.|INC|CORP|LP|LLP)",
        raw,
        flags=re.IGNORECASE
    )
    if m:
        # choose longest match
        return max(m, key=len).strip()

    return raw.strip()
