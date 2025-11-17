
import re

def extract_account_number(text: str, page_number: int = None) -> str:
    """
    Extract and normalize the account number or ORIGINAL number from page text.
    Handles:
    - 'Account Number: ####'
    - 'ORIGINAL: ####'
    - 'Account ####'
    - 'Apex Clearing' followed by account number (next line or same line)
    Automatically consolidates if both detections match.
    Prints the detected account number per page for debugging.
    Returns None if not found.
    """

    # --- 1️⃣ Standard patterns ---
    std_patterns = [
        r"Account\s*Number[:\s]*([\dA-Za-z\-]+)",
        r"Account Number:\s*([\dA-Za-z\s]+)",
        r"Account\s*No\.?[:\s]*([\dA-Za-z\-]+)",   # ✅ added: "Account No." or "Account No"s
        r"ORIGINAL:\s*([\dA-Za-z\s]+)",
        r"Account\s+(?!WITH\b)(?=[A-Za-z0-9\-]*\d)([A-Za-z0-9\-]+)",

    ]

    std_account = None
    for p in std_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            std_account = m.group(1).replace(" ", "").strip()
            break

    # --- 2️⃣ Apex Clearing: account on next line ---
    apex_match = re.search(
        r"Apex\s+Clearing[^\n\r]*\n\s*([A-Z0-9\-]{4,})",
        text,
        re.IGNORECASE
    )
    apex_nextline = apex_match.group(1).strip() if apex_match else None

    # --- 3️⃣ Apex Clearing: account on same line ---
    apex_inline = re.search(
        r"Apex\s+Clearing[^\n\r]{0,60}?([A-Z0-9\-]{4,})",
        text,
        re.IGNORECASE
    )
    apex_inline_acc = apex_inline.group(1).strip() if apex_inline else None

    # --- 4️⃣ Combine and normalize ---
    found = {std_account, apex_nextline, apex_inline_acc}
    found = {a for a in found if a}  # remove None

    detected_account = None
    if found:
        normalized = {a.replace("-", "").upper() for a in found}
        if len(normalized) == 1:
            detected_account = list(found)[0]
        else:
            detected_account = std_account or apex_nextline or apex_inline_acc

    # --- 5️⃣ Debug output ---
    if page_number is not None:
        if detected_account:
            print(f"✅ Page {page_number} → Account detected: {detected_account}")
        else:
            print(f"⚠️ Page {page_number} → No account found")

    return detected_account

