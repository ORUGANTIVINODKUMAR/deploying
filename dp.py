
def is_unused_page(text: str) -> bool:
    """
    Detect pages that are just year-end messages, instructions,
    or generic investment details (not real 1099 forms).
    """
    import re
    lower = text.lower()
    # normalize multiple spaces to single
    norm = re.sub(r"\s+", " ", lower)

    investment_details = re.search(r"\b\d{4}\s+investment details", norm)

    return (
        "understanding your form 1099" in norm
        or "year-end messages" in norm
        or "important: if your etrade account transitioned" in norm
        or "please visit etrade.com/tax" in norm
        or "tax forms for robinhood markets" in norm
        or "robinhood retirements accounts" in norm
        or "new for 2023 tax year" in norm
        or "new for 2024 tax year" in norm
        or "new for 2025 tax year" in norm
        or "that are necessary for tax" in norm
        or "please note there may be a slight timing" in norm
        or "account statement will not have included" in norm
        #1099-SA
        or "fees and interest earnings are not considered" in norm
        or "an hsa distribution" in norm
        or "death is includible in the account" in norm
        or "the account as of the date of death" in norm
        or "amount on the account holder" in norm
        #1099-Mortgage
        or "for clients with paid mortgage insurance" in norm
        or "you can also contact the" in norm
        #or "" in norm
        #fidelity
        or "the amount of tax-exempt interest paid to you must be reported on the applicable" in norm 
        or "form 1040, u.s. individual income tax return" in norm
        or "the amount of tax-exempt alternative minimum tax" in norm
        or "interest paid to you must be taken into account in computing the" in norm
        #Td Ameritrade
        or "The tax character of the distribution has been allocated based on information provided by the security issuer" in norm
        or "the tax character of the distribution has been allocated" in norm
        or "tax lot closed is a specified lot" in norm
        or "allocated based on information provided by the security issuer" in norm  
        or "importanttaxreturndocumentenclosed" in norm
        #Apex clearing
        or "please verify your personal information for accuracy and contact us" in norm
        or "if you hold these securities or another security that is" in norm
        or "information that may be helpful to you for filing your tax return" in norm
       #merrill
        #or "" in norm
        or "to view additional tax resources available online" in norm
        or "merrill is only required to revise 1099 tax reporting statements when" in norm
        or "amount shown on line 1b of form 1099 div as a qualified dividend indicates" in norm       
        or "california residents be advised that payers are required to report" in norm
        or "this page was intentionally left blank" in norm
        or "this page was intentionally left blank" in norm
       #w2
        or "for the latest information about developments related to form" in norm
        or "you and any qualifying children must have valid social security numbers" in norm
       
       #w2
        or "may be requested by the mortgagor" in norm
       
        or "you should contact a competent" in norm
        or "tax lot closed on a first in" in norm
        or "your form 1099 composite may include the following internal revenue service " in norm
        or "schwab provides your form 1099 tax information as early" in norm
        or "if you have any questions or need additional information about your" in norm
        or "schwab is not providing cost basis" in norm
        or "the amount displayed in this column has been adjusted for option premiums" in norm
        or "you may select a different cost basis method for your brokerage" in norm
        or "to view and change your default cost basis" in norm
        or "this information is not intended to be a substitue for specific individualized" in norm
        or "shares will be gifted based on your default cost basis" in norm
        or "if you sell shares at a loss and buy additional shares" in norm
        or "we are required to send you a corrected from with the revisions clearly marked" in norm
        or "referenced to indicate individual items that make up the totals appearing" in norm
        or "issuers of the securities in your account reallocated certain income distribution" in norm
        or "the amount shown may be dividends a corporation paid directly" in norm
        or "if this form includes amounts belonging to another person" in norm
        or "spouse is not required to file a nominee return to show" in norm
        or "character when passed through or distributed to its direct or in" in norm
        or "brokers and barter exchanges must report proceeds from" in norm
        or "first in first out basis" in norm
        or "see the instructions for your schedule d" in norm
        or "other property received in a reportable change in control or capital" in norm
        or "enclosed is your" in norm and "consolidated tax statement" in norm
        or "filing your taxes" in norm and "turbotax" in norm
        or ("details of" in norm and "investment activity" in norm)
        or bool(investment_details)
    )


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
    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"[\n\r]{2,}", "\n", text)
    lower = text.lower()

    lower = text.lower()

    # --- ⛔ 0️⃣ Skip extraction for forms that SHOULD NOT have account numbers ---
    skip_keywords = [
        "form 1099-r", "1099 r",
        "form 1098", "1098-t", "1098 t", "1098-e", "1098 e", "1098-q", "1098 q", 
        "form 1098 mortgage", "mortgage interest",
        "tuition statement", "qualified tuition",
        "education institution",
        "form w-2", "w-2",
        "5498-sa",
    ]

    for k in skip_keywords:
        if k in lower:
            if page_number is not None:
                print(f"⛔ Page {page_number} → Skip account extraction (detected: {k})")
            return None
    
    # --- 1️⃣ Standard patterns ---
    std_patterns = [
        r"Account\s*Number[:\s]*([\dA-Za-z\-]+)",
        r"Account Number:\s*([\dA-Za-z\s]+)",
        r"Account\s*No\.?[:\s]*([\dA-Za-z\-]+)",   # ✅ added: "Account No." or "Account No"s
        r"ORIGINAL:\s*([\dA-Za-z\s]+)",
        r"Account\s+(?!WITH\b)(?=[A-Za-z0-9\-]*\d)([A-Za-z0-9\-]+)",
        r"Account\s*No\.?\s*[:\-]?\s*([A-Za-z0-9]{1,3}[A-Za-z0-9\-]{3,})",
        r"AccountNo\.?\s*[:\-]?\s*([A-Za-z0-9]{1,3}[A-Za-z0-9\-]{3,})",
        r"Account\s*No\.?[^\n\r]*\n\s*([A-Za-z0-9]{1,3}[A-Za-z0-9\-]{3,})",
        r"AccountNo\.?[^\n\r]*\n\s*([A-Za-z0-9]{1,3}[A-Za-z0-9\-]{3,})",
        r"AccountNo\.?\s*([A-Za-z0-9]{4,})",
        r"Account\s*No\s+([A-Za-z0-9]{4,})",
        r"Account\s+([A-Za-z0-9]{1,4}\s+[0-9]{3,})",
        r"Account\s*No\.?\s*([A-Za-z0-9]+[\u2010\u2011\u2012\u2013\u2014\-][A-Za-z0-9\-]+)",
        # Case 1: Same line → "Account No. 76W-59336"
        r"Account\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-]{4,})",
        # Case 2: Next line → "Account No.\n76W-59336"
        r"Account\s*No\.?[^\n\r]*\n\s*([A-Za-z0-9\-]{4,})",
        # Merrill-specific labels
        r"(?:Merrill(?:\s+Lynch|\s+Edge)?)?\s*Account\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-]{4,})",
        # OCR distortions → Accoumt / Accouni / Accourt
        r"Accou[nmrt]+\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-]{4,})",
        # "Customer Account Number"
        r"Customer\s*Account\s*Number[:\s]*([A-Za-z0-9\s\-]{4,})",
        # Plain "Account XXXXX"
        r"Account\s+([A-Za-z0-9]{4,})",

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




# consolidated-1099 forms bookmark
def has_nonzero_misc(text: str) -> bool:
    patterns = [
        r"1\.RENTS\s*\$([0-9,]+\.\d{2})",
        r"2\.ROYALTIES\s*\$([0-9,]+\.\d{2})",
        r"3\.OTHER INCOME\s*\$([0-9,]+\.\d{2})",
        r"4\.FEDERAL INCOME TAX WITHHELD\s*\$([0-9,]+\.\d{2})",
        r"8\.SUBSTITUTE PAYMENTS.*\$\s*([0-9,]+\.\d{2})",
    ]
    return _check_nonzero(patterns, text)
def has_nonzero_oid(text: str) -> bool:
    patterns = [
        r"1\.ORIGINAL ISSUE DISCOUNT.*\$\s*([0-9,]+\.\d{2})",
        r"2\.OTHER PERIODIC INTEREST.*\$\s*([0-9,]+\.\d{2})",
        r"4\.FEDERAL INCOME TAX WITHHELD.*\$\s*([0-9,]+\.\d{2})",
        r"5\.MARKET DISCOUNT.*\$\s*([0-9,]+\.\d{2})",
        r"6\.ACQUISITION PREMIUM.*\$\s*([0-9,]+\.\d{2})",
        r"8\.OID ON.*\$\s*([0-9,]+\.\d{2})",
        r"9\.INVESTMENT EXPENSES.*\$\s*([0-9,]+\.\d{2})",
        r"10\.BOND PREMIUM.*\$\s*([0-9,]+\.\d{2})",
        r"11\.TAX-EXEMPT OID.*\$\s*([0-9,]+\.\d{2})",
    ]
    return _check_nonzero(patterns, text)


def extract_1099b_section(text: str) -> str:
    """
    Extract only the 1099-B summary table section.
    """
    pattern = r"SUMMARY OF PROCEEDS.*?Grand total.*?(?:\n|$)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(0) if m else ""


def find_nonzero_after(label: str, text: str, window: int = 120) -> bool:
    """
    Finds the label (e.g. 'Short A', 'Box A', etc.) and scans the next
    `window` characters for any non-zero numeric value.
    Works even when OCR breaks the line.
    """
    pattern = re.compile(label, re.IGNORECASE)

    for match in pattern.finditer(text):
        start = match.end()
        chunk = text[start:start + window]

        # extract numbers like 42,118.15 or 0.00 or $19,295.46
        nums = re.findall(r"-?\$?\s*[0-9][\d,]*\.\d{2}", chunk)

        for num in nums:
            val = float(num.replace("$", "").replace(",", "").strip())
            if val != 0:
                return True

    return False

def has_nonzero_1099b(text: str) -> bool:
    # FORMAT 1 — Vanguard / Robinhood / TD Ameritrade
    legacy_labels = [
        "Short A",
        "Short B",
        "Short C",
        "Total Short-term",
        "Long D",
        "Long E",
        "Long F",
        "Total Long-term",
        "Undetermined B",
        "Undetermined C",
        "Total Undetermined-term",
        "Grand total",
    ]

    # FORMAT 2 — Morgan Stanley / E*TRADE / Fidelity / Schwab
    box_labels = [
        r"Box A\b",
        r"Box A - Ordinary",
        r"Box B\b",
        r"Box B - Ordinary",
        r"Box D\b",
        r"Box D - Ordinary",
        r"Box E\b",
        r"Box E - Ordinary",

        r"Total Short\s*-?\s*Term",
        r"Total Long\s*-?\s*Term",
        r"Total Unknown\s*-?\s*Term",
        r"Unknown Term",
    ]

    # FORMAT 3 — Fidelity / Schwab "Full Text" Section Names
    fidelity_labels = [
        r"Short-term transactions for which basis is reported to the IRS",
        r"Short-term transactions for which basis is not reported to the IRS",
        r"Long-term transactions for which basis is reported to the IRS",
        r"Long-term transactions for which basis is not reported to the IRS",
        r"Transactions for which basis is not reported to the IRS and Term is Unknown",
    ]

    # Scan all available formats
    all_labels = legacy_labels + box_labels + fidelity_labels

    for label in all_labels:
        if find_nonzero_after(label, text, window=120):
            return True

    return False

def has_nonzero_div(text: str) -> bool:
    """
    Detects non-zero dividend amounts from:
      - 1a Ordinary Dividends
      - 1b Qualified Dividends
      - 2a Capital Gain Distributions
      - 3 Nondividend Distributions
      - 5 Section 199A Dividends
      - 7 Foreign Tax Paid
    """

    import re
    t = text.lower()

    # Robust OCR-safe patterns
    patterns = [
        r"1a[^0-9a-z]+.*ordinary dividends.*?([-]?\d[\d,]*\.\d{2})",
        r"1b[^0-9a-z]+.*qualified dividends.*?([-]?\d[\d,]*\.\d{2})",
        r"2a[^0-9a-z]+.*capital gain.*?([-]?\d[\d,]*\.\d{2})",
        r"3[^0-9a-z]+.*non.?dividend.*?([-]?\d[\d,]*\.\d{2})",
        r"5[^0-9a-z]+.*199a dividends.*?([-]?\d[\d,]*\.\d{2})",
        r"7[^0-9a-z]+.*foreign tax paid.*?([-]?\d[\d,]*\.\d{2})",
        r"11[^0-9a-z]+.*exempt.?interest dividends.*?([-]?\d[\d,]*\.\d{2})",
    ]

    for p in patterns:
        m = re.search(p, t, flags=re.DOTALL)
        if m:
            try:
                value = float(m.group(1).replace(",", ""))
                if value != 0.0:
                    return True
            except:
                pass

    return False


def has_nonzero_int(text: str) -> bool:
    import re

    t = text.lower()

    # Match ONLY numbers at the END of the line, not in the middle
    m = re.search(
        r"interest\s+income[^\n\r]*?([-]?[0-9oO]{1,5}\.?[0-9oO]+)\s*$",
        t,
        flags=re.IGNORECASE | re.MULTILINE
    )

    if not m:
        return False

    val = (
        m.group(1)
        .replace("O", "0")
        .replace("o", "0")
        .replace(",", "")
        .strip()
    )

    try:
        return float(val) > 0.0
    except:
        return False


import re

def _check_nonzero(patterns, text: str) -> bool:
    """
    Scan text with given regex patterns and return True
    if any captured numeric value is greater than zero.
    Works even when '$' is missing or OCR introduces stray spaces/letters.
    """
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE | re.DOTALL)
        for m in matches:
            # Clean OCR artifacts like "3.9O" or "$ 3.90"
            val_str = (
                str(m)
                .replace(",", "")
                .replace("$", "")
                .replace("O", "0")   # OCR 'O' → '0'
                .strip()
            )

            try:
                val = float(val_str)
                if val > 0.0:
                    print(f"[DEBUG] ✅ Nonzero value detected: {val_str}", file=sys.stderr)
                    return True
            except ValueError:
                continue
    return False


# --- Post-processing cleanup for bookmarks ---

def classify_text_multi(text: str) -> list[str]:
    """
    Return list of detected IRS forms on a page.
    Handles INT, DIV, B, MISC, OID.
    """
    import re, sys

    lower = text.lower()
    matches = []

    # --- Debug Area ---
    print("\n========== DEBUG PAGE ==========", file=sys.stderr)
    print(text[:500], file=sys.stderr)
    print("LOWER:", lower[:200], file=sys.stderr)
    print("===============================\n", file=sys.stderr)

    # ================================================================
    # 1️⃣ 1099-INT detection (NEW: ONLY based on non-zero interest)
    # ================================================================
    has_int = has_nonzero_int(text)

    # ================================================================
    # 2️⃣ 1099-DIV detection
    # ================================================================
    has_div = (
        ("1099-div" in lower or
         "form 1099-div" in lower or
         "ordinary dividends" in lower or
         "qualified dividends" in lower)
        and has_nonzero_div(text)
    )

    # Additional DIV safety (helps OCR cases)
    div_pattern = re.search(r"1099[\s\-]*div", lower)
    div_section = re.search(r"dividends\s+and\s+distributions", lower)
    if (div_pattern or div_section):
        if re.search(r"(ordinary|qualified)\s+divid.{0,10}[0-9,]+\.\d{2}", lower):
            matches.append("1099-DIV")

    # ================================================================
    # 3️⃣ 1099-B detection
    # ================================================================
    if "1099-b" in lower or "form 1099-b" in lower or has_nonzero_1099b(text):
        if has_nonzero_1099b(text):
            matches.append("1099-B")

    # ================================================================
    # 4️⃣ 1099-MISC
    # ================================================================
    if "1099-misc" in lower or "form 1099-misc" in lower:
        if has_nonzero_misc(text):
            matches.append("1099-MISC")

    # ================================================================
    # 5️⃣ 1099-OID
    # ================================================================
    if "1099-oid" in lower or "form 1099-oid" in lower:
        if has_nonzero_oid(text):
            matches.append("1099-OID")
    # ================================================================
    # TD Ameritrade / Consolidated Detail Pages
    # ================================================================
    
    if (
        "detail for dividends and distributions" in lower
        #or "detail for interest income" in lower
        #or "reallocation of a dividend and it’s tax character is determined by the issuer" in lower
        or "note that a payment characterized as a “qualified dividend”" in lower
    ):
        # return a single special bookmark
        return ["Consolidated-DIV"]

    # ================================================================
    # 6️⃣ Combined INT + DIV
    # ================================================================
    # --- New INT + DIV independent logic ---

    if has_int:
        matches.append("1099-INT")

    if has_div:
        matches.append("1099-DIV")

    return matches
