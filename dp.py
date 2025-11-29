
        # --- Detect Schedule K-1 (Form 1065) ---
    # --- Detect Schedule K-1 (Form 1065 / 1120-S / 1041) and Statement A (QBI) pages ---
    if any(
        kw in lower
        for kw in [
            "schedule k-1",
            "form 1065",
            "form 1120-s",
            "form 1041",
            "statement a",
            "qualified business income",
            "section 199a",
            "qbi pass-through",
            "qbi pass through",
            "partnership",
            "accumulated differences may occur",
            "K-1 rental real estate activity",
            "for owners of pass-through entities",
            "tax paid on form or-oc filed on owner's behalf",
            "don’t submit with your individual tax return or the pte return",
        ]
    ):
        ein_match = re.search(r"\b\d{2}[-–]\d{7}\b", text)
        if ein_match:
            print(f"[DEBUG] classify_text: Detected K-1 Form 1065 EIN={ein_match.group(0)}", file=sys.stderr)
        return "Income", "K-1"
