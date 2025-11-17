
# --------------------------- Consolidated-1099 issuer name --------------------------- #
def extract_consolidated_issuer(text: str) -> str | None:
    """
    Detects the brokerage or custodian issuing a Consolidated 1099.
    Supports all major U.S. firms and custodians (Fidelity, Schwab, E*TRADE, Robinhood, etc.)
    Returns a clean, friendly issuer name if detected, else None.
    """

    lower = text.lower()

    # --- Master company map (key phrases -> friendly label) ---
    issuers = {
        # Core brokerages
        r"fidelity(\s+investments)?": "Fidelity Investments",
        r"charles\s+schwab": "Charles Schwab & Co., Inc.",
        r"etrade|e\*trade": "E*TRADE (Morgan Stanley)",
        r"robinhood": "Robinhood Markets, Inc.",
        r"edward\s+jones": "Edward Jones",
        r"janney\s+montgomery\s+scott": "Janney Montgomery Scott LLC",
        r"stifel": "Stifel Financial Corp.",
        r"td\s*ameri?trade\s*clearing": "TD Ameritrade Clearing, Inc.",
        r"td\s*ameri?trade\s+clearing,\s*inc\.?": "TD Ameritrade Clearing, Inc.",
        r"td\s*ameri?trade": "TD Ameritrade (Charles Schwab)",
        r"tdameri?trade": "TD Ameritrade",
        r"td\s*ameri?trade\s+inc": "TD Ameritrade",
        r"ameritrade\s+clearing": "TD Ameritrade Clearing, Inc.",
        r"ameritrade": "TD Ameritrade",
        r"td\s+ameritrade\s+clearing": "TD Ameritrade Clearing, Inc.",
        r"ameritrade": "TD Ameritrade",
        r"td\s+ameritrade": "TD Ameritrade (Charles Schwab)",
        r"merrill|bank\s+of\s+america": "Merrill Lynch (Bank of America)",
        r"vanguard": "Vanguard Brokerage Services",
        r"interactive\s+brokers": "Interactive Brokers LLC",
        r"ally\s+invest": "Ally Invest",
        r"tastytrade|tastyworks": "Tastytrade, Inc.",
        r"morgan\s+stanley\s+wealth": "Morgan Stanley Wealth Management",
        r"raymond\s+james": "Raymond James & Associates, Inc.",
        r"pershing": "Pershing LLC",
        r"lpl\s+financial": "LPL Financial LLC",
        r"apex\s+clearing": "Apex Clearing",
        r"ameriprise": "Ameriprise Financial Services, Inc.",
        # UBS – must be BEFORE short patterns, but AFTER specific patterns like Ameritrade
        r"\bubs\s+financial\s+services\b": "UBS Financial Services Inc.",
        r"\bubs\s+financial\b": "UBS Financial Services Inc.",
        #r"ubs": "UBS Financial Services Inc.",
        r"wells\s+fargo": "Wells Fargo Advisors, LLC",
        r"j\.?p\.?\s*morgan|chase": "J.P. Morgan Securities LLC",
        r"goldman\s+sachs|marcus": "Goldman Sachs (Marcus / PWM)",
        r"sofi": "SoFi Invest",
        r"public\.com": "Public.com",
        r"acorns": "Acorns Advisers, LLC",
        r"betterment": "Betterment LLC",
        r"wealthfront": "Wealthfront Brokerage LLC",
        r"m1\s+finance": "M1 Finance LLC",
        r"firstrade": "Firstrade Securities Inc.",
        r"tradestation": "TradeStation Securities, Inc.",
        r"intelligent\s+portfolios": "Charles Schwab Intelligent Portfolios",
        r"fidelity\s+go|fidelity\s+spire": "Fidelity Go / Fidelity Spire",
        r"self[-\s]*directed\s+investing": "J.P. Morgan Self-Directed Investing",
        r"eaton\s+vance": "Eaton Vance Brokerage (Morgan Stanley)",
        r"hsbc\s+securities": "HSBC Securities (USA) Inc.",
        r"citi\s+personal\s+wealth": "Citi Personal Wealth Management",
        r"baird|robert\s+w\.?\s*baird": "Robert W. Baird & Co.",
        r"oppenheimer": "Oppenheimer & Co. Inc.",
        r"cowen": "Cowen and Company, LLC",
        r"jefferies": "Jefferies Financial Group Inc.",
        r"evercore": "Evercore ISI",
        r"hennion\s+&\s+walsh": "Hennion & Walsh, Inc.",
        r"zions": "Zions Direct (Zions Bancorporation)",
        r"fifth\s+third\s+securit": "Fifth Third Securities, Inc.",
        r"regions\s+investment": "Regions Investment Services, Inc.",
        r"pnc\s+invest": "PNC Investments, LLC",
        r"synovus": "Synovus Securities, Inc.",
        r"citizens\s+securit": "Citizens Securities, Inc.",
        r"first\s+horizon": "First Horizon Advisors, Inc.",
        
        r"merrill": "Merrill Lynch",
    }

    # --- 1️⃣ Direct known match (fast path) ---
    for pattern, name in issuers.items():
        if re.search(pattern, lower):
            return name

    # --- 2️⃣ Special explicit handling (legacy from old version) ---
    if re.search(r"morgan\s+stanley\s+capital\s+management,\s*llc", lower):
        return "Morgan Stanley Capital Management, LLC"

    # --- 3️⃣ Heuristic fallback for unknown but structured consolidated 1099s ---
    if "consolidated 1099" in lower or "composite 1099" in lower:
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            # skip headings / noise
            if re.search(r"(form|1099|copy|page|\baccount\b)", s, re.IGNORECASE):
                continue
            # probable issuer-style line
            if re.search(r"(LLC|Bank|Securities|Wealth|Brokerage|Advisors?)", s):
                return re.sub(r"[^\w\s,&.\-]+$", "", s)

    return None
