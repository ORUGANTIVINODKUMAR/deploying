
# --- Classification Helper
def is_1099r_page(text: str) -> bool:
    # Normalize whitespace & lowercase
    lower = " ".join(text.lower().split())

    blockA = [
        "form 1099-r",
        "pensions, annuities",
        "gross distribution",
        "distribution code",
        "ira/",
        "simple",
        "taxable amount",
    ]

    blockB = [
        "gross distribution",
        "taxable amount",
        "capital gain (included in box 2a)",
        "employee contributions/designated roth",
        "distribution code(s)",
        "federal income tax withheld",
        "amount allocable to irr",
        "state tax withheld",
        "state distribution",
    ]

    # Block A → must match at least 1 keyword
    matchA = any(pat in lower for pat in blockA)

    # Block B → must match at least 3 keywords
    matchB_count = sum(1 for pat in blockB if pat in lower)
    matchB = matchB_count >= 3

    # Final requirement
    return matchA and matchB

def classify_text(text: str) -> Tuple[str, str]:
    normalized = re.sub(r'\s+', '', text.lower())
    t = text.lower()
    
    lower = text.lower()
      # Detect W-2 pages by their header phrases
    t = re.sub(r"\s+", " ", text.lower()).strip()
    # --- Detect Schedule K-1 (Form 1065) ---
    if "schedule k-1" in t or "form 1065" in t:
        return "Income", "K-1"
    if "statement a" in t and "qbi" in t:
        return "Income", "K-1"

    #Property Tax
    if (
        "total allowable community college" in t
        or "school district property tax paid" in t
        or "district property tax paid" in t
        or "parcel id property property" in t
        or "axing unit taxrate previous tax" in t
        or "homestead exempt" in t
        or "real property tax proper iy location" in t
        or "property assessment" in t
        or "real property taxsssss" in t
        or "REAL PROPERTY TAX PROPERTY LOCATION" in t
        or "real property tax property location" in t
        or "www.dctreasurer.ora" in t
        or "homesteadexempt" in t
    ):
        return "Expenses", "Property Tax"

    # --------------------------- 1095-C --------------------------- #
    if (
        "form 1095-c" in lower
        or "employer-provided health insurance offer and coverage" in lower
        or "employee offer of coverage" in lower
        or "covered individuals" in lower
        or "employer-provided health insurance offer" in lower
        or "do not attach to your tax return" in lower
        
    ):
        return "Others", "1095-C"
    # --------------------------- 1095-C --------------------------- #
    


    if (
        "fees and interest earnings are not considered contributions" in t
        or "contact a competent tax advisor or the irs" in t
        or "retirement plans for small business" in t
        or "civil service retirement benefits" in t
        or "general rule for pensions and annuities" in t
        or "hsas and other tax-favored health plan" in t
        # New lines from E*TRADE statement:
        or "the following tax documents are not included in this statement" in t
        or "forms 1099-r, 1099-q, 1042-s, 2439, 5498" in t
        or "e*trade from morgan stanley is pleased to provide" in t
        or "warning - corrected tax forms possible" in t
        or "prepared based upon information provided by the issuer" in t
        or "we will be required to send you one or more corrections" in t
        # Existing unused checks...
        or "the following tax documents are not included in this statement" in t
        or "e*trade from morgan stanley" in t
        or "1099 consolidated tax statement" in t
        or "*** warning - corrected tax forms possible ***" in t
        or "prepared based upon information provided by the issuer" in t
        or "will be required to send you one or more corrections" in t
        #1042-S
        or "explanation of codes" in t
        or "einbehaltuxvxng der steuern" in t
    ):
        return "Others", "Unused"

    if is_1099r_page(t):
        return "Income", "1099-R"


    # --- Detect 1099-G (State Income Tax Refund) ---
    g1099 = [
        "1099 g",
        "form 1099 g",
        "1099-g",
        "form 1099-g",
    ]
    for pat in g1099:
        if pat in lower:
            return "Income", "1099-G"

    if (
        "child care" in lower
        or "day care" in lower
        or "to the parents" in lower
      
        or "provider information" in lower
        or "total payments paid by" in lower
        #or "dates of service" in lower
        or "late payment fee late payment fee" in lower
        or "assistant business administrator" in lower
        or "preschool tuition payments" in lower
        or "the student named above has" in lower
        or "ach - returned - online payment" in lower
        or "registration fee new enrollmeny" in lower
        #or "" in lower
        
    ):
        print(f"[DEBUG] CHILD CARE EXPENSE DETECTED in page: {text[:120]}...", file=sys.stderr)
        return "Expenses", "Child Care Expenses"
   
    unuseddiv = [
        "fundrise strives to provide your",
        "#although the fundrise team seeks to",
        "fundrise receives updated information for",
        #1099-SA
        "fees and interest earnings",
        "if you have questions regarding",
        "you should contact a competent tax advisor"
        "Fees and interest earnings are not considered contributions",
        "contact a competent tax advisor or the irs",
        "contributions or distributions and are not",
        "if you have questions regarding specific circumstances",
        "if you have questions regarding specific circumstances",
        "if you have questions regarding specific circumstances",
        #1098-T
        #"for the latest information about developments"
        "may result in an increase in tax",
        "reimbursements or refunds for the calendar",
        "rippling",
        #W2
        "if this form includes amounts belonging to",
        "a spouse is not required to file a",
        "such a legislation enacted after",
        #1099-INT
        "continued on the back of copy",
       
       
    ]
    for pat in unuseddiv:
        if pat in lower:
            return "Others", "Unused"
    # --------------------------- 529 Plan / College Savings --------------------------- #
    # Detect 529 college savings plan statements or transaction notices
    clean_text = re.sub(r'[^A-Za-z0-9\s]', '', text.lower())  # normalize OCR artifacts
   
    if (
        "529" in clean_text
        and (
            #"indiana529" in clean_text
            "indiana 529" in clean_text
            or "529 direct savings plan" in clean_text
            or "education savings authority" in clean_text
            or "college savings" in clean_text
            or "qualified tuition program" in clean_text
            or "investment allocations" in clean_text
            or "investment portfolio" in clean_text
            or "funding information" in clean_text
            or "recurring contribution" in clean_text
            or "bank information" in clean_text
            or "electronic bank transfer" in clean_text
            #or "indiana529directcom" in clean_text
            #or "indiana 529 direct com" in clean_text
            or "indiana education savings" in clean_text
            or "contribution ebt" in clean_text
            or "please see below for details pertaining to" in clean_text
        )
    ):
        return "Expenses", "529-Plan"
    if "#bwnjgwm" in normalized:
        return "Others", "Unused"
    
    if "#rippling" in normalized:
        return "Others", "Unused"
    sa_front_patterns = [
        r"earnings\s+on\s+excess\s+cont",   # will also match 'cont.'
        #r"form\s+1099-?sa",                 # matches '1099-SA' or '1099SA'
        r"fmv\s+on\s+date\s+of\s+death",
    ]

    found_sa_front = any(re.search(pat, lower) for pat in sa_front_patterns)

    # 🔁 Priority: 1099-SA > Unused
    if found_sa_front:
        return "Income", "1099-SA"

   
    # 1) Detect W-2 pages by key header phrases
    if (
        "wages, tips, other compensation" in lower or
        ("employer's name" in lower and "address" in lower)
    ):
        return "Income", "W-2"

    #5498-SA
    # --- 5498-SA detection (more tolerant OCR patterns) ---
    sa5498_front_patterns = [
        r"form\s+[s§5]\s*498-?\s*sa",             # catches “5498-SA”, “S498-SA”, “§498-SA”
        r"form\s+5498sa",                         # no dash
        r"form\s+s498-sa",                        # OCR “5”→“S”
        r"form\s+§498-sa",                        # OCR “5”→“§”
        r"total\s+contributions\s+made\s+in\s+\d{4}",
        r"fair\s+market\s+value\s+of\s+(account|hsa)",
        r"\b2[\.\-)]?\s*rollover\s+contributions",
        r"\b5[\.\-)]?\s*fair\s+market\s+value\s+of\s+(account|hsa)",
        r"\b7[\.\-)]?\s*ira\s+type",
        r"\b11[\.\-)]?\s*required\s+minimum\s+distribution.*\d{4}"
    ]
    if any(re.search(pat, lower) for pat in sa5498_front_patterns):
        return "Expenses", "5498-SA"

   
    if is_unused_page(text):
        return "Unknown", "Unused"
    if '1098-t' in t: return 'Expenses', '1098-T'
   
    # If page matches any instruction patterns, classify as Others → Unused
    instruction_patterns = [
    # full “Instructions for Employee…” block (continued from back of Copy C)
    # W-2 instructions
    #"box 1. enter this amount on the wages line of your tax return",
    #"box 2. enter this amount on the federal income tax withheld line",
    #"box 5. you may be required to report this amount on form 8959",
    ##"box 6. this amount includes the 1.45% medicare tax withheld",
    #"box 8. this amount is not included in box 1, 3, 5, or 7",
    #"you must file form 4137",
    #"box 10. this amount includes the total dependent care benefits",
    "instructions for form 8949",
    "employee w-4 profile to change your employee w-4 profile information",
    "the following information reflects your final pay statement plus employer adjustments",
    "the following information reflects your final pay statement plus statement plus",
    "regulations section 1.6045-1",
    "recipient's taxpayer identification number",
    "fata filing requirement",
    "payer’s routing transit number",
    #"refer to the form 1040 instructions",
    "earned income credit",
    "if your name, SSN, or address is incorrect",
    #"corrected wage and tax statement",
    #"credit for excess taxes",
    #"instructions for employee  (continued from back of copy c) "
    #"box 12 (continued)",
    #"f—elective deferrals under a section 408(k)(6) salary reduction sep",
    "g—elective deferrals and employer contributions (including  nonelective ",
    "deferrals) to a section 457(b) deferred compensation plan",
    "h—elective deferrals to a section 501(c)(18)(d) tax-exempt  organization ",
    "plan. see the form 1040 instructions for how to deduct.",
    #"j—nontaxable sick pay (information only, not included in box 1, 3, or 5)",
    #"k—20% excise tax on excess golden parachute payments. see the ",
    #"form 1040 instructions.",
    #"l—substantiated employee business expense reimbursements ",
    #"(nontaxable)",
    "m—uncollected social security or rrta tax on taxable cost  of group-",
    "term life insurance over $50,000 (former employees only). see the form ",
    #"1040 instructions.",
    "n—uncollected medicare tax on taxable cost of group-term  life ",
    "insurance over $50,000 (former employees only). see the form 1040 ",
    #"instructions.",
    #"p—excludable moving expense reimbursements paid directly to a ",
    "member of the u.s. armed forces (not included in box 1, 3, or 5)",
    #"q—nontaxable combat pay. see the form 1040 instructions for details ",
    "on reporting this amount.",
    # 1099-INT instructions
    "box 1. shows taxable interest",
    "box 2. shows interest or principal forfeited",
    "box 3. shows interest on u.s. savings bonds",
    "box 4. shows backup withholding",
    "box 5. any amount shown is your share",
    "box 6. shows foreign tax paid",
    "box 7. shows the country or u.s. territory",
    "box 8. shows tax-exempt interest",
    "box 9. shows tax-exempt interest subject",
    "box 10. for a taxable or tax-exempt covered security",
    "box 11. for a taxable covered security",
    "box 12. for a u.s. treasury obligation",
    "box 13. for a tax-exempt covered security",
    "box 14. shows cusip number",
    "boxes 15-17. state tax withheld",
    # 1098-T instruction lines
    "you, or the person who can claim you as a dependent, may be able to claim an education credit",
    "student’s taxpayer identification number (tin)",
    "box 1. shows the total payments received by an eligible educational institution",
    "box 2. reserved for future use",
    "box 3. reserved for future use",
    "box 4. shows any adjustment made by an eligible educational institution",
    "box 5. shows the total of all scholarships or grants",
    "tip: you may be able to increase the combined value of an education credit",
    "box 6. shows adjustments to scholarships or grants for a prior year",
    "box 7. shows whether the amount in box 1 includes amounts",
    "box 8. shows whether you are considered to be carrying at least one-half",
    "box 9. shows whether you are considered to be enrolled in a program leading",
    "box 10. shows the total amount of reimbursements or refunds",
    "future developments. for the latest information about developments related to form 1098-t",
    # 1098-Mortgage
    ]
    for pat in instruction_patterns:
        if pat in lower:
            return "Others", "Unused"
       
       
        #---------------------------1099-DIV----------------------------------#
    #1099-INT for page 1
    div_front = [
        "form 1099-div",
        #"dividends and distributions",
        "1a total ordinary dividends",
        "1b qualified dividends distributions",
        "2a Total capital gain distr",
        "specified private activity bond interest dividends",
        "qualified dividends",
        "total capital gain distr",
        "section 1202 gain",
        "section 1250 gain",
    ]

    div_unused = [
       
        "the information contained herein",
        "please note that we have changed",
        "your redeemed shares has not been",
        "we are requested by trh irs",
        ]
    lower = text.lower()
    found_div_front = any(pat.lower() in lower for pat in div_front)
    found_div_unused = any(pat.lower() in lower for pat in div_unused)

# 🔁 Priority: 1099-INT > Unused
    if found_div_front:
        return "Income", "1099-DIV"
    elif found_div_unused:
        return "Others", "Unused"
           
    # --- 1099-MISC ---
    misc_category = [
        "form 1099-misc",
        "miscellaneous information",
        "1.rents",
        "2.royalties",
        "3.other income",
        "8.substitute payments in lieu of dividends or interest"
    ]
    for pat in misc_category:
        if pat in lower:
            return "Income", "1099-MISC"

    # --- 1099-OID ---
    oid_category = [
        "form 1099-oid",
        "original issue discount",
        "1.original issue discount",
        "2.other periodic interest",
        "5.market discount",
        "6.acquisition premium",
        "8.oid on u.s. treasury obligations",
        "10.bond premium",
        "11.tax-exempt oid"
    ]
    for pat in oid_category:
        if pat in lower:
            return "Income", "1099-OID"

    # --- 1099-B ---
    b_category = [
        "form 1099-b",
        "proceeds from broker and barter exchange transactions",
        "1d.proceeds",
        "covered securities",
        "noncovered securities",
        "1e.cost or other basis of covered securities",
        "1f.accrued market discount",
        "1g.wash sale loss disallowed"
    ]
    for pat in b_category:
        if pat in lower:
            return "Income", "1099-B"

    #---------------------------Consolidated-1099----------------------------------#
   
     # E*TRADE text in parts
   


    con_unused = [
        "etrade from morgan stanley 1099 consolidated tax statement for 2023 provides your official tax information",
        "income information that was reported on your december account statement will not have included certain adjustments",
        "if your etrade account was transferred to morgan stanley smith barney llc in 2023 you may receive a separate 1099 consolidated tax statement",
        "consider and review both consolidated tax statements when preparing your 2023 income tax return",
        "for more information on what to expect, visit etrade.com/taxyear2023",
        "the following tax documents are not included in this statement and are sent individually",
        "forms 1099-q, 1042-s, 2439, 5498, 5498-esa, remic information statement, schedule k-1 and puerto rico forms 480.6a, 480.6b, 480.6c and 480.6d"
    ]
   
    for pat in con_unused:
        if pat in lower:
            return "Others", "Unused"  
    
    #---------------------------Consolidated-1099----------------------------------#

    #---------------------------1099-INT----------------------------------#
 #---------------------------1099-INT----------------------------------#
    #1099-INT for page 1
    lower = re.sub(r"\s+", " ", text.lower())
   
    int_front = [
        "form 1099-int",
        "interest income",
        "copy b",
        "early withdrawal penalty",
        "tax-exempt interest",
        "bond premium on treasury obligations",
        "bond premium on tax-exempt bond",
        "specified private activity bond",
    ]
    # Also allow number-prefixed matches (e.g., "8 tax-exempt interest")
    int_front_regex = [
        r"\d+\s*tax-exempt interest",
        r"\d+\s*specified private activity bond",
        r"\d+\s*bond premium on treasury obligations",
        r"\d+\s*bond premium on tax-exempt bond",
    ]
    found_int_front = any(pat in lower for pat in int_front) or any(re.search(p, lower) for p in int_front_regex)
   
    int_unused = [
    # Box descriptions from instructions section
        "box 1. shows taxable interest",
        "box 2. shows interest or principal forfeited",
        "box 3. shows interest on u.s. savings bonds",
        "box 4. shows backup withholding",
        "box 5. any amount shown is your share of investment expenses",
        "box 6. shows foreign tax paid",
        "box 7. shows the country or u.s. possession",
        "box 8. shows tax-exempt interest",
        "box 9. shows tax-exempt interest subject to the alternative minimum tax",
        "box 10. for a taxable or tax-exempt covered security",
        "box 11. for a taxable covered security",
        "box 12. for a u.s. treasury obligation that is a covered security",
        "box 13. for a tax-exempt covered security",
        "box 14. shows cusip number",
        "boxes 15-17. state tax withheld",
   
    # Common generic phrases in the instruction block
        "instructions for recipient",
        "for more information, see form 8912",
        "see the instructions above for a taxable covered security",
        "see pub. 550",
        "report the accrued market discount",
        "see regulations section 1.171",
        "future developments",
        "free file program",
        "nominees. if this form includes amounts belonging",
        "the promotional bonus you",
        "discover bank takes from the interest paid",
        "itemized list of interest paid",
        "once your form is available online to view",
        ""
    ]

    found_int_unused = any(pat in lower for pat in int_unused)
   
    # ✅ Priority: front wins
    if found_int_front:
        return "Income", "1099-INT"
    elif found_int_unused:
        return "Others", "Unused"
    #---------------------------1099-SA----------------------------------#
    #1099-INT for page 1

   
    #---------------------------1098-Mortgage----------------------------------#    
    #1098-Mortgage form page 1
    mort_front = [
    "Mortgage insurance premiums",
    "Mortgage origination date",
    "Number of properties securing the morgage",  # typo here, maybe fix to "mortgage"
    "Address or description of property securing",
    "form 1098 mortgage",
    "limits based on the loan amount",
    "refund of overpaid",
    "Mortgage insurance important tax Information",
    "mortgage origination date the information",
    "1 mortgage interest received from",
    #"Account number (see instructions)"
    ]
    mort_unused = [
        "instructions for payer/borrower",
        "payer’s/borrower’s taxpayer identification number",
        "box 1. shows the mortgage interest received",
        "Box 1. Shows the mortgage interest received by the recipient",
        "Box 3. Shows the date of the mortgage origination",
        "Box 5. If an amount is reported in this box",
        "Box 8. Shows the address or description",  # ← this line was missing a comma
        "This information is being provided to you as",
        "We’re providing the mortgage insurance",
        "If you received this statement as the payer of",
        "If your mortgage payments were subsidized"
       
    ]
    lower = text.lower()
    found_front = any(pat.lower() in lower for pat in mort_front)
    found_unused = any(pat.lower() in lower for pat in mort_unused)

# 🔁 Priority: 1098-Mortgage > Unused
    if found_front:
        return "Expenses", "1098-Mortgage"
    elif found_unused:
        return "Others", "Unused"

    #---------------------------1098-Mortgage----------------------------------#
#3) fallback form detectors
    if 'w-2' in t or 'w2' in t: return 'Income', 'W-2'
    #if '1099-int' in t or 'interest income' in t: return 'Income', '1099-INT'
    #if '1099-div' in t: return 'Income', '1099-DIV'
    #if 'form 1099-div' in t: return 'Income', '1099-DIV'
   
    #if '1099' in t: return 'Income', '1099-Other'
    front_donation = [
        "donation",
        "volunteers greatly appreciate your",
        "Volunteers greatly appreciate your generous coma"
    ]
   
    for pat in front_donation:
        if pat in lower:
            return "Expenses", "Donation"  
    
    return 'Unknown', 'Unused'

   

   
# Detect W-2 pages by their header phrases
    if 'wage and tax statement' in t or ("employer's name" in t and 'address' in t):
        return 'Income', 'W-2'
