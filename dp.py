
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

# ── Merge + bookmarks + cleanup
def merge_with_bookmarks(input_dir: str, output_pdf: str, tp_ssn: str = "", sp_ssn: str = ""):
    # Prevent storing merged file inside input_dir
    abs_input = os.path.abspath(input_dir)
    abs_output = os.path.abspath(output_pdf)
    if abs_output.startswith(abs_input + os.sep):
        abs_output = os.path.join(os.path.dirname(abs_input), os.path.basename(abs_output))
        logger.warning(f"Moved output outside: {abs_output}")
    # ✅ Collect all candidate files
    all_files = sorted(
        f for f in os.listdir(abs_input)
        if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff'))
        and f != os.path.basename(abs_output)
    )
    import hashlib

    # --- Detect duplicate PDFs by MD5 hash ---
    hash_map = {}         # md5 -> first filename
    duplicate_files = []  # list of duplicate filenames

    for f in all_files:
        path = os.path.join(abs_input, f)
        try:
            with open(path, "rb") as fh:
                md5 = hashlib.md5(fh.read()).hexdigest()
            if md5 in hash_map:
                duplicate_files.append(f)
            else:
                hash_map[md5] = f
        except Exception as e:
            print(f"⚠️ Could not hash {f}: {e}", file=sys.stderr)

    # Keep only unique files for processing
    files = sorted(hash_map.values())
    logger.info(f"Found {len(files)} unique files, {len(duplicate_files)} duplicates.")

   
   # remove any zero‐byte files so PdfReader never sees them
    files = []
    for f in all_files:
        p = os.path.join(abs_input, f)
        if os.path.getsize(p) == 0:
           logger.warning(f"Skipping empty file: {f}")
           continue
        files.append(f)
    # 🔄 Convert images into PDFs so the rest of the pipeline sees only PDFs
    converted_files = []
    for f in list(files):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            path = os.path.join(abs_input, f)
            try:
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                pdf_path = os.path.splitext(path)[0] + "_conv.pdf"
                img.save(pdf_path, "PDF", resolution=300.0)
                print(f"🖼️ Converted {f} → {os.path.basename(pdf_path)}", file=sys.stderr)

                # replace the image with its new PDF
                files.remove(f)
                files.append(os.path.basename(pdf_path))
                converted_files.append(pdf_path)
            except Exception as e:
                print(f"❌ Failed to convert {f}: {e}", file=sys.stderr)

    logger.info(f"Found {len(files)} files in {abs_input}")

    income, expenses, others = [], [], []
    # what bookmarks we want in workpapaer shoudl be add in this
    w2_titles = {}
    int_titles = {}
    div_titles = {} # <-- Add this line
    sa_titles = {}  
    mort_titles = {}
    sa5498_titles = {}
    t529_titles = {}
    t1098_titles = {}
    account_pages = {}  # {account_number: [(path, page_index, 'Consolidated-1099')]}
    account_names = {}
    k1_pages = {}        # {ein: [(path, page_index, 'K-1')]}
    k1_names = {}        # {ein: 'Partnership name'}
    k1_form_type = {}    # {ein: '1065' or '1120S' or '1041'}


    # ✅ Track seen page text hashes to detect duplicate pages (within or across files)
    seen_hashes = {}   # tracks duplicate page text
    seen_pages = {}    # tracks appended pages


    # --- Skip duplicates in main processing ---
    files = [f for f in files if f not in duplicate_files]

    for fname in files:
        last_ein_seen = None
        seen_pages = {}   # reset duplicate tracker per file
        path = os.path.join(abs_input, fname)
        if fname.lower().endswith('.pdf'):
            total = len(PdfReader(path).pages)
            for i in range(total):
                print("=" * 400, file=sys.stderr)
                print(f"Processing: {fname}, Page {i+1}", file=sys.stderr)

                # ── Print header before basic extract_text
                print("→ extract_text() output:", file=sys.stderr)
                try:
                    text = extract_text(path, i)
                    # --- 🆕 Detect duplicate pages across all PDFs ---
                    import hashlib

                    page_hash = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
                    file_key = (fname, page_hash)   # <-- isolate by filename
                    if file_key in seen_hashes:
                        print(f"[DUPLICATE PAGE] {fname} p{i+1} matches ...", file=sys.stderr)
                        others.append((path, i, "Duplicate"))
                        continue  # ⬅️ SKIPS classification and appending
                    else:
                        seen_hashes[file_key] = (path, i)


                    print(text or "[NO TEXT]", file=sys.stderr)
                except Exception as e:
                    print(f"[ERROR] extract_text failed: {e}", file=sys.stderr)

                print("=" * 400, file=sys.stderr)

                # Multi-method extraction
                extracts = {}

                print("→ PDFMiner:", file=sys.stderr)
                try:
                    extracts['PDFMiner'] = pdfminer_extract(path, page_numbers=[i], laparams=PDFMINER_LA_PARAMS) or ""
                    print(extracts['PDFMiner'], file=sys.stderr)
                except Exception as e:
                    extracts['PDFMiner'] = ""
                    print(f"[ERROR] PDFMiner failed: {e}", file=sys.stderr)

               

                print("→ Tesseract OCR:", file=sys.stderr)
                try:
                    img = pdf_page_to_image(path, i, dpi=150)  # ✅ use your PyMuPDF helper
                    extracts['Tesseract'] = pytesseract.image_to_string(img, config="--psm 6") or ""
                    print(extracts['Tesseract'], file=sys.stderr)
                except Exception as e:
                    extracts['Tesseract'] = ""
                    print(f"[ERROR] Tesseract failed: {e}", file=sys.stderr)

                print("→ pdfplumber:", file=sys.stderr)
                try:
                    with pdfplumber.open(path) as pdf:
                        extracts['pdfplumber'] = pdf.pages[i].extract_text() or ""
                        print(extracts['pdfplumber'], file=sys.stderr)
                except Exception as e:
                    extracts['pdfplumber'] = ""
                    print(f"[ERROR] pdfplumber failed: {e}", file=sys.stderr)

                print("→ PyMuPDF (fitz):", file=sys.stderr)
                try:
                    doc = fitz.open(path)
                    extracts['PyMuPDF'] = doc.load_page(i).get_text()
                    doc.close()
                    print(extracts['PyMuPDF'], file=sys.stderr)
                except Exception as e:
                    extracts['PyMuPDF'] = ""
                    print(f"[ERROR] PyMuPDF failed: {e}", file=sys.stderr)

                print("=" * 400, file=sys.stderr)
             

                # Collect W-2 employer names across methods
                info_by_method, names = {}, []
                for method, txt in extracts.items():
                    cat, ft = classify_text(txt)
                    if cat == 'Income' and ft == 'W-2':
                        info = parse_w2(txt)
                        if info['employer_name'] != 'N/A':
                            info_by_method[method] = info
                            names.append(info['employer_name'])
                    # --- 1099-INT bookmark extraction ---
                    if cat == 'Income' and ft == '1099-INT':
                        title = extract_1099int_bookmark(txt)
                        if title and title != '1099-INT':
                            int_titles[(path, i)] = title
                    # <<< new DIV logic
                    if cat == 'Income' and ft == '1099-DIV':
                        title = extract_1099div_bookmark(txt)
                        if title and title != '1099-DIV':
                            div_titles[(path, i)] = title
                    if cat == 'Income' and ft == '1099-SA':
                        title = extract_1099sa_bookmark(txt)
                        if title and title != '1099-SA':
                            sa_titles[(path, i)] = title

                    if cat == 'Expenses' and ft == '1098-Mortgage':
                        title = extract_1098mortgage_bookmark(txt)
                        if title and title != '1098-Mortgage':
                            mort_titles[(path, i)] = title
                    if cat == 'Expenses' and ft == '5498-SA':
                        title = extract_5498sa_bookmark(txt)
                        if title and title != '5498-SA':
                            sa5498_titles[(path, i)] = title
                           
                    if cat == 'Expenses' and ft == '1098-T':
                        title = extract_1098t_bookmark(txt)
                        if title and title != '1098-T':
                            t1098_titles[(path, i)] = title
                    if cat == 'Expenses' and ft == '529-Plan':
                        title = extract_529_bookmark(txt)
                        if title and title != '529-Plan':
                            t529_titles[(path, i)] = title

                if names:
                    common = Counter(names).most_common(1)[0][0]
                    chosen = next(m for m,i in info_by_method.items() if i['employer_name'] == common)
                    print(f"--- Chosen employer ({chosen}): {common} ---", file=sys.stderr)
                    print_w2_summary(info_by_method[chosen])
                    w2_titles[(path, i)] = common

                # Classification & grouping
                    # … after you’ve extracted text …
                   # NEW: {acct: "Issuer Name"}

                tiered = extract_text(path, i)
                
                acct_num = extract_account_number(tiered)

                # --- Detect Schedule K-1 / Form 1065 / Statement A / QBI pages ---
                # --- Detect Schedule K-1 / Form 1065 / and related pages (QBI, Statement A, etc.) ---
                text_lower = tiered.lower()
                ein_num = extract_ein_number(tiered)

                # --- Detect Schedule K-1 / Form 1065 / Statement A / QBI / worksheet pages ---
                text_lower = tiered.lower()
                ein_num = extract_ein_number(tiered)

                # ✅ If EIN missing (common on continuation or basis pages), reuse the previous one
                # Track EIN across pages
                if not ein_num:
                    ein_num = last_ein_seen
                else:
                    last_ein_seen = ein_num


                # ✅ Detect any K-1-related page (expanded keyword list)
                k1_keywords = [
                    "schedule k-1", "form 1065", "form 1120-s", "form 1041",
                    "statement a", "qualified business income", "section 199a",
                    "qbi pass-through", "qbi pass through entity reporting",
                    # new worksheet / continuation phrases
                    "basis worksheet", "partner's basis worksheet",
                    "allocation of losses and deductions",
                    "partner's share of income", "at-risk basis", "section 704(c)",
                    "k-1 supplemental information", "continuation statement",
                    "additional information from schedule k-1", "passive activity purposes",
                    
                ]
                is_k1_page = any(k in text_lower for k in k1_keywords)

                # NEW — Detect TRUE form type
                is_1120s = bool(re.search(r"schedule\s*k-1.*form\s*1120[-\s]*s", text_lower))
                is_1065 = bool(re.search(r"schedule\s*k-1.*form\s*1065", text_lower))
                
                is_1041 = bool(re.search(r"schedule\s*k-1.*form\s*1041", text_lower))

                if (is_1065 or is_1120s or is_1041 or is_k1_page) and ein_num:

                # Assign correct form type
                    k1_form_type[ein_num] = (
                        "1065" if is_1065 else
                        "1120S" if is_1120s else
                        "1041" if is_1041 else
                        k1_form_type.get(ein_num, "1065")
                    )

                    # Extract company / partnership name
                    company = extract_k1_company(tiered)



                    k1_pages.setdefault(ein_num, []).append((path, i, "K-1"))
                    if company:
                        company = clean_k1_company_name(company)
                        k1_names[ein_num] = company





                    print(
                        f"[DEBUG] Unified K-1 page: {os.path.basename(path)} p{i+1} "
                        f"EIN={ein_num}, Company={company}, Form={k1_form_type[ein_num]}",
                        file=sys.stderr
                    )


                lower_text = tiered.lower()

                # --- Only add to Consolidated-1099 if it's truly a 1099 form ---
                if acct_num:
                    # Check if it's a 1099 (INT, DIV, B, MISC, etc.)
                    if "1099" in lower_text and not re.search(r"1098[-\s]*t", lower_text, re.IGNORECASE):
                        account_pages.setdefault(acct_num, []).append((path, i, "Consolidated-1099"))

        # Capture issuer name if present
                        issuer = extract_consolidated_issuer(tiered)
                        if issuer:
                            account_names.setdefault(acct_num, issuer)

                        print(f"[DEBUG] {os.path.basename(path)} p{i+1}: Added to Consolidated-1099 (acct={acct_num})", file=sys.stderr)
                    else:
                        # Skip non-1099 forms (e.g., 1098-T, 1098-Mortgage, etc.)
                        print(f"[DEBUG] {os.path.basename(path)} p{i+1}: Skipped Consolidated-1099 (acct={acct_num}, form=non-1099)", file=sys.stderr)

# Always classify after account checks
                cat, ft = classify_text(tiered)

               
                # NEW: log every classification
                print(
                    f"[Classification] {os.path.basename(path)} p{i+1} → "
                    f"Category='{cat}', Form='{ft}', "
                    f"snippet='{tiered[:150].strip().replace(chr(80),' ')}…'",
                    file=sys.stderr
                )

                entry = (path, i, ft)
                if cat == 'Income':
                    income.append(entry)
                elif cat == 'Expenses':
                    expenses.append(entry)
                else:
                    others.append(entry)

        else:
            # Image handling
            print(f"\n=== Image {fname} ===", file=sys.stderr)
            oi = extract_text_from_image(path)
            print("--- OCR Image ---", file=sys.stderr)
            print(oi, file=sys.stderr)
            cat, ft = classify_text(oi)
            entry = (path, 0, ft)
            if cat == 'Income':
                income.append(entry)
            elif cat == 'Expenses':
                expenses.append(entry)
            else:
                others.append(entry)

        # ---- K-1 (Form 1065) grouping synthesis ----
    k1_payload = {}      # key -> list of (path, page_index, 'K-1')
    k1_pages_seen = set()  # to mark pages already grouped
    # --- 🩹 Move all raw K-1 entries into k1_pages before grouping ---
    for (path, idx, form) in list(income):
        if form == "K-1":
            text = extract_text(path, idx).lower()
            ein = extract_ein_number(text)
            k1_type = k1_form_type.get(ein, "1065")  

            if not ein and k1_pages:
                ein = list(k1_pages.keys())[-1]   # reuse last EIN if missing
            if ein:
                k1_pages.setdefault(ein, []).append((path, idx, "K-1"))
                # remove from income so it doesn't create a stray K-1 later
                income.remove((path, idx, form))

    for ein, pages in k1_pages.items():
        if pages:
            key = f"K1::{k1_type}::{ein}"
            # store real pages
            k1_payload[key] = [(p, i, "K-1") for (p, i, _) in pages]
            # remember them so we don’t process twice
            for (p, i, _) in pages:
                k1_pages_seen.add((p, i))
            # add single synthetic entry
            income.append((key, -1, "K-1"))
           # remove synthetic K-1 entries before sorting
           #income = [e for e in income if not str(e[0]).startswith("K1::")]

            print(f"[DEBUG] Added unified K-1 package for EIN={ein} with {len(pages)} page(s)", file=sys.stderr)

    # 🧹 remove any individual K-1 pages that were grouped
    income = [
        e for e in income
        if not (e[2] == "K-1" and not e[0].startswith("K1::") and (e[0], e[1]) in k1_pages_seen)
    ]
    print(f"[DEBUG] Cleaned raw K-1 pages; synthetic K1 groups = {[x[0] for x in income if 'K1::' in x[0]]}", file=sys.stderr)


            

    # ---- Consolidated-1099 synthesis (insert this BEFORE income.sort(...)) ----
    consolidated_payload = {}        # key -> list of real page entries
    consolidated_pages = set()       # pages already placed under Consolidated-1099
    # Track pages we already decided are "Unused" so we don't touch them again
    unused_pages: set[tuple[str, int]] = set()


    for acct, pages in account_pages.items():
        if len(pages) <= 1:
            continue  # only group repeated accounts
        key = f"CONSOLIDATED::{acct}"
        consolidated_payload[key] = [(p, i, "Consolidated-1099") for (p, i, _) in pages]
        for (p, i, _) in pages:
            consolidated_pages.add((p, i))
    # add a synthetic income row that will sort using priority of 'Consolidated-1099'
        income.append((key, -1, "Consolidated-1099"))
# --------------------------------------------------------------------------
    # REMOVE synthetic K-1 entries ONLY FOR SORTING
    clean_income = [e for e in income if not str(e[0]).startswith("K1::")]

    # Now sort only REAL pages
    clean_income.sort(key=lambda e:(get_form_priority(e[2],'Income'), e[0], e[1]))

        # Restore synthetic K-1 groups after sorting
    # (They must be present so K-1 bookmark block runs)
    clean_income += [e for e in income if str(e[0]).startswith("K1::")]

    income = clean_income

    # Sort
    income.sort(key=lambda e:(get_form_priority(e[2],'Income'), e[0], e[1]))
    expenses.sort(key=lambda e:(get_form_priority(e[2],'Expenses'), e[0], e[1]))
    # merge & bookmarks
    merger = PdfMerger()
    page_num = 0
    stop_after_na = False
    import mimetypes
    #seen_pages = set()
    #def append_and_bookmark(entry, parent, title, with_bookmark=True):
    
    def append_and_bookmark(entry, parent, title, with_bookmark=True):
        nonlocal page_num, seen_pages
        global k1_page_map

        p, idx, _ = entry

        # Synthetic K-1 key expansion -------------------------
        if p.startswith("K1::"):
            key = p
            real_entries = k1_payload.get(key, [])
            if not real_entries:
                return

            for i, real_entry in enumerate(real_entries):
                append_and_bookmark(
                    real_entry,
                    parent,
                    title if i == 0 and with_bookmark else "",
                    with_bookmark=(i == 0 and with_bookmark),
                )
            return

        # Skip duplicates -------------------------------------
        sig = (p, idx)
        if sig in seen_pages:
            return
        seen_pages[sig] = True

        # Write single page temp PDF --------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            w = PdfWriter()
            w.add_page(PdfReader(p).pages[idx])
            w.write(tmp)
            tmp.flush()
            tmp_path = tmp.name

        with open(tmp_path, 'rb') as fh:
            merger.append(fileobj=fh)
        os.unlink(tmp_path)

        # ⭐ Record the merged PDF page number for K-1 mapping
        # Record merged position for every REAL page (ignore synthetic)
        if not p.startswith("K1::"):
            k1_page_map[(p, idx)] = page_num


        # Add bookmark
        if with_bookmark and title:
            merger.add_outline_item(title, page_num, parent=parent)

        page_num += 1

    # --- Unified K-1 bookmark builder ----------------------------------------
    def build_k1_bookmarks(merger, root, k1_pages, k1_names,
                       extract_text, append_and_bookmark, is_unused_page):
        """
        CORRECT K-1 BUILDER
    Ensures:
    - Pages for each EIN are SORTED BEFORE merging
    - Form 1065 / 1120-S / 1041 folders appear in correct order
    - No pages are appended before sorting
    """
        global k1_page_map

        # 1. Create top-level K-1 folder
        k1_root = merger.add_outline_item("K-1", 0, parent=root)

    # 2. Determine first-page anchor for each EIN (based on file order)
        ein_first_order = {
            ein: min(idx for (_, idx, _) in pages)
            for ein, pages in k1_pages.items()
        }

        # 3. Detect form type order (1065,1120-S,1041)
        form_first_page = {}
        for ein, pages in k1_pages.items():
            form = k1_form_type.get(ein, "1065")
            first_idx = min(idx for (_, idx, _) in pages)
            if form not in form_first_page or first_idx < form_first_page[form]:
                form_first_page[form] = first_idx

        # 4. Sort form folders in true order
        sorted_forms = sorted(form_first_page, key=lambda f: form_first_page[f])

        # 5. Create folders
        form_roots = {}
        for form in sorted_forms:
            # find the first merged page for this form
            ein_for_form = [
                ein for ein in k1_pages
                if k1_form_type.get(ein, "1065") == form
            ]

            if ein_for_form:
                first_ein = ein_for_form[0]
                first_page = min(
                    k1_page_map.get((p, idx), 999999)
                    for (p, idx, _) in k1_pages[first_ein]
                )
            else:
                first_page = 0  # fallback

            form_roots[form] = merger.add_outline_item(
                f"Form {form}",
                first_page,
                parent=k1_root
            )


        # 6. NOW handle each EIN group
        for ein in sorted(k1_pages, key=lambda e: ein_first_order[e]):
            pages = k1_pages[ein]
            company = k1_names.get(ein, f"EIN {ein}")
            form_type = k1_form_type.get(ein, "1065")

            # ---- SORT PAGES FIRST (THE MOST IMPORTANT PART) ----
            sorted_pages = sorted(
                pages,
                key=lambda x: k1_page_priority(extract_text(x[0], x[1]))
            )

        # Find anchor = main K-1 form page
            anchor_path, anchor_idx = None, None
            for (p, idx, _) in sorted_pages:
                text = extract_text(p, idx).lower()
                if "schedule k-1" in text and ("form 1065" in text or "form 1120-s" in text or "form 1041" in text):
                    anchor_path, anchor_idx = p, idx
                    break
            # fallback
            if anchor_path is None:
                anchor_path, anchor_idx = sorted_pages[0][0], sorted_pages[0][1]


            # Create the EIN bookmark
            # convert original anchor index → merged page index
            anchor_page = k1_page_map.get((anchor_path, anchor_idx), 0)

            clean = extract_k1_company(company)
            comp_node = merger.add_outline_item(
                f"{clean} (EIN {ein})",
                anchor_page,
                parent=form_roots[form_type]
            )


            # ---- NOW append pages IN SORTED ORDER ----
            for (p, idx, _) in sorted_pages:
                txt = extract_text(p, idx)
                if is_unused_page(txt.lower()):
                    continue
                append_and_bookmark((p, idx, "K-1"), comp_node, "", with_bookmark=False)


    # ── Bookmarks
   
    if income:
        root = merger.add_outline_item('Income', page_num)
            # --- Custom hierarchical K-1 bookmark structure ---
        processed_pages = set()  # ✅ Track K-1/QBI pages already appended



        groups = group_by_type(income)
        for form, grp in sorted(groups.items(), key=lambda kv: get_form_priority(kv[0], 'Income')):
                # ✅ Run the unified K-1 block FIRST
            if form == 'K-1':

                # STEP 1 — Append ALL real K-1 pages FIRST (no bookmarks yet)
                for ein, pages in k1_pages.items():
                    sorted_pages = sorted(
                        pages,
                        key=lambda x: k1_page_priority(extract_text(x[0], x[1]))
                    )
                    for (p, idx, _) in sorted_pages:
                        append_and_bookmark((p, idx, "K-1"), None, "", with_bookmark=False)

                # STEP 2 — NOW build bookmarks (page numbers now exist)
                build_k1_bookmarks(
                    merger, root,
                    k1_pages, k1_names,
                    extract_text, append_and_bookmark, is_unused_page
                )

                continue


            if stop_after_na:
                break
            if form == 'Consolidated-1099':
                cons_root = merger.add_outline_item('Consolidated-1099', page_num, parent=root)

                for entry in filtered_grp:
                    key, _, _ = entry
                    acct = key.split("::", 1)[1]

                    issuer = account_names.get(acct)
                    issuer = alias_issuer(issuer) if issuer else None
                    forms_label = issuer or f"Account {acct}"
                    forms_node = merger.add_outline_item(forms_label, page_num, parent=cons_root)

                    real_entries = consolidated_payload.get(key, [])

        # (optional context labels — does NOT skip appends)
             

        # ALWAYS append the real pages
                    for real_entry in real_entries:
                        page_text = extract_text(real_entry[0], real_entry[1])
                        if is_unused_page(page_text):
                            print(f"[DROP?] {os.path.basename(real_entry[0])} page {real_entry[1]+1} "
                                    f"marked as UNUSED", file=sys.stderr)
                            others.append((real_entry[0], real_entry[1], "Unused"))
                            #append_and_bookmark(real_entry, forms_node, "Unused")
                            continue

    # 1️⃣ First, check strong classifier
                        form_type = classify_div_int(page_text)

                        if form_type == "1099-DIV":
                            append_and_bookmark(real_entry, forms_node, "1099-DIV Description")

        # 2️⃣ Also check for other forms on same page
                            extra_forms = [ft for ft in (classify_text_multi(page_text) or [])
                                           if ft != "1099-DIV"]
                            for ft in extra_forms:
                                merger.add_outline_item(ft, page_num - 1, parent=forms_node)

                        elif form_type == "1099-INT":
                            if has_nonzero_int(page_text):
                                append_and_bookmark(real_entry, forms_node, "1099-INT Description")
                            else:
        # Still append the page, but give it a neutral label
                                append_and_bookmark(real_entry, forms_node, "1099-INT (all zero)")
                                print(f"[NOTE] {os.path.basename(real_entry[0])} page {real_entry[1]+1} "
                                  f"→ 1099-INT detected but all zero; kept page with neutral bookmark", file=sys.stderr)

                        # 2️⃣ Also check for other forms on same page
                            extra_forms = [ft for ft in (classify_text_multi(page_text) or [])
                                           if ft != "1099-INT"]
                            for ft in extra_forms:
                                merger.add_outline_item(ft, page_num - 1, parent=forms_node)

                        else:
        # 3️⃣ Fallback: pure multi-form logic
                            form_matches = classify_text_multi(page_text)

                            title = None
                            extra_forms = []

                            if form_matches:
    # Special rule: drop 1099-INT if all zero
                                if "1099-INT" in form_matches and not has_nonzero_int(page_text):
                                    form_matches = [f for f in form_matches if f != "1099-INT"]

                                if form_matches:
                                    title = form_matches[0]
                                    extra_forms = form_matches[1:]

# Append once, with or without bookmark
                            if title:
                                append_and_bookmark(real_entry, forms_node, title)
                                for ft in extra_forms:
                                    merger.add_outline_item(ft, page_num - 1, parent=forms_node)
                            else:
    # Only zero INT → keep page, no bookmark
                                append_and_bookmark(real_entry, forms_node, "", with_bookmark=False)

                continue
            # 4️⃣ ── Normal single-form handling (W-2, 1099s, etc.) ─────────────
            filtered_grp = [e for e in grp if (e[0], e[1]) not in consolidated_pages]
            if not filtered_grp:
                continue

            #k1 bookmrk

  # done with this form; go to next
            #Normal Forms
            node = merger.add_outline_item(form, page_num, parent=root)
            for j, entry in enumerate(filtered_grp, 1):
                path, idx, _ = entry
                # ✅ Skip if page already processed in K-1 grouping
                if (path, idx) in processed_pages:
                    continue

                # 🚫 Skip if already appended under Consolidated-1099
                if (path, idx) in consolidated_pages:
                    continue

                # build the label
                lbl = form if len(grp) == 1 else f"{form}#{j}"
                if form == 'W-2':
                    emp = w2_titles.get((path, idx))
                    if emp:
                        lbl = emp
                elif form == '1099-INT':
                    payer = int_titles.get((path, idx))
                    if payer:
                        lbl = payer
                elif form == '1099-DIV':                  # <<< new
                    payer = div_titles.get((path, idx))
                    if payer:
                        lbl = payer
                elif form == '1099-SA':
                    payer = sa_titles.get((path, idx))
                    if payer:
                        lbl = payer

                # NEW: strip ", N.A" and stop after this bookmark
                if ", N.A" in lbl:
                    lbl = lbl.replace(", N.A", "")
                print(f"[Bookmark] {os.path.basename(path)} p{idx+1} → Category='Income', Form='{form}', Title='{lbl}'", file=sys.stderr)
                   
                # normal case
                print(f"[Bookmark] {os.path.basename(path)} p{idx+1} → Category='Income', Form='{form}', Title='{lbl}'", file=sys.stderr)
                page_text = extract_text(path, idx)
                owner = detect_ssn_owner(page_text, tp_ssn, sp_ssn)
                if owner:
                    lbl = f"{lbl} – {owner}"
 
                print(f"[SSN Tag] {os.path.basename(path)} p{idx+1} → {owner}", file=sys.stderr)
    
                append_and_bookmark(entry, node, lbl)
            
            if stop_after_na:
                break

    if expenses:
        root = merger.add_outline_item('Expenses', page_num)
        for form, grp in group_by_type(expenses).items():
            if stop_after_na:
                break
            node = merger.add_outline_item(form, page_num, parent=root)
            for j, entry in enumerate(grp, 1):
                path, idx, _ = entry
                lbl = form if len(grp) == 1 else f"{form}#{j}"
                if form == '1098-Mortgage':
                    m = mort_titles.get((path, idx))
                    if m:
                      lbl = m
                elif form == '5498-SA':
                    trustee = sa5498_titles.get((path, idx))
                    if trustee:
                        lbl = trustee
                    else:
                        lbl = extract_5498sa_bookmark(text)

                elif form == '1098-T':
                    trustee = t1098_titles.get((path, idx))
                    if trustee:
                        lbl = trustee
                    else:
                        page_text = extract_text(path, idx)  # ✅ get text for this page
                        lbl = extract_1098t_bookmark(page_text)
                elif form == "Child Care Expenses":
                    page_text = extract_text(path, idx)  # ✅ get the actual text for this page
                    provider = extract_daycare_bookmark(page_text)
                    lbl = provider if provider else "Child Care Provider"
                    append_and_bookmark(entry, node, lbl)
                elif form == '529-Plan':
                    title = t529_titles.get((path, idx))
                    if title:
                        lbl = title

               
                # NEW: strip ", N.A" and stop
                if ", N.A" in lbl:
                    lbl = lbl.replace(", N.A", "")
                print(f"[Bookmark] {os.path.basename(path)} p{idx+1} → Category='Expenses', Form='{form}', Title='{lbl}'", file=sys.stderr)
                   
                # normal case
                print(f"[Bookmark] {os.path.basename(path)} p{idx+1} → Category='Expenses', Form='{form}', Title='{lbl}'", file=sys.stderr)
                # 🆕 Detect SSN owner for this page and label it
                page_text = extract_text(path, idx)
                owner = detect_ssn_owner(page_text, tp_ssn, sp_ssn)
                if owner:
                    lbl = f"{lbl} – {owner}"
 
                print(f"[SSN Tag] {os.path.basename(path)} p{idx+1} → {owner}", file=sys.stderr)
    
                append_and_bookmark(entry, node, lbl)
            if stop_after_na:
                break

# --- Add Others section with Unused and Duplicate pages ---
    if (others and any(e[2] != "Unused" for e in others)) or duplicate_files:
        root = merger.add_outline_item('Others', page_num)

        # Unused pages
        unused_pages = [e for e in others if e[2] == 'Unused']
        if unused_pages:
            node_unused = merger.add_outline_item('Unused', page_num, parent=root)
            for entry in unused_pages:
                append_and_bookmark(entry, node_unused, "", with_bookmark=False)
                print(f"[Bookmark] {os.path.basename(entry[0])} p{entry[1]+1} → Category='Others', Form='Unused'", file=sys.stderr)

    # 🆕 Duplicate pages or duplicate files
        # 🆕 Duplicate pages or duplicate files
        dup_pages = [e for e in others if e[2] == 'Duplicate']
        if dup_pages or duplicate_files:
            node_dupe = merger.add_outline_item('Duplicate', page_num, parent=root)

            # Page-level duplicates (append without bookmarks)
            for entry in dup_pages:
                append_and_bookmark(entry, node_dupe, "", with_bookmark=False)

            # File-level duplicates (append without bookmarks)
            for f in duplicate_files:
                dup_path = os.path.join(abs_input, f)
                try:
                    reader = PdfReader(dup_path)
                    for i in range(len(reader.pages)):
                        append_and_bookmark((dup_path, i, "Duplicate"), node_dupe, "", with_bookmark=False)
                    print(f"[Duplicate] Added file {f} under 'Others → Duplicate'", file=sys.stderr)
                except Exception as e:
                    print(f"⚠️ Failed to append duplicate file {f}: {e}", file=sys.stderr)




            #append_and_bookmark(entry, node, lbl)

    input_count = sum(
    len(PdfReader(os.path.join(input_dir, f)).pages)
    for f in files if f.lower().endswith(".pdf")
    )
    print(f"[SUMMARY] Input pages={input_count}, Output pages={page_num}", file=sys.stderr)

    # Write merged output
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)
    with open(abs_output,'wb') as f:
        merger.write(f)
    merger.close()
    print(f"Merged PDF created at {abs_output}", file=sys.stderr)

    # Cleanup uploads
    # Cleanup uploads
    # Cleanup uploads (originals + converted PDFs)
    to_delete = set(files) | set(os.path.basename(f) for f in converted_files)

    for fname in list(to_delete):
        fpath = os.path.join(input_dir, fname)
        try:
            if os.path.exists(fpath):
                os.remove(fpath)
                print(f"🧹 Deleted {fname}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Failed to delete {fname}: {e}", file=sys.stderr)

    # Also remove any leftover images (JPG, PNG, etc.)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            try:
                os.remove(os.path.join(input_dir, fname))
                print(f"🧹 Deleted leftover image {fname}", file=sys.stderr)
            except Exception as e:
                print(f"⚠️ Failed to delete leftover image {fname}: {e}", file=sys.stderr)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Merge PDFs with robust text extraction and TP/SP labeling")
    p.add_argument('input_dir', help="Folder containing PDFs to merge")
    p.add_argument('output_pdf', help="Path for the merged PDF (outside input_dir)")
    p.add_argument('tp_ssn', nargs='?', default='', help="Taxpayer SSN last 4 digits")
    p.add_argument('sp_ssn', nargs='?', default='', help="Spouse SSN last 4 digits")
    args = p.parse_args()
    merge_with_bookmarks(args.input_dir, args.output_pdf, args.tp_ssn, args.sp_ssn)

