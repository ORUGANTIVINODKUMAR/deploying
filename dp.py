
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


            
