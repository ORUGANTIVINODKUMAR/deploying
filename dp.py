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

