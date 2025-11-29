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

