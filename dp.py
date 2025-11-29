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


                
