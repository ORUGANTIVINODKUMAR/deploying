# REMOVE synthetic K-1 entries ONLY FOR SORTING
    clean_income = [e for e in income if not str(e[0]).startswith("K1::")]

    # Now sort only REAL pages
    clean_income.sort(key=lambda e:(get_form_priority(e[2],'Income'), e[0], e[1]))

        # Restore synthetic K-1 groups after sorting
    # (They must be present so K-1 bookmark block runs)
    clean_income += [e for e in income if str(e[0]).startswith("K1::")]

    income = clean_income

    
