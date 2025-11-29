
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
