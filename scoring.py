import pandas as pd

def apply_reverse_score(row):
    score = row["numeric_score"]
    reverse_flag = row.get("reverse_score", False)
    scale_name = row.get("scale_name", "")

    if pd.isna(score):
        return score

    if not reverse_flag:
        return score

    if scale_name == "MFQ":
        return 6 - score
    elif scale_name == "RWA":
        return 8 - score
    elif scale_name == "LWA":
        return 8 - score
    # For NFC, the scale is -4 to 4. A reversed item would be -1 * score.
    # However, the current notebook doesn't seem to use reverse scoring for NFC in this way.
    # If NFC items are reversed, the logic might need to be: -1 * score to flip around 0.
    # Or, if it's about mapping to a positive scale first, then reversing, it's more complex.
    # For now, assuming NFC reverse scoring isn't applied in the sum, or handled differently.
    # If NFC questions *are* reversed and it means flipping sign (e.g. 3 becomes -3), use: `return -score`
    # If it implies reversing on a 0-8 scale first (after shifting -4 to 4 to 0 to 8), then it's different.
    # Sticking to how RWA/LWA/MFQ are handled as per notebook for now.
    else:
        return score # Default for other scales or if logic is not defined 