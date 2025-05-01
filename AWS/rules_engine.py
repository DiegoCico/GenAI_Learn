import re

def extract_revenue(results, quarter, year):
    """
    Applies rules to extract revenue numbers for a specific quarter and year.
    """
    print(f"[RULES] Looking for revenue in {quarter} {year}...\n")

    for result in results:
        text = result["text"]
        confidence = result["confidence"]

        print(f"→ Checking: {text}")
        
        # Rule: Look for exact quarter and year
        if quarter in text and year in text:
            print("✅ Quarter and year matched.")
            
            # Rule: Look for $ followed by number
            match = re.search(r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?", text)
            if match:
                value = match.group()
                print(f"✅ Revenue extracted: {value}")
                return {
                    "revenue": value,
                    "source": text,
                    "confidence": confidence
                }
            else:
                print("⚠️ No dollar amount found.")
        else:
            print("❌ Quarter/year mismatch.")
        print()

    print("[RULES] No matching revenue found.\n")
    return None
