import re
from typing import List, Dict

# ----------------------------
# 🔹 Mocked "Kendra" Output
# ----------------------------
# Simulates what Amazon Kendra would return after processing the query
def mocked_kendra_search(query: str) -> List[Dict]:
    """
    Simulates Amazon Kendra's response for a financial query.
    Each result includes a 'text' snippet and a 'confidence' score.
    """
    print(f"[KENDRA] Query received: '{query}'")
    return [
        {
            "text": "The total revenue for Q1 2025 was $180,000 according to the finance report.",
            "confidence": 0.92
        },
        {
            "text": "Revenue in Q1 last year hit $150,000, while this year saw an increase.",
            "confidence": 0.85
        },
        {
            "text": "Our Q2 projections exceed $200,000, but Q1 finished at $180,000.",
            "confidence": 0.83
        }
    ]


# ----------------------------
# 🔹 Rule-Based Engine
# ----------------------------
def extract_revenue(results: List[Dict], quarter: str, year: str) -> Dict:
    """
    Processes Kendra-like results and extracts the revenue for the specified quarter/year.

    Args:
        results: List of Kendra search result dicts with 'text' and 'confidence'.
        quarter: Quarter to look for (e.g., "Q1").
        year: Year to look for (e.g., "2025").

    Returns:
        A dictionary with the answer and metadata.
    """
    print(f"[RULE ENGINE] Extracting revenue for {quarter} {year}...\n")

    for result in results:
        text = result["text"]
        confidence = result["confidence"]

        # Print the snippet and confidence
        print(f"[MATCH] Text: {text}")
        print(f"         Confidence: {confidence}")

        # Simple rule: Match Q1 2025 and look for a dollar amount
        if quarter in text and year in text:
            # Extract the first $XXX,XXX pattern
            match = re.search(r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?", text)
            if match:
                amount = match.group()
                print(f"[RULE] Found revenue match: {amount}")
                return {
                    "answer": f"The total revenue for {quarter} {year} is {amount}.",
                    "source_snippet": text,
                    "confidence": confidence
                }

    # If no match is found
    return {
        "answer": f"Sorry, no clear revenue information was found for {quarter} {year}.",
        "source_snippet": None,
        "confidence": 0.0
    }


# ----------------------------
# 🔹 Main Function
# ----------------------------
def main():
    # User query
    user_question = "What was the revenue for Q1 2025?"

    # Step 1: Simulate Kendra search
    kendra_results = mocked_kendra_search(user_question)

    # Step 2: Use rule engine to find the correct number
    final_answer = extract_revenue(kendra_results, quarter="Q1", year="2025")

    # Step 3: Display the final result
    print("\n✅ FINAL ANSWER:")
    print(final_answer["answer"])
    if final_answer["source_snippet"]:
        print(f"🔎 From: \"{final_answer['source_snippet']}\" (Confidence: {final_answer['confidence']})")


# ----------------------------
# 🔹 Run the Simulation
# ----------------------------
if __name__ == "__main__":
    main()
