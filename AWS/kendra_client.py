# Simulated Amazon Kendra query client

def query_kendra(question):
    """
    Pretend to query Amazon Kendra. Returns fake top 3 matches with text and confidence.
    """
    print(f"[KENDRA] Querying for: '{question}'\n")

    results = [
        {
            "text": "The revenue for Q1 2025 was $180,000 as reported in the financial statement.",
            "confidence": 0.91
        },
        {
            "text": "In Q1 2024, we reached $150,000 in revenue, a 20% drop from Q1 2025.",
            "confidence": 0.85
        },
        {
            "text": "Projected Q2 2025 revenue exceeds $200,000, but Q1 finalized at $180,000.",
            "confidence": 0.79
        }
    ]

    print("[KENDRA] Returned results:")
    for i, r in enumerate(results):
        print(f"  {i+1}. [{r['confidence']*100:.1f}%] {r['text']}")
    print()

    return results
