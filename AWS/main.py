# Main controller script
from kendra_client import query_kendra
from rules_engine import extract_revenue
from bedrock_summarizer import summarize_result

def main():
    # Step 1: User query
    question = "What was the revenue for Q1 2025?"
    quarter = "Q1"
    year = "2025"

    print("=== FINANCIAL GENAI SYSTEM ===\n")

    # Step 2: Query simulated Kendra
    kendra_results = query_kendra(question)

    # Step 3: Apply rule-based post-processing
    extracted_info = extract_revenue(kendra_results, quarter, year)

    # Step 4: (Optional) Generate natural response using Bedrock-like summarizer
    summary = summarize_result(extracted_info)

    # Step 5: Print result
    print("\n=== FINAL RESPONSE ===")
    print(summary)
    print("======================\n")

if __name__ == "__main__":
    main()
