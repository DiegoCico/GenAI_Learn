# Main controller script
from kendra_client import query_kendra
from rules_engine import extract_revenue
from bedrock_summarizer import summarize_result

def main():
    print("=== 💼 FINANCIAL GENAI SYSTEM (SIMULATED) ===\n")

    # Step 1: User enters a natural language financial question
    user_question = "What was the revenue for Q1 2025?"
    quarter = "Q1"
    year = "2025"

    print(f"🔹 USER ASKED: \"{user_question}\"\n")

    # Step 2: Simulate Amazon Kendra returning top semantic matches
    kendra_results = query_kendra(user_question)

    # Step 3: Run rule-based post-processing to extract the structured value
    extracted_info = extract_revenue(kendra_results, quarter, year)

    # Step 4: Send final structured result to simulated LLM (via Bedrock) for summarization
    print("\n🧠 Sending result to LLM (simulated via Amazon Bedrock)...\n")
    summary = summarize_result(extracted_info)

    # Step 5: Final display to end-user
    print("=== ✅ FINAL RESPONSE ===")
    print(summary)
    print("=========================\n")

if __name__ == "__main__":
    main()
