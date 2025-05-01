# Simulated Amazon Bedrock summarizer (Claude, Titan, etc.)

def summarize_result(info):
    """
    Mocked summarization — like what an LLM via Bedrock would return.
    """
    if not info:
        return "I'm sorry, I couldn't find revenue data for the given period."

    return (
        f"Based on financial documents, the revenue for the specified quarter is "
        f"{info['revenue']}. This value was reported in a statement with "
        f"{info['confidence']*100:.1f}% confidence."
    )
