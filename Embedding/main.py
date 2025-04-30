import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Sample business data (normally this would be loaded from a JSON file)
data = [
    {
        "id": 1,
        "name": "John Doe",
        "position": "Senior Business Strategist",
        "productivity": 85.4,
        "sales": 125000,
        "expenses": 55000,
        "projects_completed": 4,
        "client_satisfaction": 92.0,
        "growth_percentage": 7.8,
        "revenue": 180000,
        "profit": 70000
    },
    {
        "id": 2,
        "name": "Jane Smith",
        "position": "Marketing Manager",
        "productivity": 78.2,
        "sales": 95000,
        "expenses": 30000,
        "projects_completed": 3,
        "client_satisfaction": 88.5,
        "growth_percentage": 5.2,
        "revenue": 140000,
        "profit": 50000
    }
]

# Load the transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    print(f"[DEBUG] Embedding text: {text}")
    embedding = model.encode(text, convert_to_tensor=False)
    print(f"[DEBUG] Embedding vector (first 5 dims): {embedding[:5]}")
    return embedding

def get_most_relevant_field(question, dataset):
    question_vec = embed_text(question)
    best_score = -1
    best_match = None

    print("\n[DEBUG] Matching question with data fields...")
    for entry in dataset:
        for key, value in entry.items():
            field_string = f"{key}: {value}"
            field_vec = embed_text(field_string)
            score = cosine_similarity([question_vec], [field_vec])[0][0]
            print(f"[DEBUG] Compared to '{field_string}' → Score: {score:.4f}")
            if score > best_score:
                best_score = score
                best_match = (key, value, entry)

    print(f"[DEBUG] Best match: {best_match[0]} with score {best_score:.4f}")
    return best_match

def main():
    print("\n🔍 GenAI Business Q&A System (Debug Mode)\n")
    while True:
        question = input("💬 Ask something (or type 'exit'): ").strip()
        if question.lower() == 'exit':
            break
        try:
            key, value, row = get_most_relevant_field(question, data)
            name = row.get("name", "Someone")
            print(f"\n✅ Answer: {name}'s {key} is {value}\n")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()
