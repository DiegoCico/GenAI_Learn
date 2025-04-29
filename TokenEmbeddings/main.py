import numpy as np

def get_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads

def get_dummy_token_embeddings(seq_len, d_model):
    np.random.seed(42)
    return np.random.rand(seq_len, d_model)

def print_explanation():
    print("=== GENERATIVE AI TEXT INPUT EXPLANATION ===")
    print("\nStep 1: Token Embeddings")
    print("→ These are vectors representing the meaning of each word.")
    print("Example: 'love' → [0.6, 0.8, ..., 0.2]\n")

    print("Step 2: Positional Encodings")
    print("→ These help the model understand word order (1st, 2nd, etc.).")
    print("Example for Position 0: [0.0, 1.0, 0.0, 1.0, ...]")
    print("Example for Position 1: [0.84, 0.54, 0.10, 0.99, ...]\n")

    print("Step 3: Final Input = Embedding + Positional Encoding")
    print("→ This tells the model both the word’s meaning and position.")
    print("Example: [0.6, 0.8, ...] + [0.84, 0.54, ...] = [1.44, 1.34, ...]\n")
    print("=============================================")

def print_detailed_embedding_analysis(token_embeddings, positional_encodings, final_input, token_labels):
    for i in range(len(token_embeddings)):
        print(f"\n🧠 Token: '{token_labels[i]}' (Position {i})")
        print("Embedding    :", np.round(token_embeddings[i], 3))
        print("Positional   :", np.round(positional_encodings[i], 3))
        print("Final Input  :", np.round(final_input[i], 3))

# Simulated input tokens
tokens = ["I", "love", "pizza", ".", "<EOS>"]
seq_len = len(tokens)
d_model = 8

# Run explanation
print_explanation()

# Generate and analyze embeddings
token_embeddings = get_dummy_token_embeddings(seq_len, d_model)
positional_encodings = get_positional_encoding(seq_len, d_model)
final_input = token_embeddings + positional_encodings
print_detailed_embedding_analysis(token_embeddings, positional_encodings, final_input, tokens)
