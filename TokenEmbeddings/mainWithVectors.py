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

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def self_attention(q, k, v):
    d_k = q.shape[-1]
    scores = q @ k.T / np.sqrt(d_k)
    weights = softmax(scores)
    output = weights @ v
    return weights, output

def print_explanation():
    print("=== GENERATIVE AI TEXT INPUT EXPLANATION ===")
    print("\nStep 1: Token Embeddings — Represents the meaning of each word.")
    print("Step 2: Positional Encodings — Adds order info to each token.")
    print("Step 3: Final Input = Embedding + Positional Encoding.")
    print("Step 4: Self-Attention — Determines what each word should focus on.")
    print("=============================================\n")

def print_embedding_analysis(token_embeddings, positional_encodings, final_input, token_labels):
    for i in range(len(token_embeddings)):
        print(f"🧠 Token: '{token_labels[i]}' (Position {i})")
        print("Embedding    :", np.round(token_embeddings[i], 3))
        print("Positional   :", np.round(positional_encodings[i], 3))
        print("Final Input  :", np.round(final_input[i], 3))
        print()

def print_attention_matrix(tokens, weights):
    print("=== SELF-ATTENTION WEIGHTS ===")
    print("Bigger numbers = stronger relationships") 
    print("Rows = focusing token | Columns = attended token\n")
    print(f"{'':<8}" + "".join([f"{t:<8}" for t in tokens]))
    for i, row in enumerate(weights):
        print(f"{tokens[i]:<8}" + "".join([f"{w:<8.2f}" for w in row]))
    print("\n" + "="*50 + "\n")

def run_example(sentence_tokens):
    seq_len = len(sentence_tokens)
    d_model = 8

    token_embeddings = get_dummy_token_embeddings(seq_len, d_model)
    positional_encodings = get_positional_encoding(seq_len, d_model)
    final_input = token_embeddings + positional_encodings

    print(f"🔡 Sentence: {' '.join(sentence_tokens)}")
    print_embedding_analysis(token_embeddings, positional_encodings, final_input, sentence_tokens)

    q = final_input
    k = final_input
    v = final_input
    attention_weights, _ = self_attention(q, k, v)
    print_attention_matrix(sentence_tokens, np.round(attention_weights, 2))

def main():
    print_explanation()
    examples = [
        ["I", "love", "pizza", ".", "<EOS>"],
        ["She", "reads", "books", "every", "night"],
        ["The", "cat", "sat", "on", "the", "mat"]
    ]

    for example in examples:
        run_example(example)

if __name__ == "__main__":
    main()
