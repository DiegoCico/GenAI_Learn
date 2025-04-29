# model.py
import numpy as np

def get_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads

def get_dummy_token_embeddings(seq_len, d_model):
    np.random.seed(42)
    return np.random.rand(seq_len, d_model)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def self_attention(q, k, v):
    d_k = q.shape[-1]
    scores = q @ k.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ v, weights

class SimpleGenAI:
    def __init__(self, d_model=32):
        self.d_model = d_model

    def embed(self, tokens):
        seq_len = len(tokens)
        emb = get_dummy_token_embeddings(seq_len, self.d_model)
        pos = get_positional_encoding(seq_len, self.d_model)
        return emb + pos

    def encode(self, tokens):
        emb = self.embed(tokens)
        out, attn = self_attention(emb, emb, emb)
        return out.mean(axis=0), attn

    def rank_fields(self, q_vec, f_vecs):
        scores = {f: float(q_vec @ vec) for f, vec in f_vecs.items()}
        names, vals = zip(*scores.items())
        probs = softmax(np.array(vals))
        return sorted(zip(names, probs), key=lambda x: x[1], reverse=True)

    def answer(self, question, fields):
        tokens = question.lower().split()
        print(f"> Tokenized question: {tokens}")
        q_vec, attn = self.encode(tokens)
        print(f"> Question vec (first 5 dims): {q_vec[:5]}")
        print(f"> Attention shape: {attn.shape}")

        f_vecs = {}
        for name, val in fields.items():
            doc = f"{name} {val}".split()
            vec, _ = self.encode(doc)
            f_vecs[name] = vec

        ranked = self.rank_fields(q_vec, f_vecs)
        print("> Top matches:")
        for nm, p in ranked[:3]:
            print(f"  • {nm}: {p:.4f}")

        best, _ = ranked[0]
        return f"Your {best} is {fields[best]}."
