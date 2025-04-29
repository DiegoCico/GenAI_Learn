import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters (tweakable settings)
vocab_size = 100        # size of your vocabulary (number of unique tokens)
embed_size = 64         # size of each word vector
num_heads = 4           # number of attention heads
hidden_dim = 128        # size of hidden layer in feedforward network
num_layers = 2          # number of transformer layers
seq_length = 16         # length of each input sequence
batch_size = 32         # number of sequences per training batch

# 1. A single Transformer block (used multiple times)
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, hidden_dim):
        super().__init__()
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(embed_size, heads, batch_first=True)
        # Feedforward neural network after attention
        self.ff = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_size)
        )
        # LayerNorms and residual connections
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)     # self-attention (Q=K=V=x)
        x = self.ln1(x + attn_output)                # add & normalize (residual connection)
        ff_output = self.ff(x)                       # feedforward network
        x = self.ln2(x + ff_output)                  # add & normalize again
        return x

# 2. The overall GPT-style model
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, hidden_dim):
        super().__init__()
        # Token embedding: turns token IDs into vectors
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        # Positional embedding: encodes position (0 to seq_length)
        self.pos_embed = nn.Embedding(seq_length, embed_size)
        # Stack of Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, hidden_dim) for _ in range(num_layers)
        ])
        # Final normalization and output projection
        self.ln_final = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)  # project back to vocabulary size

    def forward(self, x):
        B, T = x.shape
        tokens = self.token_embed(x)  # shape: (B, T, embed_size)
        positions = self.pos_embed(torch.arange(T, device=x.device))  # shape: (T, embed_size)
        x = tokens + positions  # add token + positional info

        for layer in self.layers:
            x = layer(x)  # pass through each transformer block

        x = self.ln_final(x)       # normalize final output
        logits = self.fc_out(x)    # shape: (B, T, vocab_size)
        return logits

# 3. Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniGPT(vocab_size, embed_size, num_layers, num_heads, hidden_dim).to(device)

# 4. Generate fake data for training (random inputs & targets)
x = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
y = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)

# 5. Training loop (very simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(100):  # run 100 training steps (not enough for real use)
    logits = model(x)  # get output predictions
    # reshape both to (batch_size * seq_length, vocab_size) for loss calculation
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    
    optimizer.zero_grad()  # reset gradients
    loss.backward()        # compute gradients
    optimizer.step()       # update model weights

    if step % 10 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
