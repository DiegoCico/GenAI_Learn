import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
vocab_size = 100  # pretend small vocab
embed_size = 64
num_heads = 4
hidden_dim = 128
num_layers = 2
seq_length = 16
batch_size = 32

# 1. Tiny Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_size)
        )
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_output)  # residual connection
        ff_output = self.ff(x)
        x = self.ln2(x + ff_output)    # another residual
        return x

# 2. Mini GPT Model
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, hidden_dim):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(seq_length, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, hidden_dim) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        tokens = self.token_embed(x)
        positions = self.pos_embed(torch.arange(T, device=x.device))
        x = tokens + positions

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        logits = self.fc_out(x)
        return logits

# 3. Instantiate model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniGPT(vocab_size, embed_size, num_layers, num_heads, hidden_dim).to(device)

# 4. Fake Data for Training (just random numbers for demo)
x = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
y = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)

# 5. Train for a few steps
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(100):  # normally you'd do millions
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

