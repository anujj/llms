# Import necessary libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
import time

# -------------------------- Data Preparation -------------------------- #
# Read the file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# All the unique unicode characters in the training dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(f"Vocab size: {vocab_size}")

# Create a tokenizer for training the dataset
itos = {i: ch for i, ch in enumerate(chars)}
stoi = {ch: i for i, ch in enumerate(chars)}
encoder = lambda s: [stoi[c] for c in s]
decoder = lambda l: [itos[i] for i in l]

# -------------------------- Hyperparameters -------------------------- #

batch_size = 64
block_size = 256
max_iters = 10000
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layers = 6
dropout_rate = 0.1  # Added dropout rate

print(f"Using device: {device}")

# -------------------------- Data Tokenization -------------------------- #

# Tokenize the entire data for training the NN
data = torch.tensor(encoder(text), dtype=torch.long, device=device)  # Move data to device

# Create training and validation data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Visualizing the concept
print("\nVisualizing data samples:")
for i in range(block_size):
    x = data[i:i + block_size]
    y = data[i + 1:i + block_size + 1]
    print(f"Input: {x.tolist()} --> Target: {y.tolist()}")

# -------------------------- Batch Generation -------------------------- #

torch.manual_seed(1337)

def get_batch(split):
    # Select the data split
    data_split = train_data if split == 'train' else val_data
    # Randomly choose batch indices
    ix = torch.randint(len(data_split) - block_size, (batch_size,), device=device)
    # Gather input and target sequences
    x = torch.stack([data_split[i: i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1: i + block_size + 1] for i in ix])
    return x, y


# -------------------------- Model Definition -------------------------- #

class Head(nn.Module):
    """ One head for the self-attention block """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)  # Added dropout
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # Scaled dot-product attention
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Causal masking
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)  # Apply dropout to attention weights
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention running in parallel """

    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_head)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_rate)  # Added dropout

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate outputs of all heads
        out = self.projection(out)
        out = self.dropout(out)  # Apply dropout after projection
        return out

class FeedForward(nn.Module):
    """ Feed-forward neural network """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # Changed to GELU activation
            nn.Dropout(dropout_rate),  # Added dropout
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_rate),  # Added dropout
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block consisting of attention and feed-forward layers """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Residual connection
        x = x + self.ffwd(self.ln2(x))  # Residual connection
        return x

class BigramLanguageModel(nn.Module):
    """ Language model based on the transformer architecture """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout_rate)  # Added dropout
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx is the incoming token indices
        B, T = idx.shape
        # Token embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        # Positional embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.dropout(x)  # Apply dropout to embeddings
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # (B, T)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_token), dim=1)  # (B, T+1)
        return idx

# Initialize the model and move it to the device
model = BigramLanguageModel(vocab_size).to(device)

# -------------------------- Training Setup -------------------------- #

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Function to estimate loss on a given split
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        total_loss = 0
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            total_loss += loss.item()
        avg_loss = total_loss / eval_iters
        losses[split] = avg_loss
    model.train()
    return losses

# -------------------------- Training Loop -------------------------- #

print("\nStarting training...\n")
best_val_loss = float('inf')
start_time = time.time()

for iter in range(1, max_iters + 1):
    # Determine the split (train or val)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        train_loss = losses['train']
        val_loss = losses['val']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model
            torch.save(model.state_dict(), 'best_model.pth')
        print(f"Step {iter}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    else:
        # Get a batch of training data
        xb, yb = get_batch('train')
        # Forward pass
        logits, loss = model(xb, yb)
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Update parameters
        optimizer.step()

    # Optional: Early stopping or other stopping criteria can be added here

end_time = time.time()
elapsed = end_time - start_time
print(f"\nTraining completed in {elapsed:.2f} seconds.")
print(f"Best Val Loss: {best_val_loss:.4f}")

# -------------------------- Generation -------------------------- #

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Generation example
with torch.no_grad():
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)  # Initialize with a starting token (e.g., index 0)
    generated_idx = model.generate(idx, 1000)
    generated_text = ''.join(decoder(generated_idx[0].tolist()))
    print("\nGenerated Text:\n")
    print(generated_text)
