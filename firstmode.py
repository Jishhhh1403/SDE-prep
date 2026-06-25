import torch
import torch.nn as nn
import os
import tiktoken
import time
import math # Added missing import
import torch.nn.functional as F # Added missing import
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device="cuda" if torch.cuda.is_available else "CPU"
if device == "cuda":
  print(torch.cuda.get_device_name())

data_url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not os.path.exists("data.txt"):
  import urllib.request
  urllib.request.urlretrieve(data_url,"data.txt")

with open("data.txt","r") as f:
  text=f.read()
print("total cahracters:" ,len(text))

# text1="hello this is just a test sentance, LOL so funny. i am hardcoding this by the way"

tokenizer=tiktoken.get_encoding("cl100k_base")
tokens=tokenizer.encode(text)
# for token in tokens:
#     print(f"{token} -->'{tokenizer.decode([token])}'")

# training data prep
data=torch.tensor(tokens,dtype=torch.long)
# 90% training 10% val
n = int(0.9 * len(data))
train_data = data[:n]
train_data=train_data.to(device)
val_data = data[n:]
val_data=val_data.to(device)
print(f"Training tokens:   {len(train_data):,}")
print(f"Validation tokens: {len(val_data):,}")

# Batching: grab random chunks of text
def get_batch(split, batch_size, context_length):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - context_length, (batch_size,))
    x = torch.stack([d[i:i+context_length] for i in ix])
    y = torch.stack([d[i+1:i+context_length+1] for i in ix])
    return x.to(device), y.to(device)
# implementing rms
class RMSNorm(nn.Module):
  def __init__ (self,dim,eps=1e-6):
    super().__init__()
    self.eps=eps
    self.weights= nn.Parameter(torch.ones(dim))
  def forward(self ,x ):
    rms=torch.sqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)
    return (x/rms)*self.weights

# implementing rope
def precompute_rope(head_dim, max_seq_len,base=1000.0):
  freq=1.0/(base**(torch.arange(0,head_dim,2).float()/head_dim))
  positions=torch.arange(max_seq_len).float()
  angles=torch.outer(positions,freq)
  return torch.cos(angles),torch.sin(angles)
def apply_rope(x,cos,sin):
      seq_len = x.shape[2]
      cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, hd//2]
      sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

      x1 = x[..., ::2]   # even dims
      x2 = x[..., 1::2]  # odd dims

      out1 = x1 * cos - x2 * sin
      out2 = x1 * sin + x2 * cos

      return torch.stack([out1, out2], dim=-1).flatten(-2)

# GQA
def repeat_kv(x, n_rep):
    """
    Repeat KV heads to match the number of query heads.
    x: [batch, n_kv_heads, seq_len, head_dim]
    Returns: [batch, n_kv_heads * n_rep, seq_len, head_dim]
    """
    if n_rep == 1:
        return x
    b, n_kv, seq, hd = x.shape
    return (x[:, :, None, :, :]
            .expand(b, n_kv, n_rep, seq, hd)
            .reshape(b, n_kv * n_rep, seq, hd))


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention with RoPE.
    n_heads query heads, n_kv_heads key/value heads.
    Groups of (n_heads // n_kv_heads) query heads share one KV pair.
    """
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        b, seq, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape into heads
        q = q.view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (not V!)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Repeat KV heads to match Q heads
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        # Dropout on attention weights (regularization)
        weights = F.dropout(weights, p=DROPOUT, training=self.training)

        out = weights @ v

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(b, seq, -1)
        return self.o_proj(out)
# implement swiglu
class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Two paths:
      gate: SiLU(x @ W_gate) - controls flow
      up:   x @ W_up         - carries information

    Combined: gate * up -> W_down

    SiLU(x) = x * sigmoid(x), a smooth version of ReLU.
    """
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up   = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        up   = self.w_up(x)
        return F.dropout(self.w_down(gate * up), p=DROPOUT, training=self.training)

# final assemble of model
class TransformerBlock(nn.Module):
    """
    One layer of a modern transformer.

    Pre-norm architecture:
      x -> RMSNorm -> GQA Attention -> + residual
      x -> RMSNorm -> SwiGLU FFN     -> + residual
    """
    def __init__(self, d_model, n_heads, n_kv_heads, ffn_hidden_dim):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.ffn_norm  = RMSNorm(d_model)
        self.ffn       = SwiGLU(d_model, ffn_hidden_dim)

    def forward(self, x, rope_cos, rope_sin):
        # Pre-norm -> Attention -> Residual
        x = x + self.attention(self.attn_norm(x), rope_cos, rope_sin)
        # Pre-norm -> FFN -> Residual
        x = x + self.ffn(self.ffn_norm(x))
        return x

class MiniLLM(nn.Module):
    """
    A small but modern language model.

    Architecture: modern transformer with all 4 upgrades.
    Training objective: next character prediction.
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, n_kv_heads,
                 ffn_hidden_dim, max_seq_len):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding (no positional embedding -- RoPE handles position)
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, ffn_hidden_dim)
            for _ in range(n_layers)
        ])

        # Final norm and output head
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding and output weights
        self.lm_head.weight = self.token_emb.weight

        # Precompute RoPE frequencies
        head_dim = d_model // n_heads
        rope_cos, rope_sin = precompute_rope(head_dim, max_seq_len)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, seq_len = idx.shape

        # Token embedding
        x = self.token_emb(idx)

        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)

        # Final norm + project to vocabulary
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss


# --- Model Configuration ---

config = {
    "vocab_size":     vocab_size, # Corrected vocab_size
    "d_model":        256,
    "n_layers":       4,
    "n_heads":        8,
    "n_kv_heads":     2,
    "ffn_hidden_dim": 680,
    "max_seq_len":    256,
}

model = MiniLLM(**config).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 50)
print("  MODEL SUMMARY")
print("=" * 50)
print(f"  Vocabulary:      {config['vocab_size']}")
print(f"  Embedding dim:   {config['d_model']}")
print(f"  Layers:          {config['n_layers']}")
print(f"  Query heads:     {config['n_heads']}")
print(f"  KV heads:        {config['n_kv_heads']} (GQA ratio: {config['n_heads']//config['n_kv_heads']}:1)")
print(f"  FFN hidden dim:  {config['ffn_hidden_dim']}")
print(f"  Context length:  {config['max_seq_len']}")
print(f"  Head dim:        {config['d_model'] // config['n_heads']}")
print(f"{'=' * 50}")
print(f"  Total parameters:     {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size (approx):  {total_params * 4 / 1e6:.1f} MB (float32)")
print(f"{'=' * 50}")



# training
BATCH_SIZE = 64
CONTEXT_LEN = config["max_seq_len"]
LEARNING_RATE = 3e-4
MAX_STEPS = 3000
EVAL_INTERVAL = 250
EVAL_STEPS = 20
LOG_INTERVAL = 50

# DROPOUT was not defined. Assuming a default value based on common practice in similar models.
# Added DROPOUT definition here.
DROPOUT = 0.1 

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and val splits."""
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(EVAL_STEPS):
            xb, yb = get_batch(split, BATCH_SIZE, CONTEXT_LEN)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out
# --- Training Loop ---
print("Starting training...")
print(f"  {MAX_STEPS} steps, batch_size={BATCH_SIZE}, context_len={CONTEXT_LEN}")
print(f"  Evaluating every {EVAL_INTERVAL} steps")
print("-" * 60)

train_losses = []
val_losses = []
step_log = []
start_time = time.time()

model.train()
for step in range(MAX_STEPS):
    xb, yb = get_batch("train", BATCH_SIZE, CONTEXT_LEN)

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step % LOG_INTERVAL == 0:
        elapsed = time.time() - start_time
        print(f"  Step {step:5d}/{MAX_STEPS} | Loss: {loss.item():.4f} | Time: {elapsed:.0f}s")

    if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
        losses = estimate_loss()
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])
        step_log.append(step)
        if step > 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            remaining = (MAX_STEPS - step) / steps_per_sec
            print(f"  >>> Eval @ step {step}: train={losses['train']:.4f}, val={losses['val']:.4f} | ~{remaining:.0f}s remaining")

total_time = time.time() - start_time
print("-" * 60)
print(f"Training complete! Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"Final train loss: {train_losses[-1]:.4f}")
print(f"Final val loss:   {val_losses[-1]:.4f}")


@torch.no_grad()
def generate(model, prompt, max_new_tokens=500, temperature=0.8):
    """
    Generate text autoregressively.

    temperature controls randomness:
      low (0.3)  -> conservative, repetitive
      mid (0.8)  -> balanced
      high (1.2) -> creative, chaotic
    """
    model.eval()
    tokens = encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        context = tokens[:, -config["max_seq_len"]:]
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

    return decode(tokens[0].tolist())
prompt = "ROMEO:"

print("=" * 60)
print(f"  PROMPT: {prompt!r}")
print("=" * 60)

for temp in [0.5, 0.8, 1.0, 1.2]:
    print(f"\n{'_' * 60}")
    print(f"  Temperature = {temp}")
    print(f"{'_' * 60}")
    output = generate(model, prompt, max_new_tokens=300, temperature=temp)
    print(output)