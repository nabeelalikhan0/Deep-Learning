import torch
from torch import nn
import torch.nn.functional as F

# hyperparams
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
learning_rate = 1e-2
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ---------------------------------

torch.manual_seed(1337)

with open("input.txt") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: [
    "".join(itos[c] for c in l)
]  # decoder: take a list of integers, output a string


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size,bias=False)
        self.query = nn.Linear(n_embd, head_size,bias=False)
        self.value = nn.Linear(n_embd, head_size,bias=False)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))

        def forward(self,x):
            B,T,C = x.shape
            k = self.key(x)
            q = self.query(x)
            # compute attention scores ("affinities")
            wei = q@k.transpose(-2,-1) * C**-0.5 
            wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
            # perform the softmax to get the probabilities
            v = self.value(x)
            out = wei @ v 
            return out 


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logtis for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size,vocab_size)

        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)


    def forward(self, idx, target=None):
        B,T = idx.shape

        # idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))
        x = tok_emb + pos_emb # BTC
        x = self.sa_head(x) # apply one head self attention
        logits = self.lm_head(x) # B,T,vocab_size

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)  # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}: val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, 500)[0].tolist()))
