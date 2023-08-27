from typing import List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

with open("input.txt", "r", encoding="utf-8") as f:
    input_text = f.read()


chars = sorted(set(input_text))
vocabulary_size = len(chars)
print("".join(chars))
print(vocabulary_size)

i2char = dict(enumerate(chars))
char2i = {v: k for k, v in i2char.items()}


def encode(text: str) -> List[int]:
    return [char2i[c] for c in text]


def decode(indices: List[int]) -> str:
    return "".join(i2char[i] for i in indices)


data = torch.tensor(encode(input_text), dtype=torch.long)
print(data.shape, data.dtype)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)

# hyperparameters
batch_size = 4
block_size = 8  # maximum context length
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

xb = train_data[:block_size]
yb = train_data[1 : block_size + 1]
for t in range(block_size):
    context = xb[: t + 1]
    target = xb[t]
    print(context, " -> ", target.item())


def get_batch(split: str) -> Tuple[Tensor, Tensor]:
    assert split in ("train", "val")
    d = train_data if split == "train" else val_data
    idx = torch.randint(len(d) - batch_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in idx])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    result = dict()
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss
        result[split] = losses.mean()

    model.train()
    return result


xb, yb = get_batch("train")
print()
print(xb.shape)
print(xb)
print(yb.shape)
print(yb)

b = 0  # only look at first batch
for t in range(block_size):
    context = xb[b, : t + 1]
    target = yb[b, t]
    print(context, " -> ", target.item())


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token reads logits for next token from a lookup table
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: Tensor, targets: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # idx and targets are of shape (B, T)

        # B: batch-size
        # T: time, block-size
        # C: channels, vocabulary size
        logits = self.token_embedding(idx)  # (B, T, C)  = (4, 8, 65)

        if targets is None:
            return logits

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        # idx: (B, T) array of indices in the current context

        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

        return idx


model = BigramLanguageModel(vocabulary_size)
model = model.to(device)
out, l = model.forward(xb, yb)
print()
print(out.shape, l)

start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(single_idx, max_new_tokens=10)[0].tolist()))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 32
for step in range(10000):
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print()
print(loss.item())
print(decode(model.generate(start_context, max_new_tokens=100)[0].tolist()))
