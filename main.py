import requests, os


def get_data():
    dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(dataset_url)

    return response.text

    # if not os.path.exists('data'):
    #   os.mkdir('data')
    # with open('data/input.txt', 'w') as f:
    #   f.write(response.text)


text = get_data()

# Grab unique characters that occur in text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # takes string and outputs a list of integers
decode = lambda l: "".join(
    itos[i] for i in l
)  # takes list of integers and outputs a string

# simple example
string = "hii there"
encoding = encode(string)  # long sequences
decode(encoding)

# Encode text dataset and store in torch.Tensor
import torch

data = torch.tensor(encode(text), dtype=torch.long)

data.shape, data.dtype
data[:1000]

# split data into train and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
# using sliding window approach where overlap differs by 1
x = train_data[:block_size]
y = train_data[1 : block_size + 1]

for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    print(f"when input is {context} the target is {target}")

# batching (group code like up above but for multiple sequences)
torch.manual_seed(1337)
batch_size = 4
block_size = 8


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    # high is off-by-1 becuase we need to leave room for y value
    ix = torch.randint(high=len(data) - block_size, size=(batch_size,))  # 4 x 1

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")  # 4 x 8
xb.shape, xb
yb.shape, yb

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target is {target}")

# Implementation of bigramlanguagemodel
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=vocab_size
        )  # 65 x 65 (C x C)

    def forward(self, idx, targets):
        # idx and targets are both (B,T) tensor of integers -> 4 x 8
        logits = self.token_embedding_table(idx)  # (B, T, C)

        return logits


m = BigramLanguageModel(vocab_size=vocab_size)
out = m(xb, yb)
out.shape  # 4, 8, 65
