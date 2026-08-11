import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

if hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')
# parameters
batch_size = 64
block_size=128

max_iters = 2_000
learning_rate=5e-3
eval_iters = 200

eval_interval=100

n_embd=16*6
n_head=6

n_layer=6
dropout=0.2


#code
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key    = nn.Linear(n_embd,head_size,bias=False)
        self.query  = nn.Linear(n_embd,head_size,bias=False)
        self.value  = nn.Linear(n_embd,head_size,bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
    def forward(self, x ):
        B, T, C = x.shape
        k=self.key(x)
        q=self.query(x)
        wei = q@ k.transpose(-2,-1)*k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei= self.dropout(wei)
        v=self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size,n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out =torch.cat( [h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd,n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x=x + self.sa(self.ln1(x))
        x=x + self.ffwd(self.ln2(x))
        return(x)





#code
class BigramLanguageModel(nn.Module):
    def __init__(self,):#vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        #self.sa_head=MultiHeadAttention(4,n_embd//4 )
        #self.ffwd =FeedForward(n_embd)
        #self.blocks = nn.Sequential(
        #    Block(n_embd,n_head=4),
        #    Block(n_embd,n_head=4),
        #    Block(n_embd,n_head=4),
        #)
        self.blocks = nn.Sequential(*[Block(n_embd,n_head) for _ in range(n_layer)])

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self,idx,targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x=tok_emb +pos_emb
        #x = self.sa_head(x)
        #x = self.ffwd(x)
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss=None
        else:
            B, T ,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss

    def generate(self, idx,max_new_tokens):
        for _ in range(max_new_tokens):
            #print(idx)
            idx_cond = idx[:,-block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples =1)
            idx= torch.cat((idx,idx_next), dim=1)
        return idx

#m = BigramLanguageModel(vocab_size)
#logits,loss=m(xb,yb)

def get_batch(split):
    data =train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out





if __name__ == '__main__':
    #print('device: ', device)
    with open('tinystories.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    #print(len(text))
    #print(text[:500])
    #encoder and decoder
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    #train test split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    xb, yb = get_batch('train')

    #we define and train the BigramLanguageModel
    model = BigramLanguageModel()#vocab_size)
    m= model.to(device)
    logits, loss = m(xb, yb)
    optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

    for steps in range(max_iters):


        if steps % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    m.eval()
    checkpoint = {
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in m.state_dict().items()
        },
        "chars": chars,
        "config": {
            "block_size": block_size,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "dropout": dropout,
        },
    }

    torch.save(checkpoint, "tiny_transformer.pt")
    print("Model saved to tiny_transformer.pt")

    with torch.no_grad():
        for token_id in [38, 34, 31, 27]:
            start = torch.tensor([[token_id]], dtype=torch.long, device=device)
            generated = m.generate(start, max_new_tokens=500)
            print(decode(generated[0].tolist()))
