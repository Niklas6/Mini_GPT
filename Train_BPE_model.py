import pandas as pd
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import Train_letter_model as llm
#import Train_BPE_model as bpellm

import math
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

if hasattr(torch, 'cuda') and torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')

print(f'Using device: {device}, for training')
# parameters
batch_size = 64
block_size=256

max_iters = 25_000

learning_rate=1e-3
min_lr=1e-5


eval_iters = 50

eval_interval=  500
save_interval = 1000

n_embd=64*12
n_head=12

n_layer=12
dropout=0.25


warmup_steps=100

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
class Tiny_GPT(nn.Module):
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



def lr_decay(step):

    if step< warmup_steps:
        return learning_rate *(step+1)/warmup_steps
    return min_lr + (learning_rate-min_lr)*0.5*(1+math.cos(math.pi*((step-warmup_steps))/(max_iters-warmup_steps)))



if __name__ == '__main__':

    mode = input(
    "Start a new model or resume training? [new/resume]: "
        ).strip().lower()
    

    
    #print('device: ', device)
    with open('TinyStoriesV2-GPT4-train.txt', 'r', encoding='utf-8') as f:
        text_train = f.read(100_000_000)
    with open('TinyStoriesV2-GPT4-valid.txt', 'r', encoding='utf-8') as f:
        text_valid = f.read(1_000_000)
    vocab_size = tokenizer.get_vocab_size()

    train_data = torch.tensor(tokenizer.encode(text_train).ids, dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(text_valid).ids, dtype=torch.long)




    xb, yb = get_batch('train')

    #we define and train the BigramLanguageModel
    if mode == 'new':
        model = Tiny_GPT()
        warmup_steps=100
    if mode == 'resume':
        checkpoint = torch.load(
            "tiny_transformer_bpe.pt",
            map_location=device,
            weights_only=True,
            )

        config = checkpoint["config"]

        model = Tiny_GPT()

        model.load_state_dict(
            checkpoint["model_state_dict"]
            )
        warmup_steps=25
    
    m= model.to(device)
    logits, loss = m(xb, yb)
    optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

    
    
    print('Training starts')

    start = time.time()
    for steps in range(max_iters):
        lr = lr_decay(steps)

        for group in optimizer.param_groups:
            group["lr"] = lr


        if steps % eval_interval == 0 and steps!=0:
            losses = estimate_loss()
            print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, learning rate {lr_decay(steps)}, time { round(time.time() - start,2)}")
            start = time.time()

            with open('TinyStoriesV2-GPT4-train.txt', 'r', encoding='utf-8') as f:
                text_train = f.read(100_000_000)
                        
            train_data = torch.tensor(tokenizer.encode(text_train).ids, dtype=torch.long)
        
        if steps!=0 and steps % save_interval == 0:
            m.eval()
            checkpoint = {
                            "model_state_dict": {
                                name: tensor.detach().cpu()
                                for name, tensor in m.state_dict().items()
                            },
                            "config": {
                                "vocab_size": vocab_size,
                                "block_size": block_size,
                                "n_embd": n_embd,
                                "n_head": n_head,
                                "n_layer": n_layer,
                                "dropout": dropout,
                            },
                            "tokenizer_path": "tokenizer.json",
                        }
                
            torch.save(checkpoint, "tiny_transformer_bpe.pt")
            
            model.train()


            
            
            

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
                "config": {
                    "vocab_size": vocab_size,
                    "block_size": block_size,
                    "n_embd": n_embd,
                    "n_head": n_head,
                    "n_layer": n_layer,
                    "dropout": dropout,
                },
                "tokenizer_path": "tokenizer.json",
            }
    
    torch.save(checkpoint, "tiny_transformer_bpe.pt")
    

    print("Model saved to tiny_transformer_bpe.pt")

    with torch.no_grad():
        for token_id in [38]:
            start = torch.tensor([[token_id]], dtype=torch.long, device=device)
            generated = m.generate(start, max_new_tokens=100)
            print(tokenizer.decode(generated[0].tolist()))
