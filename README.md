# Mini GPT Character Transformer

This project implements a small GPT-style character-level language model in PyTorch. It is trained on the TinyStories dataset and generates text one character at a time.

The code is intended as an educational implementation of the main transformer components rather than a production language model.

## Model architecture

`LLM_train.py` contains:

- Character token embeddings
- Learned positional embeddings
- Causal multi-head self-attention
- Feed-forward networks
- Residual connections
- Layer normalization
- Dropout
- A linear language-model head

The current default configuration is:

| Parameter | Value |
|---|---:|
| Batch size | 64 |
| Context length | 128 characters |
| Embedding dimension | 96 |
| Attention heads | 6 |
| Transformer layers | 6 |
| Dropout | 0.2 |
| Training iterations | 2,000 |
| Learning rate | 0.005 |

## Requirements

- Python 3.10 or newer
- PyTorch
- NumPy
- pandas

Install the standard packages with:

```bash
python -m pip install torch numpy pandas
```

### Intel GPU support

The training script prefers an Intel XPU when one is available and otherwise runs on the CPU:

```python
if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")
```

For a supported Intel GPU, install the XPU build of PyTorch:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

Verify the installation with:

```python
import torch

print(torch.__version__)
print(torch.xpu.is_available())
```

## Dataset

Training expects a UTF-8 text file named `tinystories.txt` in the project directory:

```text
LLM/
├── LLM_train.py
└── tinystories.txt
```

The script creates a vocabulary from the unique characters in the file and uses the first 90% of the encoded text for training and the remaining 10% for validation.

## Training

Run:

```bash
python LLM_train.py
```

At startup, the script reports the selected device:

```text
Using device: xpu
```

During training, it periodically prints average training and validation loss. After training, it generates several text samples and saves a checkpoint to:

```text
tiny_transformer.pt
```

## Saved checkpoint

The checkpoint contains:

- `model_state_dict`: trained model parameters stored on the CPU for portability
- `chars`: the character vocabulary
- `config`: context length and model dimensions

Saving tensors on the CPU allows the checkpoint to be loaded later on a CPU or supported accelerator. Loading the checkpoint does not train the model again.

## Loading the trained model

The model architecture must be reconstructed before loading its trained parameters. With the current code, the architecture can be imported from `LLM_train.py`:

```python
import torch
import LLM_train as llm

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

checkpoint = torch.load("tiny_transformer.pt", map_location="cpu")
chars = checkpoint["chars"]
config = checkpoint["config"]

# The current architecture reads these values as module-level globals.
llm.vocab_size = len(chars)
llm.block_size = config["block_size"]
llm.n_embd = config["n_embd"]
llm.n_head = config["n_head"]
llm.n_layer = config["n_layer"]
llm.dropout = config["dropout"]

model = llm.BigramLanguageModel()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

stoi = {character: index for index, character in enumerate(chars)}
itos = {index: character for index, character in enumerate(chars)}

prompt = "Once upon a time"
tokens = torch.tensor(
    [[stoi[character] for character in prompt]],
    dtype=torch.long,
    device=device,
)

with torch.no_grad():
    generated = model.generate(tokens, max_new_tokens=500)

text = "".join(itos[token] for token in generated[0].tolist())
print(text)
```

## Limitations

- Tokenization is character-based rather than subword-based.
- The context window is limited to 128 characters.
- Text generation samples directly from the predicted distribution and has no temperature or top-k controls.
- The model configuration currently relies on module-level global variables.
- Dataset loading and batch construction occur on the CPU and can limit accelerator utilization.

## Project files

- `LLM_train.py` — transformer definition, training, checkpoint saving, and sample generation
- `train.py` — earlier character-level language-model experiment
- `tiny_transformer.pt` — saved trained checkpoint
- `Try.ipynb` and `try2.ipynb` — development notebooks
- `download.ipynb` — dataset preparation notebook
