import torch
import Train_letter_model as llm
import Train_BPE_model as bpellm

from tokenizers import Tokenizer



device = torch.device("cpu")

# Load checkpoint
checkpoint = torch.load(
    "mini_GPT.pt",
    map_location="cpu",
    weights_only=True,
)

config = checkpoint["config"]

# Set the globals used by your model classes
bpellm.vocab_size = config["vocab_size"]
bpellm.block_size = config["block_size"]
bpellm.n_embd = config["n_embd"]
bpellm.n_head = config["n_head"]
bpellm.n_layer = config["n_layer"]
bpellm.dropout = config["dropout"]

# Reconstruct model
model = bpellm.Tiny_GPT()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

# Load the matching tokenizer
tokenizer = Tokenizer.from_file(
    checkpoint["tokenizer_path"]
)


prompt = "Once upon a time there was a young women called Yourong."

prompt_ids = tokenizer.encode(prompt).ids

input_tokens = torch.tensor(
    [prompt_ids],
    dtype=torch.long,
    device=device,
)

with torch.no_grad():
    generated = model.generate(
        input_tokens,
        max_new_tokens=1_000,
    )

generated_text = tokenizer.decode(
    generated[0].tolist()
)

print(generated_text)