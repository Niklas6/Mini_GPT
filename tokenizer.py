from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


#with open('TinyStoriesV2-GPT4-train.txt', 'r', encoding='utf-8') as f:
#    text = f.read()

tokenizer = Tokenizer(
        BPE(unk_token="<unk>")
    )

tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )
tokenizer.decoder = ByteLevelDecoder()
trainer = BpeTrainer(
        vocab_size=1024,
        min_frequency=2,
        special_tokens=["<unk>"],
        # initial_alphabet=ByteLevel.alphabet(),
    )

tokenizer.train(
        ["TinyStoriesV2-GPT4-train.txt"],
        trainer,
    )

tokenizer.save("tokenizer.json")

