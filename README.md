## Mini GPT
In this project, we implement a small-scale GPT language model. The model predicts the next character based on the previous characters and is trained on tinystories.txt.

The model construction is based on the transformer structure of the Attention is all you need paper. The implementation follows the ideas of Andrej Karpathy.

There are two models, one uses letters as tokens and one Byte-Pair Encoding (BPE) tokens adapted to the tiny_stories.txt.

Train_BPE_model.py -> Requires a 48 GB NVIDIA GPU and a download of the larger tiny stories file 

Train_letter_model.py-> Runnable on a usual Laptop

The trained BPE model can be downloaded in Hugging face over (https://huggingface.co/Niklas1/GPT_tiny_storys/tree/main)



## How to run the BPE model

To employ the model the user needs to use the terminal. The model can be downloaded by:

hf download Niklas1/GPT_tiny_storys mini_GPT.pt --local-dir 

Then the model can be run over by passing 

python employ.py 






