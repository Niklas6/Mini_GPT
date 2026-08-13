# Mini GPT

In this project, we implement a small-scale GPT language model. The model predicts the next character based on the previous characters and is trained on tinystories.txt.


The model construction is based on the transformer structure of the Attention is all you need paper. The implementation follows the ideas of Andrej Karpathy. 

This is the first version of the project and will be refined soon. 


# How to use

The model can be run through the 'Employ_model.ipynb' file using the saved parameters in 'tiny_transformer.pt'. The model is trained 'Train_model.py' which takes minutes to hours depending on the CPU or GPU/XPU configuration. 

