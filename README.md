# SequenceToSequence
My implementation of Sequence To Sequence model (Seq2Seq) using Tensorflow. This encoder / decode allow learning semantic representations of variables sequences (like words or sentenses) using Gated Recurrents Units (GRU cells).

## Random letter sequences example
File: random_sequences.py  
The implemented exemple in the main.py file is a word embedding model that take letters as features sequences (26 features as one hot vector for 15 step random sequences).

### Grid search hyperparameter optimization
A grid search hyperparameter optimization was done on this example.  
Optimized hyperparameters are:  
- Learning Rate (optimal value: 1e-2)
- Number of hidden lstm units (optimal value: 250)
- Training batch size (optimal value: 50)

### Results
Reached 100% accuracy in 4k training steps

## Sentence sentences exemple
File: sentences.py  
The goal of this exemple is to compare distances between sentences's embeddings.

### Grid search hyperparameter optimization
A grid search hyperparameter optimization was done on this example.  
Optimized hyperparameters are:  
- Learning Rate (optimal value: 2e-2)
- Number of hidden lstm units (optimal value: 10)

### Results
Reached 100% accuracy in 50 to 100 training steps.  
The sentence "This cat is cute" is closer to "This cat is eating" (distance: 0.3331) than "My name is Florian" (distance: 1.2888).

# Useful links
[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)
