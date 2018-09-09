# SequenceToSequence
My implementation of Sequence To Sequence model (Seq2sSq) using Tensorflow. This Long Short Term Memory (LSTM) based encoder / decode allow learning semantic representations of variables sequences (like words or sentenses)
## Letter sequences example
The implemented exemple in the main.py file is a word embedding model that take letters as features sequences (26 features as one hot vector).

### Grid search hyperparameter optimization
A grid search hyperparameter optimization was done on this example.  
Optimized hyperparameters are:  
- Learning Rate (optimal value: 2e-2)
- Number of hidden lstm units (optimal value: 150)
- Training batch size (optimal value: 50)

# Useful links
[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)
