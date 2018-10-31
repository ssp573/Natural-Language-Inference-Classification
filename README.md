# Natural-Language-Inference-Classification

Use the RNN.py file to train biGRU encoder based classifier.

Use the CNN.py file to train the CNN encoder based classifier.

The model which performs the best on validation set is can be downloaded from here: https://drive.google.com/file/d/1I1P-7bWsDRNBUCNpUpJHssSgZ88DRFhY/view?usp=sharing (hidden size: 300 and concatenation)

For both the codes mentioned above, a model state dictionary file is saved after every epoch which can later be used in evaluate.py or 3_examples.py

evaluate.py takes two arguments: name of the model state file saved during training and which encoder to use ('RNN' or 'CNN'). It evaluates accuracy on various genres in the MNLI dataset using the saved model

3_examples.py generates 3 correctly classified and 3 incorrectly classified examples from the SNLI validation dataset. It takes one argument and that is which saved model to use.

