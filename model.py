import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import random
import pdb
import csv
from torch.autograd import Variable

PAD_IDX = 0
UNK_IDX = 1
BATCH_SIZE = 32

num_classes=3
EMBED_SIZE=300
Encoder_Hidden=300
fc_hidden_size=1024

class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, num_classes, vocab_size, pretrained_vecs):
        # RNN Accepts the following hyperparams:
        # emb_size: Embedding Size
        # hidden_size: Hidden Size of layer in RNN
        # num_layers: number of layers in RNN
        # num_classes: number of output classes
        # vocab_size: vocabulary size
        super(RNN, self).__init__()
        pretrained_vecs_tensor=torch.from_numpy(pretrained_vecs).float()
        if torch.cuda.is_available():
            pretrained_vecs_tensor=pretrained_vecs_tensor.cuda()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.embedding = nn.Embedding.from_pretrained(pretrained_vecs_tensor)
        self.rnn = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=True) #First dimension is the batch size
        #self.linear = nn.Linear(hidden_size, num_classes)

    def init_hidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.randn(self.num_layers*2, batch_size, self.hidden_size)

        return hidden

    def forward(self, x, lengths):
        # reset hidden state

        batch_size, seq_len = x.size()
        if torch.cuda.is_available():
            self.hidden = self.init_hidden(batch_size).cuda()
        else:
            self.hidden = self.init_hidden(batch_size)

        # get embedding of characters
        embed = self.embedding(x)
        # pack padded sequence
        #pytorch wants sequences to be in decreasing order of lengths
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu().numpy(), batch_first=True)
        # fprop though RNN
        rnn_out, self.hidden = self.rnn(embed, self.hidden)
        # undo packing
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        # sum hidden activations of RNN across time
        rnn_out = torch.sum(rnn_out, dim=1)

        #logits = self.linear(rnn_out)
        return self.hidden.transpose(0,1).contiguous().view(batch_size, -1)



class CNN(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, num_classes, vocab_size, pretrained_vecs):

        super(CNN, self).__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        pretrained_vecs_tensor=torch.from_numpy(pretrained_vecs).float()
        if torch.cuda.is_available():
            pretrained_vecs_tensor=pretrained_vecs_tensor.cuda()

        self.embedding = nn.Embedding.from_pretrained(pretrained_vecs_tensor)
    
        self.conv1 = nn.Conv1d(emb_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)

    def forward(self, x, lengths):
        batch_size, seq_len = x.size()

        embed = self.embedding(x)
        hidden = self.conv1(embed.transpose(1,2)).transpose(1,2)
        hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))

        hidden = self.conv2(hidden.transpose(1,2)).transpose(1,2)
        hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))
        
        hidden=torch.max(hidden,dim=1)[0]
        return hidden


class Classifier(nn.Module):
    def __init__(self, Enc,voc_size,pretrained_vecs):
        super(Classifier, self).__init__()
        self.hidden_size=fc_hidden_size
        self.linear1=nn.Linear(4*Encoder_Hidden, self.hidden_size)
        self.linear2=nn.Linear(self.hidden_size, num_classes)
        if Enc=="RNN":
            self.biRNN=RNN(emb_size=EMBED_SIZE, hidden_size=Encoder_Hidden, num_layers=1, num_classes=3, vocab_size=voc_size,pretrained_vecs=pretrained_vecs)
        else:
            self.CNN=CNN(emb_size=EMBED_SIZE, hidden_size=Encoder_Hidden, num_layers=1, num_classes=3, vocab_size=voc_size,pretrained_vecs=pretrained_vecs)
        
    def forward(self, sent1, sent2, length1, length2, idx_unsort_1, idx_unsort_2):
        sent1_tensor = self.biRNN(sent1,length1)
        #print(sent1_tensor.shape)
        #print(idx_unsort_1.shape)
        sent1_tensor = sent1_tensor.index_select(0, idx_unsort_1)
        sent2_tensor = self.biRNN(sent2,length2)
        sent2_tensor = sent2_tensor.index_select(0, idx_unsort_2)

        combined= torch.cat([sent1_tensor, sent2_tensor],dim=1)
        #print(combined.shape)
        hidden=F.relu(self.linear1(combined))
        return self.linear2(hidden)
