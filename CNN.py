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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(134)


#if not torch.cuda.is_available():
#    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_IDX = 0
UNK_IDX = 1
BATCH_SIZE = 32

def load_data(filename):
    with open('data/'+filename,'r') as f:
        f_list=list(csv.reader(f,delimiter="\t"))[1:]
    return f_list

dict_labels={'neutral':0,'entailment':1,'contradiction':2}
'''def build_vocab(data):
    # Returns:
    # id2char: list of chars, where id2char[i] returns char that corresponds to char i
    # char2id: dictionary where keys represent chars and corresponding values represent indices
    # some preprocessing
    max_len1 = max([len(row[0]) for row in data])
    max_len2 = max([len(row[1]) for row in data])
    all_words= []
    for sample in data:
        all_words += sample[0]+sample[1]
    unique_words = list(set(all_words))

    id2token = unique_words
    token2id = dict(zip(unique_words, range(2,2+len(unique_words))))
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX
    token2id['<unk>'] = UNK_IDX

    return token2id, id2token, max_len1,max_len2'''

def build_vocab(data):
    # Returns:
    # id2char: list of chars, where id2char[i] returns char that corresponds to char i
    # char2id: dictionary where keys represent chars and corresponding values represent indices
    # some preprocessing
    words_to_load=100000
    PAD_IDX=words_to_load
    UNK_IDX=words_to_load+1
    max_len1 = max([len(sent[0]) for sent in data])
    max_len2 = max([len(sent[1]) for sent in data])
    with open('wiki-news-300d-1M.vec') as f:
        loaded_embeddings_ft = np.zeros((words_to_load+1, 300))
        token2id = {}
        id2token = {}
        ordered_words_ft = []
        for i, line in enumerate(f):
            if i >= words_to_load: 
                break
            s = line.split()
            loaded_embeddings_ft[i, :] = np.asarray(s[1:])
            token2id[s[0]] = i
            id2token[i] = s[0]
            ordered_words_ft.append(s[0])
        loaded_embeddings_ft[PAD_IDX,:] = np.zeros(300)
        id2token[PAD_IDX] = '<pad>'
        token2id['<pad>'] = PAD_IDX
    """
    all_words = []
    for sent in data:
        all_words += sent[0]
        all_words += sent[1]
        
    unique_words = list(set(all_words))

    id2word = unique_words
    word2id = dict(zip(unique_words, range(2,2+len(unique_words))))
    id2word = ['<pad>', '<unk>'] + id2word
    word2id['<pad>'] = PAD_IDX
    word2id['<unk>'] = UNK_IDX
    """
    return token2id, id2token, max_len1, max_len2, loaded_embeddings_ft

def convert_to_tokens(data):
    return [(sample[0].split(" "),sample[1].split(" "), dict_labels[sample[2]]) for sample in data]

### Function that preprocessed dataset
'''def read_data():
    #data = pkl.load(open("data.p", "rb"))
    #print(data)
    train_data, val_data, test_data = load_data('snli_train.tsv'), load_data('snli_val.tsv'), load_data('mnli_val.tsv')
    train_data, val_data, test_data = convert_to_tokens(train_data), convert_to_tokens(val_data), convert_to_tokens(test_data)
    #print(train_data)
    token2id, id2token, max_len1, max_len2 = build_vocab(train_data)
    return train_data, val_data, test_data, token2id, id2token, max_len1, max_len2'''

def read_data():
    #data = pkl.load(open("data.p", "rb"))
    #print(data)
    train_data, val_data, test_data = load_data('snli_train.tsv'), load_data('snli_val.tsv'), load_data('mnli_val.tsv')
    train_data, val_data, test_data = convert_to_tokens(train_data), convert_to_tokens(val_data), convert_to_tokens(test_data)
    #print(train_data)
    token2id, id2token, max_len1, max_len2, pretrained_vecs = build_vocab(train_data)
    return train_data, val_data, test_data, token2id, id2token, max_len1, max_len2,pretrained_vecs


train_data, val_data, test_data, token2id, id2token, MAX_SENTENCE_LENGTH_1, MAX_SENTENCE_LENGTH_2, pretrained_vecs = read_data()

print ("Maximum sentence 1 length of dataset is {}".format(MAX_SENTENCE_LENGTH_1))
print ("Maximum sentence 2 length of dataset is {}".format(MAX_SENTENCE_LENGTH_2))
print ("Number of words in dataset is {}".format(len(id2token)))
#print ("Characters:")
#print (token2id.keys())

class VocabDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_tuple, token2id):
        """
        @param data_list: list of character
        @param target_list: list of targets
        """
        #print(data_tuple)
        self.sent1_list, self.sent2_list, self.target_list = zip(*data_tuple)
        #print(self.sent1_list, self.sent2_list,self.target_list)
        assert (len(self.sent1_list) == len(self.sent2_list) == len(self.target_list))
        self.token2id = token2id

    def __len__(self):
        return len(self.sent1_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        sent1_token_idx = [self.token2id[w] if w in self.token2id.keys() else UNK_IDX  for w in self.sent1_list[key][:MAX_SENTENCE_LENGTH_1]]
        sent2_token_idx = [self.token2id[w] if w in self.token2id.keys() else UNK_IDX  for w in self.sent2_list[key][:MAX_SENTENCE_LENGTH_2]]
        label = self.target_list[key]
        return [sent1_token_idx, sent2_token_idx, len(sent1_token_idx), len(sent2_token_idx), label]
    
def vocab_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    sent1_list = []
    sent2_list = []
    label_list = []
    sent1_length_list = []
    sent2_length_list=[]
    #print()
    for datum in batch:
        label_list.append(datum[4])
        sent1_length_list.append(datum[2])
        sent2_length_list.append(datum[3])
    #print(label_list)

    # padding
    for datum in batch:
        padded_vec_sent1 = np.pad(np.array(datum[0]),
                                pad_width=((0,MAX_SENTENCE_LENGTH_1-datum[2])),
                                mode="constant", constant_values=0)
        sent1_list.append(padded_vec_sent1)
        padded_vec_sent2 = np.pad(np.array(datum[1]),
                                pad_width=((0,MAX_SENTENCE_LENGTH_2-datum[3])),
                                mode="constant", constant_values=0)
        sent2_list.append(padded_vec_sent2)

    _, idx_sort_1 = torch.sort(torch.tensor(sent1_length_list), dim=0, descending=True)
    _, idx_unsort_1 = torch.sort(idx_sort_1, dim=0)
    
    sent1_length_list = list(torch.tensor(sent1_length_list)[idx_sort_1])
    idx_sort_1 = Variable(idx_sort_1)
    sent1_list = torch.tensor(sent1_list).index_select(0,idx_sort_1)


    _, idx_sort_2 = torch.sort(torch.tensor(sent2_length_list), dim=0, descending=True)
    _, idx_unsort_2 = torch.sort(idx_sort_2, dim=0)
    
    sent2_length_list = list(torch.tensor(sent2_length_list)[idx_sort_2])
    idx_sort_2 = Variable(idx_sort_2)
    #print(len(sent2_list))
    #print(idx_sort_2.size())
    sent2_list = torch.tensor(sent2_list).index_select(0,idx_sort_2)
    
    #print(sent1_list)
    #print(sent2_list)
    #print(label_list)
    
    return [sent1_list, torch.LongTensor(sent1_length_list), idx_unsort_1, sent2_list, torch.LongTensor(sent2_length_list), idx_unsort_2, torch.LongTensor(label_list)]

train_dataset = VocabDataset(train_data, token2id)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=True,pin_memory=True)

val_dataset = VocabDataset(val_data, token2id)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=True,pin_memory=True)

test_dataset = VocabDataset(test_data, token2id)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=False,pin_memory=True)


class CNN(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, num_classes, vocab_size):

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

num_classes=3
EMBED_SIZE=300
Encoder_Hidden=300

class Classifier(nn.Module):
    def __init__(self, fc_hidden_size):
        super(Classifier, self).__init__()
        self.hidden_size=fc_hidden_size
        self.linear1=nn.Linear(2*Encoder_Hidden, self.hidden_size)
        self.linear2=nn.Linear(self.hidden_size, num_classes)
        self.CNN=CNN(emb_size=EMBED_SIZE, hidden_size=Encoder_Hidden, num_layers=1, num_classes=3, vocab_size=len(id2token))
        
    def forward(self, sent1, sent2, length1, length2, idx_unsort_1, idx_unsort_2):
        sent1_tensor = self.CNN(sent1,length1)
        #print(sent1_tensor.shape)
        #print(idx_unsort_1.shape)
        sent1_tensor = sent1_tensor.index_select(0, idx_unsort_1)
        sent2_tensor = self.CNN(sent2,length2)
        sent2_tensor = sent2_tensor.index_select(0, idx_unsort_2)
        combined= torch.cat([sent1_tensor, sent2_tensor], dim=1)
        #print(combined.shape)
        hidden=F.relu(self.linear1(combined))
        return self.linear2(hidden)


def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for sent1, length1, unsort1, sent2, length2, unsort2, labels in loader:
            if torch.cuda.is_available():
                sent1_batch, length1_batch, unsort1_batch, sent2_batch, length2_batch, unsort2_batch, label_batch = sent1.cuda(), length1.cuda(), unsort1.cuda(), sent2.cuda(), length2.cuda(), unsort2.cuda(), labels.cuda()
            else:
                sent1_batch, length1_batch, unsort1_batch, sent2_batch, length2_batch, unsort2_batch, label_batch = sent1, length1, unsort1, sent2, length2, unsort2, labels

            outputs = F.softmax(model(sent1_batch, sent2_batch, length1_batch, length2_batch, unsort1_batch, unsort2_batch), dim=1)
            loss = criterion(outputs, label_batch)
            predicted = outputs.max(1, keepdim=True)[1]

            total += labels.size(0)
            correct += predicted.eq(label_batch.view_as(predicted)).sum().item()
    return (100 * correct / total),loss


model = Classifier(fc_hidden_size=1024)
model.to(device)

learning_rate = 3e-4
num_epochs = 10 # number epoch to train

# Criterion and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
total_train=0
correct_train=0

train_accs=[]
train_losses=[]
val_accs=[]
val_losses=[]
for epoch in range(num_epochs):
    for i, (sent1, length1, unsort1, sent2, length2, unsort2, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            sent1, length1, unsort1, sent2, length2, unsort2, labels=sent1.cuda(), length1.cuda(), unsort1.cuda(), sent2.cuda(), length2.cuda(), unsort2.cuda(), labels.cuda()
        model.train()
        optimizer.zero_grad()
        # Forward pass
        outputs = model(sent1, sent2, length1, length2, unsort1, unsort2)
        
        #print(outputs.shape)
        #print(labels.shape)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        # validate every 100 iterations
        if i > 0 and i % 100 == 0:
            # validate
            #train_loss_epoch.append(loss)
            #train_accuracy_epoch.append(train_acc)
            train_acc,val_acc= test_model(train_loader, model)
            val_acc,val_loss = test_model(val_loader, model)
            
            #val_loss_epoch.append(val_loss)
            print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}, Train Accuracy: {}'.format(
                       epoch+1, num_epochs, i+1, len(train_loader), val_acc, train_acc))
    
    train_acc,train_loss= test_model(train_loader, model)
    val_acc,val_loss = test_model(val_loader, model)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    model_file = 'model_concat_hidden_300_' + str(epoch+1) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file)


epochs=[i for i in range(1,num_epochs+1)]

plt.subplot(223)
plt.plot(epochs,train_accs, label="training accuracy")
plt.plot(epochs, val_accs, label="validation accuracy")
plt.ylabel("accuracy")
plt.xlabel("number of epochs")
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.savefig('accuracies_concat_CNN_hidden_300.png')

plt.clf()
plt.subplot(223)
plt.plot(epochs,train_losses, label="training loss")
plt.plot(epochs, val_losses, label="validation loss")
plt.ylabel("loss")
plt.xlabel("number of epochs")
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.savefig('losses_concat_CNN_hidden_300.png')
