#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout_rate, 
                 pad_index):
        super().__init__()
 #       self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, 
                                              n_filters, 
                                              filter_size) 
                                    for filter_size in filter_sizes])
        self.fc = nn.Sequential(

            nn.Linear(300, 120),#349
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(84, 2)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, input):
        b,c,h,l = input.shape
        pad = 0
        if h < 12:
            pad = 12 - h
            padding = torch.zeros((b, c, pad, l),device = 'cuda')
            input = torch.cat((input,padding),2)
        # ids = [batch size, seq len]
       # embedded = self.dropout(self.embedding(ids))
        # embedded = [batch size, seq len, embedding dim]
       # embedded = embedded.permute(0,2,1)
        # embedded = [batch size, embedding dim, seq len]
        #input = input.permute(0,1,3,2)
        input = input.squeeze(1)
        conved = [torch.relu(conv(input)) for conv in self.convs]
        
#        print(len(conved))
 #       for i in range(len(conved)):
 #           print(conved[i].shape)
        # conved_n = [batch size, n filters, seq len - filter_sizes[n] + 1]
        pooled = [conv.max(dim=-1).values for conv in conved]
        # pooled_n = [batch size, n filters]
        cat = self.dropout(torch.cat(pooled, dim=-1))
        # cat = [batch size, n filters * len(filter_sizes)]
        prediction = self.fc(cat)
        # prediction = [batch size, output dim]
        return prediction

