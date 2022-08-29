#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn

import torch
class BiRNN(nn.Module):
    def __init__(self):
        super(BiRNN, self).__init__()
        #self.embedding = nn.Embedding(len(vocab), embed_size)
# bidirectional􁦡􀔅True􀜨􀮑􀚩􀝌􀝻􀮗􁈾􁐟􁕪􁗑􁕶
        self.encoder = nn.LSTM(input_size=12,
                                hidden_size=100,
                                num_layers=2,
                                bidirectional=True)
# 􀚡􀦤􀷸􁳵􀾍􀞾􀹋􁕣􀷸􁳵􀾍􁌱􁵌􁡐􁇫􀮾􀖢􀔅􀙂􁬳􀴳􀩶􁬌􀙁
      #  self.decoder = nn.Linear(400, 2)
        self.decoder = nn.Sequential(
            nn.Dropout(0.6),
            #1*20*361(361-13+1)
            nn.Linear(400, 120),#349
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(84, 2)
        )
    def forward(self, inputs):
        b,c,h,l = inputs.shape
        pad = 0
        if h < 12:
            pad = 12 - h
            padding = torch.zeros((b, c, pad, l),device = 'cuda')
            inputs = torch.cat((inputs,padding),2)
        inputs = inputs.permute(0,1,3,2)        
       # print('input shape is', inputs.shape)
        embeddings = inputs.squeeze()
        
      #  print('input squeeze shape is',embeddings.shape)
# from ([batch 30, length 365, vector 12])
        # to ((length 33, batch 5, vector 100])

        embeddings = embeddings.permute(1,0,2)
      #  embeddings = self.embedding(inputs.permute(1, 0))
       # print('len of embedding',len(embeddings))
      #  print('embeddings shape is ',embeddings.shape)
        outputs, _ = self.encoder(embeddings) # output, (h, c)
# 􁬳􁕮􀚡􀦤􀷸􁳵􀾍􀞾􀹋􁕣􀷸􁳵􀾍􁌱􁵌􁡐􁇫􀮾􀖢􀔅􀙂􁬳􀴳􀩶􁬌􀙁􀌶􀨙􁌱􀭵􁇫􀔅
# (􀲢􁰁􀥟􀩜, 4 * 􁵌􁡐􀜔􀘲􀓻􀷄)􀌶

        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        outs = torch.softmax(outs,dim = 1)
     #   print('outs shape is', outs.shape)
        return outs

