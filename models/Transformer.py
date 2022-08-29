#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
import torch
import copy
import math
from torch.autograd import Variable
import torch.nn.functional as F
import random

class InputEmbeddings(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(InputEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.embedding_dim)
class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_dim)

        position = torch.arange(0., max_len).unsqueeze(1)      #[max_len, 1]
        div_term = torch.exp(torch.arange(0., embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)              #内存中定一个常量，模型保存和加载的时候可以写入和读出

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)  #Embedding+PositionalEncoding
        return self.dropout(x)
    
    
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):   #q,k,v:[batch, h, seq_len, d_k]

    d_k = query.size(-1)                                                    #query的维度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)    #打分机制 [batch, h, seq_len, seq_len]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)       #mask==0的内容填充-1e9，使计算softmax时概率接近0
    p_attn = F.softmax(scores, dim = -1)                   #对最后一个维度归一化得分 [batch, h, seq_len, seq_len]

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn             #[batch, h, seq_len, d_k]


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, embedding_dim, dropout=0.1):

        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % h == 0

        self.d_k = embedding_dim // h   #将embedding_dim分割成h份后的维度
        self.h = h                      #h指的是head数量
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):      #q,k,v:[batch, seq_len, embedding_dim]

        if mask is not None:
            mask = mask.unsqueeze(1)                      #[batch, seq_len, 1]
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from embedding_dim => h x d_k
        # [batch, seq_len, h, d_k] -> [batch, h, seq_len, d_k]
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)   #x:[batch, h, seq_len, d_k], attn:[batch, h, seq_len, seq_len]

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)       #[batch, seq_len, embedding_dim]
        return self.linears[-1](x)
    
class MyTransformerModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, p_drop, h, output_size):
        super(MyTransformerModel,self).__init__()
        self.drop = nn.Dropout(p_drop)

        self.embeddings = InputEmbeddings(vocab_size, embedding_dim)
        self.position = PositionalEncoding(embedding_dim, p_drop)
        self.attn = MultiHeadedAttention(h, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.linear = nn.Linear(embedding_dim, 2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self,inputs,mask=None): #维度均为[batch, seq_len]
        b,c,h,l = inputs.shape
        pad = 0
        if h < 12:
            pad = 12 - h
            padding = torch.zeros((b, c, pad, l),device = 'cuda')
            inputs = torch.cat((inputs,padding),2)
        inputs = inputs.permute(0,1,3,2)          
       # embeded = self.embeddings(inputs)        #1.InputEmbedding [batch, seq_len, embedding_dim]
    #    print('input shape is', inputs.shape)
        inputs = inputs.squeeze()
     #   print('input shape is', inputs.shape)
        embeded = self.position(inputs)         #2.PostionalEncoding [batch, seq_len, embedding_dim]
     #   print('embeded position shape is', embeded.shape)
       # mask = mask.unsqueeze(2)                 #[batch,seq_len,1]

        inp_attn = self.attn(embeded, embeded, embeded, mask)    #3.1MultiHeadedAttention [batch, seq_len, embedding_dim]
      #  print('inp_atten shape is', embeded.shape)
        inp_attn = self.norm(inp_attn + embeded)                 #3.2LayerNorm
      #  print('after norm attn shape is', embeded.shape)
      #  inp_attn = inp_attn * mask                               #4. linear [batch, seq_len, embedding_dim]

        h_avg = inp_attn.sum(1)/(150 + 1e-5)             #[batch, embedding_dim]
      #  print('h_avg shape is', h_avg.shape)
      #  print('the  shpape is ', self.linear(h_avg).squeeze().shape)
      #  print('the transformer type is ', self.linear(h_avg).squeeze().type())
        y = self.linear(h_avg)
        
        y = torch.softmax(y,dim = 1)
      #  y = y.type(torch.cuda.LongTensor)
     #   print(y)
      #  print('the after linear output type is ', y.shape)
        return y                      #[batch, 1] -> [batch]

