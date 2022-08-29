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
        self.decoder = nn.Linear(400, 2)
    def forward(self, inputs):
        b,c,h,l = inputs.shape
        pad = 0
        if h < 12:
            pad = 12 - h
            padding = torch.zeros((b, c, pad, l),device = 'cuda')
            inputs = torch.cat((inputs,padding),2)
        inputs = inputs.permute(0,1,3,2)
      #  print('input shape is', inputs.shape)
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
       # print('outs shape is', outs.shape)
        return outs

