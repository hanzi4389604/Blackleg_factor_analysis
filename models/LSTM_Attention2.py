import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class BiLSTM_Attention2(nn.Module):

    def __init__(self, vocab_size=150, embedding_dim=12, hidden_dim=128, n_layers=1):

        super(BiLSTM_Attention2, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)        #单词数，嵌入向量维度
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True)
        #self.fc = nn.Linear(hidden_dim * 2, 2)
        self.fc = nn.Sequential(

            nn.Linear(hidden_dim * 2, 120),#349
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(120, 2)
        )
        self.dropout = nn.Dropout(0.9)

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)


    def attention_net(self, x):       #x:[batch, seq_len, hidden_dim*2]

        u = torch.tanh(torch.matmul(x, self.w_omega))         #[batch, seq_len, hidden_dim*2]
        att = torch.matmul(u, self.u_omega)                   #[batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)

        scored_x = x * att_score                              #[batch, seq_len, hidden_dim*2]

        context = torch.sum(scored_x, dim=1)                  #[batch, hidden_dim*2]
        return context


    def forward(self, x):
        b,c,h,l = x.shape
        pad = 0
        if h < 12:
            pad = 12 - h
            padding = torch.zeros((b, c, pad, l),device = 'cuda')
            x = torch.cat((x,padding),2)
            #[batch,channel,length, embedding_dim]
        x = x.permute(0,1,3,2)
        x = x.squeeze(1)#[batch,length, embedding_dim]
       # embedding = self.dropout(x)       #[batch,length, embedding_dim]
       # print('x shjape is ',x.shape)
        embedding = x.permute(1, 0, 2)# #[seq_len, batch, embedding_dim]
        # output: [seq_len, batch, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        output = output.permute(1, 0, 2)                  #[batch, seq_len, hidden_dim*2]


        attn_output = self.attention_net(output)       #和LSTM的不同就在于这一句
        logit = self.fc(attn_output)
        logit = torch.softmax(logit,dim = 1)
        return logit

