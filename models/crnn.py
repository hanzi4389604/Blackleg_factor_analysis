import torch.nn as nn
import torch

class BidirectionalLSTM1(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM1, self).__init__()
        
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.linear = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
            recurrent, _ = self.rnn(input)
            T, b, h = recurrent.size()
     #       print('T,b,h',T,b,h)
            t_rec = recurrent.view(T * b, h)

            output = self.linear(t_rec)  # [T * b, nOut]
            output = output.view(T, b, -1)

            return output
class BidirectionalLSTM2(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM2, self).__init__()
        
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.linear = nn.Linear(nHidden * 4, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        encoding = torch.cat((recurrent[0], recurrent[-1]), -1)
        output = self.linear(encoding)
        return output

class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
       # assert imgH % 12 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))#
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(1, 2))  # 64x16x64   C->64 32 * 128
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(1, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (1, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (1, 1), (0, 1)))  # 512x2x16
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM1(512, nh, nh),
            BidirectionalLSTM2(nh, nh, nclass))

    def forward(self, input):
        b,c,h,l = input.shape
        pad = 0
        if h < 12:
            pad = 12 - h
            padding = torch.zeros((b, c, pad, l),device = 'cuda')
            input = torch.cat((input,padding),2)
        # conv features
        conv = self.cnn(input)
 #       print(conv.shape)
        
        b, c, h, w = conv.size()
#        print(b,c,h,w)
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
#        print(conv.shape)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
  #      print(conv.shape)

        # rnn features
        output = self.rnn(conv)

        return output
fixed_noise = torch.randn(222, 1, 12, 150, device='cpu')
net = CRNN(16,1,2,100) #imgH, nc, nclass, nh,
#print(net)
x = net(fixed_noise)
print(x.shape)

