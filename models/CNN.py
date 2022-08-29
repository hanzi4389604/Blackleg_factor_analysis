import torch.nn as nn
import torch

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        print('test+++++++++++++')
        self.conv = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 20, (5,12)), 
            ##shape from 365*5  to 20*361
            nn.BatchNorm2d(20),
            nn.ReLU(),
            Reshape(1,20,146),  #361
            nn.MaxPool2d((1,13), (1,1)), 
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            #1*20*361(361-13+1)
            nn.Linear(1*20*134, 120),#349
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(84, 2)
        )
        

    def forward(self, img):
        b,c,h,l = img.shape
        pad = 0
        if h < 12:
            pad = 12 - h
            padding = torch.zeros((b, c, pad, l),device = 'cuda')
            img = torch.cat((img,padding),2)
        
        img = img.permute(0,1,3,2)
       # print(img.shape)
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        output = torch.softmax(output,dim = 1)
        return output
    

