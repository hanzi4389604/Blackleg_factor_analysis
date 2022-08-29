#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold,KFold
import torch
import time
import xlrd
from torch import nn, optim
import collections
import sys
import csv
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd

import random
from dba import calculate_dist_matrix
from knn import get_neighbors

import sys
sys.path.append("/home/l/20211218 practice/aaltd18-master1/creat_ALL_data_set")
import numpy
import Cython


import sklearn
import scipy
import utils
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
get_ipython().run_line_magic('matplotlib', '')
from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import MAX_PROTOTYPES_PER_CLASS
from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES

from utils.utils import read_all_datasets
from utils.utils import calculate_metrics
from utils.utils import transform_labels
from utils.utils import create_directory
from utils.utils import plot_pairwise
from dba import dba

#from augment import augment_train_set

import numpy as np

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm
device1 = torch.device('cpu')
device = torch.device('cuda')
print(torch.__version__)
print(device)


# In[2]:


filename_his = "/home/l/20211218 practice/svm/blackleg_3classes.csv"
print('The path of histone_data',filename_his)
with open(filename_his) as fi_his:
    csv_reader=csv.reader(fi_his)
    data=list(csv_reader)
    ncols=(len(data[0]))
fi_his.close()

nrows = (len(data)-1)
ncols = (len(data[1])-2)
#print(nrows)
#print(ncols)
y_kfold=[]
x_kfold = np.empty([nrows,ncols], dtype = float) 
#print(x_kfold.shape)
for i in range (0, nrows):
    if data[i+1][1] == 'A':
        y_kfold.append(0)
    elif data[i+1][1] == 'B':
        y_kfold.append(1)
    elif data[i+1][1] == 'C':
        y_kfold.append(1)
    for j in range (0, ncols): 
  #      print(data[i+1][j+2])
        x_kfold[i][j] = data[i+1][j+2]
#    print(i)
#    print(data[i+1][1]) 
#    print(y[i])
print(y_kfold)    
len(y_kfold)
print(x_kfold.shape)
print(x_kfold[62][1829])
#nrows=len(data)
#ngenes=nrows/windows
#nfeatures=ncols-1
#print("Number of genes: %d" % ngenes)
#print("Number of entries: %d" % nrows)
#print("Number of HMs: %d" % nfeatures)


# In[3]:


#
def loadData(windows,file1,file2):
  #  print('Data Reading>>>>>>>>>>>>>>>*****')
    filename_his = file1
 #   print('The path of histone_data',filename_his)
    with open(filename_his) as fi_his:
        csv_reader=csv.reader(fi_his)
        data=list(csv_reader)
        ncols=(len(data[0]))
    fi_his.close()

    nrows=len(data)
    ngenes=nrows/windows
    nfeatures=ncols-1
 #   print("Number of entries: %d" % ngenes)
 #   print("Number of rows: %d" % nrows)
 #   print("Number of features: %d" % nfeatures)

    filename_gene = file2
 #   print('The path of gene_expression',filename_gene)
    with open(filename_gene) as fi_gene:
        csv_reader=csv.reader(fi_gene)
        data_gene=list(csv_reader)
        ncols=(len(data_gene[0]))
    fi_gene.close()

    
    count=0
    
    attr=collections.OrderedDict()

    
    for i in range(0,nrows-1,windows):
        
        hm1=torch.zeros(windows,1)
        hm2=torch.zeros(windows,1)
        hm3=torch.zeros(windows,1)
        hm4=torch.zeros(windows,1)
        hm5=torch.zeros(windows,1)
        hm6=torch.zeros(windows,1)
        hm7=torch.zeros(windows,1)
        hm8=torch.zeros(windows,1)
        hm9=torch.zeros(windows,1)
        hm10=torch.zeros(windows,1)
        hm11=torch.zeros(windows,1)
        hm12=torch.zeros(windows,1)
        for w in range(0,windows):
            hm1[w][0]=float(data[i+1+w][1])
            
            hm2[w][0]=float(data[i+1+w][2])
            
            hm3[w][0]=float(data[i+1+w][3])
            hm4[w][0]=float(data[i+1+w][4])
            hm5[w][0]=float(data[i+1+w][5])
            hm6[w][0]=float(data[i+1+w][6])
            hm7[w][0]=float(data[i+1+w][7])
            hm8[w][0]=float(data[i+1+w][8])
            hm9[w][0]=float(data[i+1+w][9])
            hm10[w][0]=float(data[i+1+w][10])
            hm11[w][0]=float(data[i+1+w][11])
            hm12[w][0]=float(data[i+1+w][12])
            
        geneID=str(data[i+1][0])
        
        thresholded_expr = int(data_gene[count+1][1])
        
       # thresholded_expr = int(data_gene[int(i/100)+1][1])
        attr[count]={
            'entryID':geneID,
            'label':thresholded_expr,
            'wea1':hm1,
            'wea2':hm2,
            'wea3':hm3,
            'wea4':hm4,
            'wea5':hm5,
            'wea6':hm6,
            'wea7':hm7,
            'wea8':hm8,
            'wea9':hm9,
            'wea10':hm10,
            'wea11':hm11,
            'wea12':hm12
        }
        count+=1
        
    return attr


# In[4]:


#
def loadData2(windows,file1,file2):
  #  print('Data Reading>>>>>>>>>>>>>>>*')
    filename_his = file1
    print('The path of histone_data',filename_his)
    with open(filename_his) as fi_his:
        csv_reader=csv.reader(fi_his)
        data=list(csv_reader)
        ncols=(len(data[0]))
    fi_his.close()

    nrows=len(data)
    ngenes=nrows/windows
    nfeatures=ncols-1
 #   print("Number of entries: %d" % ngenes)
 #   print("Number of rows: %d" % nrows)
 #   print("Number of features: %d" % nfeatures)

    filename_gene = file2
 #   print('The path of gene_expression',filename_gene)
    with open(filename_gene) as fi_gene:
        csv_reader=csv.reader(fi_gene)
        data_gene=list(csv_reader)
        ncols=(len(data_gene[0]))
    fi_gene.close()

    
    count=0
    
    attr=collections.OrderedDict()

    
    for i in range(0,nrows-1,windows):
        
        hm1=torch.zeros(windows,1)
        hm2=torch.zeros(windows,1)
        hm3=torch.zeros(windows,1)
        hm4=torch.zeros(windows,1)
        hm5=torch.zeros(windows,1)
        hm6=torch.zeros(windows,1)
        hm7=torch.zeros(windows,1)
        hm8=torch.zeros(windows,1)
        hm9=torch.zeros(windows,1)
        hm10=torch.zeros(windows,1)
        hm11=torch.zeros(windows,1)
        hm12=torch.zeros(windows,1)
        for w in range(0,windows):
            hm1[w][0]=float(data[i+1+w][1])
            
            hm2[w][0]=float(data[i+1+w][2])
            
            hm3[w][0]=float(data[i+1+w][3])
            hm4[w][0]=float(data[i+1+w][4])
            hm5[w][0]=float(data[i+1+w][5])
            hm6[w][0]=float(data[i+1+w][6])
            hm7[w][0]=float(data[i+1+w][7])
            hm8[w][0]=float(data[i+1+w][8])
            hm9[w][0]=float(data[i+1+w][9])
            hm10[w][0]=float(data[i+1+w][10])
            hm11[w][0]=float(data[i+1+w][11])
            hm12[w][0]=float(data[i+1+w][12])
            
        geneID=str(data[i+1][0])
        
        thresholded_expr = int(data_gene[count+1][1])
        
       # thresholded_expr = int(data_gene[int(i/100)+1][1])
        attr[count]={
            'entryID':geneID,
            'label':thresholded_expr,
            'wea1':hm1,
            'wea2':hm2,
            'wea3':hm3,
            'wea4':hm4,
            'wea5':hm5,
            'wea6':hm6,
            'wea7':hm7,
            'wea8':hm8,
            'wea9':hm9,
            'wea10':hm10,
            'wea11':hm11,
            'wea12':hm12
        }
        count+=1
        
    return attr


# In[5]:


#
class HMData2(Dataset):
    def __init__(self,dataset,transform=None):
        self.c1=dataset
    def __len__(self):
        return len(self.c1)
    def __getitem__(self,i):
        final_data_c1=torch.cat((self.c1[i]['wea1'],self.c1[i]['wea2'],
                                 self.c1[i]['wea3'],self.c1[i]['wea4'],
                                 self.c1[i]['wea5'],self.c1[i]['wea6'],
                                 self.c1[i]['wea7'],self.c1[i]['wea8'],
                                 self.c1[i]['wea9'],self.c1[i]['wea10'],
                                
                                 self.c1[i]['wea11'],self.c1[i]['wea12']
                                 ),1)
        
        entryID=self.c1[i]['entryID']
   #     print('+1+1+1+1+1+1+1+1',entryID)
        label=self.c1[i]['label']
     #   print(self.c1[i]['wea1'],self.c1[i]['wea2'],
      #                           self.c1[i]['wea3'],self.c1[i]['wea4'],self.c1[i]['wea5'])            
        final_data_c1=torch.cat((self.c1[i]['wea1'],self.c1[i]['wea2'],
                                 self.c1[i]['wea3'],self.c1[i]['wea4'],
                                 self.c1[i]['wea5'],self.c1[i]['wea6'],
                                 self.c1[i]['wea7'],self.c1[i]['wea8'],
                                 self.c1[i]['wea9'],self.c1[i]['wea10'],
                                 self.c1[i]['wea11'],self.c1[i]['wea12']
                                 ),1)
        
        
  #      print('1st shape')
    #    print(final_data_c1.shape)
        final_data_c1 = final_data_c1.reshape(1, final_data_c1.shape[0], final_data_c1.shape[1])
     #   print('+1+1+1+1+1+1+1+1',entryID)
     #   print('+1+1+1+1+1+1+1+1',final_data_c1)
     #   print('+1+1+1+1+1+1+1+1',label)
   #     print('2nd shape')
  #      print(final_data_c1.shape)
        sample={'entryID':entryID,
               'input':final_data_c1,
               'label':label
               }

        return sample


# In[6]:


#
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

    

#print(net)
def evaluate_accuracy(data_iter, net, w,device=None):
    net = net.to(device)
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for XX in data_iter:
            X= XX['input'].to(device)
            y = XX['label'].to(device)
            Z = XX['entryID']
           # print('input 第一行 is', X[w][0][0])
           # print('ID is\n',Z[w])
           # print('y value is\n',y[w])
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
        #        print('y_hat is\n',net(X.to(device)).argmax(dim=1))
        #        print('y value is\n',y)
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
            y_pre = net(X)
            y_pre = y_pre.argmax(dim=1).cpu()
            y = y.cpu()
     #       print(y_pre)
     #       print(y)
            y_p_np=y_pre.numpy()
            
            y_np=y.numpy()
            f1_test = f1_score(y_np,y_p_np,average = 'weighted')
      #  print('f1 is ',f1_test)
#        f= open(result_file,'a')
#        f.write(str(acc_sum/n))
#        f.write(',')
#        f.write('%')
#        f.write(',')
#        f.write(str(f1_test))
#        f.write(',')
#        f.close()
        f1_score_orig.append(f1_test)
        acc_orig.append(acc_sum/n)

#    print('test_acc is and n is', acc_sum, n)
    return acc_sum / n

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs,w):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    best_loss = float('inf')    

    for epoch in range(num_epochs):
        
        batch_count = 0
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for XX in train_iter: 
            
    #        print(XX['input'].type())
            X = XX['input'].to(device)
  #          print(X.shape)
     #       print(XX['label'].type()) 
            y = XX['label'].to(device) 
            z = XX['entryID']
            X= X
           # print(X.shape)
            y_hat = net(X)  
        #    print('\n\n ID is',z)
        #    print('\n \n \n y is', y)
       #     print('y_hat is', y_hat, '/n /n')
            if epoch >(num_epochs - 1):
                ans = []
                for t in y_hat:
                    if t[0]>t[1]:
                        ans.append(0)
                    else:ans.append(1)
            l = loss(y_hat, y)
         #   print('loss is',l)
           # if l < best_loss:
           #     best_loss = l
           #     torch.save(net, 'best_model.pth') 
                
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
 #           print('train_l  ++ train_acc',train_l_sum, train_acc_sum)            
#            if epoch >80:
 #               print('yhat', y_hat.argmax(dim=1))
   #              print('y is',y)
   #             print('geneID is',XX['entryID'])
   #             print('train_acc_sum is ',train_acc_sum)
   #             print('input matrix is',XX['input'])
            n += y.shape[0]
            batch_count += 1
            
        
        #print('epoch is', epoch,'num_epoch is', num_epochs-5)
        if epoch > (num_epochs-50):
            #f= open(result_file,'a')

            #f.write(str(epoch))
            #f.write(',')
            #f.close()

            test_acc = evaluate_accuracy(test_iter, net,w)
             #   print('test_acc is', test_acc)
            Loss.append(train_l_sum / batch_count)
     #       print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ epoch', epoch)
     #       print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
     #                 % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, 
     #                    test_acc, time.time() - start))


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
            Reshape(1,20,361),  
            nn.MaxPool2d((1,13), (1,1)), 
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            #1*20*361(361-13+1)
            nn.Linear(1*20*349, 120),
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
       # print(img.shape)
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        output = torch.softmax(output,dim = 1)
        return output
    


# In[1]:


#
def load_svm(file1, file2):
    filename_his = file1 + '.csv'
    print('The path of histone_data',filename_his)
    with open(filename_his) as fi_his:
        csv_reader=csv.reader(fi_his)
        data=list(csv_reader)
        ncols=(len(data[0]))
    fi_his.close()
    nrows = int((len(data)-1)/365)
    ncols = int(((len(data[1])-2))*365)
    y=[]
    x = np.empty([nrows,ncols], dtype = float)
    #print(x.shape)
    for i in range (0, nrows):
        index_x = 0
        for m in range(0,365):
            for k in range(0,12):
                x[i][m*12+k] = data[i*365+m+1][k+1]
    #            if k*365+m >1822:
    #                print(k*365+m, k+1,m+1)
    #                print(data[i*365+m+1][k+1])
    #                print(x[i][k*365+m])
                index_x = k*365+m
        #print(index_x)        
#        for l in range(0,7):
      #      print(i)
#            x[i][index_x+1+l] = data[i*365+1][l+6] 
    x = x[:,1164:2964]
    print(x.shape)
    #print(x[257][1820:])
    #print(x[200][1000])
    test_num = (x[0])
    #for xx in range(1832):
    #    print(x[0][xx])
    #    print(xx)
    #print

   # print(len(test_num))
    filename_his = file2  + '.csv'
    #print('The path of histone_data',filename_his)
    with open(filename_his) as fi_his:
        csv_reader=csv.reader(fi_his)
        data=list(csv_reader)
        ncols=(len(data[0]))
    fi_his.close()
    y = []
    for i in range(0, nrows):
      #  print(i)
        y.append(data[i+1][1])
    #print(y)
    file_name_x =  file1 + '_PCA_12_variable.csv'
    print(file_name_x)
    np.savetxt(file_name_x, x, fmt='%s',delimiter=',')
    file_name_y =  file2 + '_PCA_12_variable.csv'
    
    np.savetxt(file_name_y, y, fmt='%s')
    
    
    return x,y


# In[2]:



ss = StandardScaler()
fold_num = [3,4,5]
for i in range(0,len(fold_num)):
    num_fold = fold_num[i]
    kf = StratifiedKFold(n_splits=num_fold,shuffle=True)
    for k,(train_kfold,test_kfold) in enumerate(kf.split(x_kfold,y_kfold)):
        num_syn = [50,100,150,200,300]

        for n in range(0, len(num_syn)):
            num_synth = num_syn[n]
            print('the num of syn is', num_synth)

    # read synthesized and validation data from .csv files        
            folder = "/home/l/20211218 practice/aaltd18-master1/creat_ALL_data_set/1_CNN_data_prepare_1000_problem_solved_PCA_2_class/"

            file_name_x = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_x_orig'
            file_name_y = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_y_orig'
            x_file = folder+file_name_x
            y_file  = folder+file_name_y
            print('reading file',x_file)
            
            x_svm_orig,y_svm_orig = load_svm(x_file,y_file)
        
            file_name_x = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_x_syn'               
            file_name_y = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_y_syn'
            x_file = folder+file_name_x
            y_file  = folder+file_name_y
            print('reading file',x_file)
            
            x_svm_syn,y_svm_syn = load_svm(x_file,y_file)

            

            
##orig          
            
           
f.close()

