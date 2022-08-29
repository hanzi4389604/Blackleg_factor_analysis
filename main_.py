#!/usr/bin/env python
# coding: utf-8

# In[4]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import torch
import time
import xlrd
from torch import nn, optim
import collections
import sys
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
sys.path.append("/home/l/20211218 practice/aaltd18-master1/creat_ALL_data_set")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.axes3d import Axes3D

import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm

device1 = torch.device('cpu')
device = torch.device('cuda')
print(torch.__version__)
print(device)
##open the csv file for svm analysis
filename_his = "/home/l/20211218 practice/svm/blackleg_3classes.csv"
print('The path of histone_data',filename_his)
with open(filename_his) as fi_his:
    csv_reader=csv.reader(fi_his)
    data=list(csv_reader)
    ncols=(len(data[0]))
fi_his.close()

nrows = (len(data)-1)
ncols = (len(data[1])-2)
y_kfold=[]
x_kfold = np.empty([nrows,ncols], dtype = float) 
print(x_kfold.shape)
for i in range (0, nrows):
    if data[i+1][1] == 'A':
        y_kfold.append(0)
    elif data[i+1][1] == 'B':
        y_kfold.append(1)
    elif data[i+1][1] == 'C':
        y_kfold.append(1)
    for j in range (0, ncols): 
        x_kfold[i][j] = data[i+1][j+2]
#print(y_kfold)    
#len(y_kfold)
#print(x_kfold.shape)
#print(x_kfold[62][1829])

import models.CNN as CNN
import models.crnn as crnn
import models.crnn_dropoff as crnn_dropoff
import models.BiRNN as BiRNN
import models.BiRNN_dropoff as BiRNN_dropoff
import models.LSTM_Attention1 as LSTM_Attention1
import models.LSTM_Attention2 as LSTM_Attention2
import models.Transformer as Transformer
import models.TEXTcnn as TEXTcnn

from utils1 import *
fixed_noise = torch.randn(222, 1, 12, 150, device='cpu')
ss = StandardScaler()
#print(net)
result_matrix = np.zeros((3,5,5,10,20))


batch_size = 100   
fold_num = [3,4,5]    
num_syn = [50,100,150,200,300]  
lr, num_epochs = 0.001, 300
filter_sizes = [3,5,7]
pad_index = 0

net1 = CNN.Net()
net2 = TEXTcnn.CNN(30, 12, 100, filter_sizes, 2, 0.6, pad_index)
net3 = BiRNN.BiRNN()
net4 = BiRNN_dropoff.BiRNN()
net5 = Transformer.MyTransformerModel(vocab_size=150, embedding_dim= 12, p_drop=1, h=1, output_size=2)
net6 = crnn.CRNN(16,1,2,100) #imgH, nc, nclass, nh,
net7 = crnn_dropoff.CRNN(16,1,2,100) #imgH, nc, nclass, nh,
net8 = LSTM_Attention1.BiLSTM_Attention1(vocab_size = 150, embedding_dim= 12, hidden_dim= 128, n_layers=2)
net9 = LSTM_Attention2.BiLSTM_Attention2(vocab_size = 150, embedding_dim= 12, hidden_dim= 128, n_layers=2)
net_dict = {'CNN':net1,'TEXTcnn':net2,'BiRNN':net3,'BiRNN_dropoff':net4,'Transformer':net5,
                'crnn':net6,'crnn_dropoff':net7,
                'LSTM_Attention1':net8,'LSTM_Attention2':net9}
    


main_variable = ['weather_only','no_pest','full','no_rotation']

folder = "/home/l/20211218 practice/aaltd18-master1/creat_ALL_data_set/2_CNN_data_prepare_blk_sev_2class/"

# this is for differnet networks
for v_type in enumerate(main_variable):
    v_type=v_type[1]

    
    for key in net_dict.keys():
        net = net_dict[key]
        print(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        # this iteration is for differnet k folds
        for i in range(0,len(fold_num)):
            print('i values is',i)
            num_fold = fold_num[i]
            kf = StratifiedKFold(n_splits=num_fold,shuffle=True)
            for k,(train_kfold,test_kfold) in enumerate(kf.split(x_kfold,y_kfold)):
                     ##############################################
                print('num_fold and num_syn', num_fold, num_syn)
                for n in range(0, len(num_syn)):
                    num_synth = num_syn[n]
                    print('the num of syn is', num_synth)
        # read synthesized and validation data from .csv files        
                    attr_orig, attr_syn, attr_test,x_svm_orig,y_svm_orig, x_svm_syn,y_svm_syn, x_svm_test,y_svm_test = load_data_svm(folder,k,num_fold,num_synth,v_type)
        ##orig          
                    x_svm_orig_ss, x_svm_test_ss,x_svm_syn_ss = ss.fit_transform(x_svm_orig), ss.transform(x_svm_test),ss.transform(x_svm_syn)  ###IMPORTANT
                    model = OneVsRestClassifier(svm.SVC(kernel='linear',probability=True))
                    clt = model.fit(x_svm_orig_ss,y_svm_orig)
                    print("train with orig, validate with Orig：%.2f%%"%(clt.score(x_svm_orig_ss,y_svm_orig)*100))
                    print("train with Orig, validate with Test：%.2f%%"%(clt.score(x_svm_test_ss,y_svm_test)*100))
                    print("train with Orig, validate with Syn：%.2f%%"%(clt.score(x_svm_syn_ss,y_svm_syn)*100))
                    f1_test = f1_score(y_svm_test,clt.predict(x_svm_test_ss),average = 'weighted')
                    svm_orig_acc = clt.score(x_svm_test_ss,y_svm_test)*100
                    result_matrix[i][k][n][0], result_matrix[i][k][n][1] =  svm_orig_acc, f1_test

        #svm   syn
                    x_svm_syn_ss, x_svm_orig_ss = ss.fit_transform(x_svm_syn), ss.transform(x_svm_orig)  ###IMPORTANT
                    model = OneVsRestClassifier(svm.SVC(kernel='linear',probability=True))
                    clt = model.fit(x_svm_syn_ss,y_svm_syn)
                    print("train with syn, validate with syn：%.2f%%"%(clt.score(x_svm_syn_ss,y_svm_syn)*100))
                    print("train with syn, validate with orig：%.2f%%"%(clt.score(x_svm_orig_ss,y_svm_orig)*100))
                    print("train with syn, validate with test：%.2f%%"%(clt.score(x_svm_test_ss,y_svm_test)*100))
                    f1_test = f1_score(y_svm_test,clt.predict(x_svm_test_ss),average = 'weighted')
                    svm_orig_acc = clt.score(x_svm_test_ss,y_svm_test)*100
                    result_matrix[i][k][n][2],result_matrix[i][k][n][3]  =  svm_orig_acc, f1_test
        #CNN
            #dataset splition in to training and testing        
                    full_dataset_syn, full_dataset_test,full_dataset_orig = HMData2(attr_syn,v_type), HMData2(attr_test,v_type), HMData2(attr_orig,v_type)
                    print('CNN full_dataset_syn x shpae is', full_dataset_syn[0]['input'].shape)
                    print('The length of total for training_syn dataset',len(full_dataset_syn))
                    train_size_syn = int(1 * len(full_dataset_syn))
                    test_size_syn = len(full_dataset_syn) - train_size_syn
                    train_dataset_syn, test_dataset_syn = torch.utils.data.random_split(full_dataset_syn, [train_size_syn, test_size_syn])
                    print('CNN full_dataset_test shpae is', full_dataset_test[0]['input'].shape)
                    print('The length of total for test_validation dataset',len(full_dataset_test))
                    train_size_test = int(1 * len(full_dataset_test)) 
                    test_size_test = len(full_dataset_test) - train_size_test
                    train_dataset_test, test_dataset_test = torch.utils.data.random_split(full_dataset_test, [train_size_test, test_size_test])
                    print('The number of train_dataset_test is:',len(train_dataset_test))
                    print('The number of train_dataset_test dataset is:',len(test_dataset_test))
                    print('CNN full_dataset_orig x shpae is', full_dataset_orig[0]['input'].shape)
                    print('The length of total for Orig dataset',len(full_dataset_orig))
                    train_size_orig = int(1 * len(full_dataset_orig))
                    test_size_orig = len(full_dataset_orig) - train_size_orig
                    train_dataset_orig, test_dataset_orig = torch.utils.data.random_split(full_dataset_orig, [train_size_orig, test_size_orig])
                    print('The number of orig dataset is:',len(train_dataset_orig))
                    print('The number of val_orig dataset is:',len(test_dataset_orig))

            #batch size of data_loader #        
                    train_loader_syn = torch.utils.data.DataLoader(dataset=train_dataset_syn,
                                                               batch_size=batch_size, 
                                                               shuffle=True)
                    test_loader_test = torch.utils.data.DataLoader(dataset=train_dataset_test,
                                                               batch_size=batch_size, 
                                                               shuffle=True)

                    train_loader_orig = torch.utils.data.DataLoader(dataset=train_dataset_orig,
                                                               batch_size=batch_size, 
                                                               shuffle=True)
           ########## Orig CNN Training

                    net = net
                    
                    net_name = v_type+'/'+key
                    os.makedirs("weights/%s" % net_name, exist_ok=True)
                    data_type = '_orig'

                    w=5
                    f1_score_orig, Loss, acc_orig = train_ch5(net, train_loader_orig, test_loader_test, 13, optimizer, device, num_epochs,w,net_name,data_type)
                    f1_score_orig, acc_orig,Loss = sorted(f1_score_orig, reverse=True)[:20], sorted(np.array(acc_orig)*100, reverse=True)[:20],sorted(Loss, reverse=False)[:20]
                    print('f1_score_orig length',len(f1_score_orig))
                    print('f1_score_orig','\n',f1_score_orig,'\n','acc_orig','\n',acc_orig,'\n','Loss is',Loss)
                    print('i,k,n',i,k,n)
                    result_matrix[i][k][n][4], result_matrix[i][k][n][5], result_matrix[i][k][n][6] =  f1_score_orig, acc_orig, Loss

           ##### syn CNN training         
                    net = net
                    data_type = '_syn'
                    print('CNN_train_syn================')
                    f1_score_orig, Loss, acc_orig = train_ch5(net, train_loader_syn, test_loader_test, 13, optimizer, device, num_epochs,w,net_name,data_type)
                    f1_score_orig,acc_orig,Loss = sorted(f1_score_orig, reverse=True)[:20],sorted(np.array(acc_orig)*100, reverse=True)[:20],sorted(Loss, reverse=False)[:20]
                    print('f1_score_orig length',len(f1_score_orig))
                    print('f1_score_orig','\n',f1_score_orig,'\n','acc_orig','\n',acc_orig,'\n','Loss is',Loss)
                    print('i,k,n',i,k,n)
                    result_matrix[i][k][n][7],result_matrix[i][k][n][8],result_matrix[i][k][n][9] =  f1_score_orig,acc_orig,Loss


    #visulization
        # dsta organization

        # select data for visulization
        result_m = np.zeros((3,5,5,16))
        # take average values for f1 acc and loss, and separate into max value and ave value
        result_m = selected_data(result_matrix, result_m)
        # for dimension 1, the dimension wthe valuas will be 3,4,5, becuase the k_fold setting , 
        #thus need to clean zeros for 3 and 4 dimension
        result_final = clean_zeros(result_m) 

        for s in range (len(result_final)):
            save_file_path='./weights/' + net_name+'/'
            save_file_name = save_file_path+'result'+str(s)+'.txt'
            print(save_file_name)
            np.savetxt(save_file_name, result_final[s], fmt="%.8e", delimiter=",")

        result_final = []
        for s in range(3):
            save_file_path='./weights/' + net_name+'/'
            save_file_name = save_file_path+'result'+str(s)+'.txt'
            print(save_file_name)
            result_final.append(np.loadtxt(save_file_name, dtype=float, delimiter=","))  


        #result_matrix= np.loadtxt(save_file_name, dtype=float, delimiter=",")

        print('--------',result_final[1].shape,'-----------')
        #choose which data to be visulized. 
        data_show = 'acc' # or 'f1' or 'loss'
        max_or_ave = 'ave'# or 'max'
        # set up indix to be kept
        x = max_or_ave_acc_f1_or_loss(max_or_ave, data_show)
        # only the dataset that will be visulized will be kept
        result_analysis = target_value(result_final,x)
        test = result_analysis.transpose((0,2,1))
        matrix1 = test.reshape(12,5)
        #print(matrix1)

    # 3D_1
        #matrix1 will be visulized
        fig = plt.figure(figsize=(30,10))
        ax = fig.add_subplot(111, projection='3d')  
        len_x, len_y = matrix1.shape
        _x = np.arange(len_x)
        _y = np.arange(len_y)
        print(_x,_y)
        xpos, ypos = np.meshgrid(_x, _y)
        xpos = xpos.flatten('F')
        ypos = ypos.flatten('F')
        zpos = np.zeros_like(xpos)

        dx = np.ones_like(zpos)
        dy = dx.copy()
        dz = matrix1.flatten()

        #print(dz)
        cmap=plt.cm.magma(plt.Normalize(0,60)(dz))

        ax.bar3d(xpos+0.12, ypos-0.1, zpos, dx-0.8, dy-0.8, dz, zsort='max', alpha = 1,
                 color=get_color(dz))
        #alpha=get_alpha(dz)
        #print(alpha[0])

        labels = ['3_svm_orig','3_svm_syn',
                            '3_cnn_orig','3_cnn_syn',
                            '4_svm_orig','4_svm_syn',
                            '4_cnn_orig','4_cnn_syn',
                            '5_svm_orig','5_svm_syn',
                            '5_cnn_orig','5_cnn_syn',
                            ]

        ax.set_xlabel('x')
        ax.set_xticks(np.arange(len_x+1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0,12)
        ax.set_ylabel('y')
        ax.set_yticks(np.arange(len_y+1))
        ax.set_yticklabels(['50','100','150','200','300'])
        ax.set_ylim(0,5)
        ax.set_zlabel('z')
        ax.set_zlim(0,100)
        ax.view_init(ax.elev+10, ax.azim+140)

        save_file_path='./weights/' + net_name+'/'
        save_file_name = save_file_path+'3D1.png'
        plt.savefig(save_file_name)

    #3D_2
        fig = plt.figure(figsize=(30,10))
        ax = fig.add_subplot(111, projection= Axes3D.name)
        X, Y = np.meshgrid([1,2,3,4,5,6,7,8,9,10,11,12], [1,2,3,4,5])
        Z = np.sin(X*Y)+1.5

        matrix2 = np.swapaxes(matrix1,0,1)
        print(matrix2.shape)
        #print(Z)

        matrix3 = matrix2
        print(matrix3)

        make_bars(ax, X,Y,matrix3, width=0.1,  )
        ax.set_xlabel('x')
        ax.set_xticks(np.arange(len_x+1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0,12)
        ax.set_ylabel('y')
        ax.set_yticks(np.arange(len_y+1))
        ax.set_yticklabels(['50','100','150','200','300'])
        ax.set_ylim(0,5)
        ax.set_zlabel('z')
        ax.set_zlim(0,80)
        ax.view_init(ax.elev+20, ax.azim+138)


        save_file_name = save_file_path+'3D_2.png'
        plt.savefig(save_file_name)

    #2D_1
        import seaborn as sns
        get_ipython().run_line_magic('matplotlib', 'inline')

        matrix2 = np.swapaxes(matrix1,0,1)
        df = pd.DataFrame({'50':matrix2[0],
                           '100':matrix2[1],
                           '150':matrix2[2],
                           '200':matrix2[3],
                           '300':matrix2[4],
                          },
                          index = labels
                          )

        df1 = pd.DataFrame({'A': [11, 21, 31],
                           'B': [12, 22, 32],
                           'C': [13, 23, 33]},
                          index=['ONE', 'TWO', 'THREE'])
        print(df)
        sns.set(font_scale=1.5)
        df.head()
        sns.set_context({"figure.figsize":(18,18)})
        ax = sns.heatmap(data=df,square=False) 
        ax.set_ylim([12, 0])

        ax=sns.heatmap(data=df,annot=True,cmap="RdBu_r")
        ax.set_ylim([12, 0])
        save_file_name = save_file_path+'2D_1.png'
        plt.savefig(save_file_name)


    #2D_2    

        fig = plt.figure(figsize=(25,5))
        ax = fig.add_subplot(111, projection= Axes3D.name)
        X, Y = np.meshgrid([1,2,3,4,5,6,7,8,9,10,11,12], [1,2,3,4,5])
        Z = np.sin(X*Y)+1.5
        matrix2 = np.swapaxes(matrix1,0,1)
        matrix3 = matrix2
        print(matrix3)

        ax.plot_surface(X, Y, matrix3, cmap=plt.cm.viridis, cstride=1, rstride=1)
        ax.set_xlabel('x')
        ax.set_xticks(np.arange(len_x+1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0,12)
        ax.set_ylabel('y')
        ax.set_yticks(np.arange(len_y+1))
        ax.set_yticklabels(['50','100','150','200','300'])
        ax.set_ylim(0,5)
        ax.set_zlabel('z')
        ax.set_zlim(0,80)
        ax.view_init(ax.elev+40, ax.azim+135)



        save_file_name = save_file_path+'2D_2.png'
        plt.savefig(save_file_name)

