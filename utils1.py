import matplotlib
import matplotlib.colors as colors
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import collections
import csv
import numpy as np
import time
from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader
import os


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
#
class HMData2(Dataset):
    def __init__(self,dataset,main_variable, transform=None):
        self.c1=dataset
        self.main_variable = main_variable
        
    def __len__(self):
        return len(self.c1)
    def __getitem__(self,i):

        if self.main_variable == 'full':
            final_data_c1=torch.cat((self.c1[i]['wea1'],self.c1[i]['wea2'],
                                 self.c1[i]['wea3'],self.c1[i]['wea4'],
                                 self.c1[i]['wea5'],self.c1[i]['wea6'],
                                 self.c1[i]['wea7'],self.c1[i]['wea8'],
                                 self.c1[i]['wea9'],self.c1[i]['wea10'],
                                 self.c1[i]['wea11'],self.c1[i]['wea12']
                                 ),1)
        elif self.main_variable == 'no_pest':
            final_data_c1=torch.cat((self.c1[i]['wea1'],self.c1[i]['wea2'],
                                 self.c1[i]['wea3'],self.c1[i]['wea4'],
                                 self.c1[i]['wea5'],
                                 self.c1[i]['wea7'],self.c1[i]['wea8'],
                                 self.c1[i]['wea9'],self.c1[i]['wea10'],
                                
                                 ),1)
        elif self.main_variable == 'no_rotation':
            final_data_c1=torch.cat((self.c1[i]['wea1'],self.c1[i]['wea2'],
                                 self.c1[i]['wea3'],self.c1[i]['wea4'],
                                 self.c1[i]['wea5'],self.c1[i]['wea6'],
                                
                                 self.c1[i]['wea11'],self.c1[i]['wea12']
                                 ),1)
        else: final_data_c1=torch.cat((self.c1[i]['wea1'],self.c1[i]['wea2'],
                                 self.c1[i]['wea3'],self.c1[i]['wea4'],
                                 self.c1[i]['wea5']
                                 ),1)
        #print('yyyy shape is ',final_data_c1.shape)
        entryID=self.c1[i]['entryID']
   #     print('+1+1+1+1+1+1+1+1',entryID)
        label=self.c1[i]['label']

        
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
            X= X[:,:,97:247,:]
            X = X.permute(0,1,3,2) #1,150,12, 1,12,150
            
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
            y_pre = net(X)
            y_pre = y_pre.argmax(dim=1).cpu()
            y = y.cpu()
            y_p_np=y_pre.numpy()
            y_np=y.numpy()
            f1_test = f1_score(y_np,y_p_np,average = 'weighted')
     #   f1_score_orig.append(f1_test)
      #  acc_orig.append(acc_sum/n)
    return acc_sum / n, f1_test

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs,w,net_name,data_type):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    best_loss = float('inf')    
    f1_score_orig = []
    Loss = []
    acc_orig = []
    for epoch in range(num_epochs):
        
        batch_count = 0
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for XX in train_iter: 
            X = XX['input'].to(device)
            y = XX['label'].to(device) 
            z = XX['entryID']
            X= X[:,:,97:247,:]
            #print(X.shape)
            X = X.permute(0,1,3,2) #1,150,12, 1,12,150
            y_hat = net(X)  
            if epoch >(num_epochs - 1):
                ans = []
                for t in y_hat:
                    if t[0]>t[1]:
                        ans.append(0)
                    else:ans.append(1)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        if epoch >=0 :
            test_acc,f1_test = evaluate_accuracy(test_iter, net,w)
             #   print('test_acc is', test_acc)
            Loss.append(train_l_sum / batch_count)
            f1_score_orig.append(f1_test)
            
            
            #print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
            #print('test_acc',test_acc)
            #print('max_acc',max(acc_orig))
            if len(acc_orig)>0:
                if test_acc > max(acc_orig):
                    print(net_name)
                    torch.save(net.state_dict(), "weights/%s/net%s.pth" % (net_name,data_type))
            acc_orig.append(test_acc)
     #       print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ epoch', epoch)
     #       print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
     #                 % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, 
     #                    test_acc, time.time() - start))
    return f1_score_orig, Loss, acc_orig
#
def load_svm(file1, file2,v_type):
    filename_his = file1
    print('The path of histone_data',filename_his)
    print('v_type is',v_type)
    if v_type == 'full':
        with open(filename_his) as fi_his:
            csv_reader=csv.reader(fi_his)
            data=list(csv_reader)
            ncols=(len(data[0]))
        fi_his.close()
        nrows = int((len(data)-1)/365)
        ncols = int((len(data[1])-2)*365)
        y=[]
        x = np.empty([nrows,ncols], dtype = float)
        #print(x.shape)
        for i in range (0, nrows):
            index_x = 0
            for m in range(0,365):
                for k in range(0,12):
                    x[i][m*12+k] = data[i*365+m+1][k+1]

        x = x[:,1164:2964]
        print('-------------------svm shape is',x.shape)
        test_num = (x[0])
    elif v_type == 'no_pest':
        with open(filename_his) as fi_his:
            csv_reader=csv.reader(fi_his)
            data=list(csv_reader)
            length = len(data)
            for i in range (0,length):
                del data[i][6]
                del data[i][-1]
                del data[i][-1]
    #            del data[i][-1]
            ncols=(len(data[0]))
        fi_his.close()
        nrows = int((len(data)-1)/365)
        ncols = int((len(data[1])-2)*365)
     #   print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
     #   print('nrow is', nrows)
        print('ncols is', ncols)
     #   print('data[1]-2 is', len(data[1])-2)
        y=[]
        x = np.empty([nrows,ncols], dtype = float)
        #print(x.shape)
        for i in range (0, nrows):
            index_x = 0
            for m in range(0,365):
                for k in range(0,9):
                    x[i][m*9+k] = data[i*365+m+1][k+1]
        x = x[:,873:2223]

        print(x.shape)
        print(x[1][0:20])
        test_num = (x[0])


    if v_type == 'no_rotation':
        with open(filename_his) as fi_his:
            csv_reader=csv.reader(fi_his)
            data=list(csv_reader)
            length = len(data)
            for i in range (0,length):
                del data[i][7]
                del data[i][7]
                del data[i][7]
                del data[i][7]
            ncols=(len(data[0]))
        fi_his.close()
        nrows = int((len(data)-1)/365)
        ncols = int((len(data[1])-2)*365)

        print('ncols is', ncols)
        y=[]
        x = np.empty([nrows,ncols], dtype = float)
        for i in range (0, nrows):
            index_x = 0
            for m in range(0,365):
                for k in range(0,8):
                    x[i][m*8+k] = data[i*365+m+1][k+1]
        x = x[:,776:1976]
        print(x.shape)
        print(x[0][0:20])
        test_num = (x[0])
    if v_type == 'weather_only':
        with open(filename_his) as fi_his:
            csv_reader=csv.reader(fi_his)
            data=list(csv_reader)
            length = len(data)
            for i in range (0,length):
                del data[i][7]
                del data[i][7]
                del data[i][7]
                del data[i][7]
                del data[i][6]

                del data[i][-1]
                del data[i][-1]
            ncols=(len(data[0]))
        fi_his.close()
        nrows = int((len(data)-1)/365)
        ncols = int((len(data[1])-2)*365)
        print('ncols is', ncols)
        y=[]
        x = np.empty([nrows,ncols], dtype = float)
        for i in range (0, nrows):
            index_x = 0
            for m in range(0,365):
                for k in range(0,5):
                    x[i][m*5+k] = data[i*365+m+1][k+1]
        x = x[:,485:1235]
        print(x.shape)
        print(x[0][0:20])
        test_num = (x[0])
    
    filename_his = file2 
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
    return x,y
def load_data_svm(folder,k,num_fold,num_synth,v_type):
    file_name_x = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_x_orig.csv'
    file_name_y = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_y_orig.csv'
    x_file = folder+file_name_x
    y_file  = folder+file_name_y
    print('reading file',x_file)
    attr_orig = loadData2(365,x_file,y_file)
    x_svm_orig,y_svm_orig = load_svm(x_file,y_file,v_type)
    file_name_x = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_x_syn.csv'               
    file_name_y = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_y_syn.csv'
    x_file = folder+file_name_x
    y_file  = folder+file_name_y
    print('reading file',x_file)
    attr_syn = loadData(365,x_file,y_file) 
    x_svm_syn,y_svm_syn = load_svm(x_file,y_file,v_type)

    file_name_x = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_x_test.csv'
    file_name_y = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_y_test.csv'
    x_file = folder+file_name_x
    y_file  = folder+file_name_y
    print('reading file',x_file)
    attr_test = loadData2(365,x_file,y_file)
    x_svm_test,y_svm_test = load_svm(x_file,y_file,v_type)
    return attr_orig, attr_syn, attr_test, x_svm_orig,y_svm_orig, x_svm_syn,y_svm_syn, x_svm_test,y_svm_test
def selected_data(result_matrix, result_m):
    for i in range(3):
        for j in range(5):
            for k in range(5):
                result_m[i][j][k][0] = result_matrix[i][j][k][0].mean()
                #print(result_m[i][j][k][0])
                result_m[i][j][k][1] = result_matrix[i][j][k][1].mean()
                #print(result_m[i][j][k][1])
                result_m[i][j][k][2] = result_matrix[i][j][k][2].mean()
                #print(result_m[i][j][k][2])
                result_m[i][j][k][3] = result_matrix[i][j][k][3].mean()
                #print(result_m[i][j][k][3])
                #f1
                result_m[i][j][k][4] = max(result_matrix[i][j][k][5])
                #print(result_m[i][j][k][4])
                result_m[i][j][k][5] = max(result_matrix[i][j][k][4])
                #print(result_m[i][j][k][5])
                result_m[i][j][k][6] = min(result_matrix[i][j][k][6])
                #print(result_m[i][j][k][6])
                result_m[i][j][k][7] = result_matrix[i][j][k][5].mean() #
                #print(result_m[i][j][k][7])
                result_m[i][j][k][8] = result_matrix[i][j][k][4].mean()
                #print(result_m[i][j][k][8])
                result_m[i][j][k][9] = result_matrix[i][j][k][6].mean()
                #print(result_m[i][j][k][9])


                result_m[i][j][k][10] = max(result_matrix[i][j][k][8])
                #print(result_m[i][j][k][10])
                result_m[i][j][k][11] = max(result_matrix[i][j][k][7])
                #print(result_m[i][j][k][11])
                result_m[i][j][k][12] = min(result_matrix[i][j][k][9])
                #print(result_m[i][j][k][12])
                result_m[i][j][k][13] = result_matrix[i][j][k][8].mean() 
                #print(result_m[i][j][k][13])
                result_m[i][j][k][14] = result_matrix[i][j][k][7].mean()
                #print(result_m[i][j][k][14])
                result_m[i][j][k][15] = result_matrix[i][j][k][9].mean()
                #print(result_m[i][j][k][15])
    return result_m

def clean_zeros(result_m):
    result_final = []
    for i in range(0,3):
        if i == 0:
            result_final.append(np.mean(result_m[i][0:3], axis = 0))
        elif i == 1:
            result_final.append(np.mean(result_m[i][0:4], axis = 0))
        elif i == 2:
            result_final.append(np.mean(result_m[i][0:5], axis = 0))
    return result_final

def max_or_ave_acc_f1_or_loss(max_or_ave, data_show):
    if max_or_ave == 'ave':
        if data_show == 'acc':
            x = np.array([1, 0,1, 0, 0, 0,0,1,0,0,0,0,0,1,0,0])
        elif data_show == 'f1':
            x = np.array([0, 1,0, 1, 0, 0,0,0,1,0,0,0,0,0,1,0])
        else: x = np.array([0, 1,0, 1, 0, 0,0,0,0,1,0,0,0,0,0,1])
    else:
        if data_show == 'acc':
            x = np.array([1, 0,1, 0, 1, 0,0,0,0,0,1,0,0,0,0,0])
        elif data_show == 'f1':
            x = np.array([0, 1,0, 1, 0, 1,0,0,0,0,0,1,0,0,0,0])
        else: x = np.array([0, 1,0, 1, 0, 0,1,0,0,0,0,0,1,0,0,0])
    return x

def target_value(result_final, x): 
    result_analysis = np.zeros((3,5,4))
    for i in range(len(result_final)):  
        for j in range(len(result_final[0])):
            print(result_final[i][j][x==1])
            result_analysis[i][j] = result_final[i][j][x==1]
    return result_analysis
def get_color(value_array):
    color = []
    for v in value_array:
        if (v < 0):
            color.append('k')
        elif (v < 55):
            color.append('y')
        elif (v < 60):
            color.append('g')
        elif (v < 65):
            color.append('b')
        elif (v < 70):
            color.append('c')
        elif (v < 80):
            color.append('m')
        else:
            color.append('r')
    return color
def make_bar(ax, x0=0, y0=0, width = 5, height=30 , cmap=plt.get_cmap('rainbow'),  
              norm=matplotlib.colors.Normalize(vmin=0, vmax=1), **kwargs ):
    # Make data
    u = np.linspace(0, 2*np.pi, 4+1)+np.pi/4.
    v_ = np.linspace(np.pi/4., 3./4*np.pi, 100)
    v = np.linspace(0, np.pi, len(v_)+2 )
    v[0] = 0 ;  v[-1] = np.pi; v[1:-1] = v_
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    xthr = np.sin(np.pi/4.)**2 ;  zthr = np.sin(np.pi/4.)
    x[x > xthr] = xthr; x[x < -xthr] = -xthr
    y[y > xthr] = xthr; y[y < -xthr] = -xthr
    z[z > zthr] = zthr  ; z[z < -zthr] = -zthr

    x *= 1./xthr*width; y *= 1./xthr*width
    z += zthr
    z *= height/(2.*zthr)
    #translate
    x += x0; y += y0
    #plot
    ax.plot_surface(x, y, z, cmap=cmap, norm=norm, **kwargs)

def make_bars(ax, x, y, height, width=0.3):
    widths = np.array(width)*np.ones_like(x)
    x = np.array(x).flatten()
    y = np.array(y).flatten()

    h = np.array(height).flatten()
    w = np.array(widths).flatten()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=h.max())
    for i in range(len(x.flatten())):
        make_bar(ax, x0=x[i], y0=y[i], width = w[i] , height=h[i], norm=norm)
