#!/usr/bin/env python
# coding: utf-8

# In[16]:


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


device = torch.device('cpu')

print(torch.__version__)
print(device)


# In[17]:


#
def clear_sheet_1():
    print('Concatenating histone modification and label>>>>>>>>>>>>')
#full_dataset = HMData(cell_train_dict1)
#print('The length of total dataset',len(full_dataset))
    filename = "/home/l/20211218 practice/Blackleg/"
    filename_blk = filename+'Blackleg.xlsx'
    data = xlrd.open_workbook(filename_blk)
    names = data.sheet_names()  
    table=data.sheets()
    print(table[0].col(1)[6])
    table1 = data.sheets()[1]
    nrows=table1.nrows
    print(names[0])
    table0 = data.sheets()[0]
    print(table0.col(1)[6])
#print(table.shape)

    table0_BEE = table0.col(2)
    table0_INC = table0.col(7)
    table0_SEV = table0.col(6)
    table0_WEA = table0.col(18)
    
    table0_2020 = table0.col(9)
    table0_2019 = table0.col(10)
    table0_2018 = table0.col(11)
    table0_2017 = table0.col(12)   
    table0_ROOT_INC = table0.col(4)
    table0_ROOT_SEV = table0.col(3)
    
    table1_BEE = table0.col(2)
    table1_INC = table0.col(7)
    table1_SEV = table0.col(6)
    table1_WEA = table0.col(18)
    
    table1_2020 = table0.col(9)
    table1_2019 = table0.col(10)
    table1_2018 = table0.col(11)
    table1_2017 = table0.col(12)  
    table1_ROOT_INC = table0.col(4)
    table1_ROOT_SEV = table0.col(3)
    
    for i in range(0,len(table0_INC)):
    #print(table0_WEA[i], table1_WEA[i])
        table0_WEA[i] = table0_WEA[i].value
        table1_WEA[i] = table1_WEA[i].value

    print('------27  +62')
    
    table1_BEE.append(table1_BEE[27])
    
    table1_INC.append(table1_INC[27])
    table1_SEV.append(table1_SEV[27])
    table1_WEA.append(table1_WEA[27])
    
    table1_2020.append(table1_2020[27])
    table1_2019.append(table1_2019[27])
    table1_2018.append(table1_2018[27])
    table1_2017.append(table1_2017[27])   
    table1_ROOT_INC.append(table1_ROOT_INC[27])
    table1_ROOT_SEV.append(table1_ROOT_SEV[27])
#print(table0_WEA[27])
    table1_WEA[27]='Three Hills'
    table1_WEA[62]='Delburne'


    print('------44   + 63')
    table1_BEE.append(table1_BEE[44])
    
    table1_INC.append(table1_INC[44])
    table1_SEV.append(table1_SEV[44])
    table1_WEA.append(table1_WEA[44])
    
    table1_2020.append(table1_2020[44])
    table1_2019.append(table1_2019[44])
    table1_2018.append(table1_2018[44])
    table1_2017.append(table1_2017[44])   
    table1_ROOT_INC.append(table1_ROOT_INC[44])
    table1_ROOT_SEV.append(table1_ROOT_SEV[44])
#print('now 63 WEA is',table0_WEA[63])
    table1_WEA[44]='Delburne'
    table1_WEA[63]='Stettler'

    print('------45  +64')
    table1_BEE.append(table1_BEE[45])
    
    table1_INC.append(table1_INC[45])
    table1_SEV.append(table1_SEV[45])
    table1_WEA.append(table1_WEA[45])
    
    
    table1_2020.append(table1_2020[45])
    table1_2019.append(table1_2019[45])
    table1_2018.append(table1_2018[45])
    table1_2017.append(table1_2017[45])  
    table1_ROOT_INC.append(table1_ROOT_INC[45])
    table1_ROOT_SEV.append(table1_ROOT_SEV[45])
#print('now 64 WEA is',table0_WEA[64])
    table1_WEA[45]='Delburne'
    table1_WEA[64]='Stettler'

    print('------47   +65')
    table1_BEE.append(table1_BEE[47])
    
    table1_INC.append(table1_INC[47])
    table1_SEV.append(table1_SEV[47])
    table1_WEA.append(table1_WEA[47])
    
    table1_2020.append(table1_2020[47])
    table1_2019.append(table1_2019[47])
    table1_2018.append(table1_2018[47])
    table1_2017.append(table1_2017[47])  
    table1_ROOT_INC.append(table1_ROOT_INC[47])
    table1_ROOT_SEV.append(table1_ROOT_SEV[47])
#print('now 64 WEA is',table0_WEA[65])
    table1_WEA[47]='Delburne'
    table1_WEA[65]='Stettler'

    print('------55   +++ 66')
    table1_BEE.append(table1_BEE[55])
    
    table1_INC.append(table1_INC[55])
    table1_SEV.append(table1_SEV[55])
    table1_WEA.append(table1_WEA[55])

    table1_2020.append(table1_2020[55])
    table1_2019.append(table1_2019[55])
    table1_2018.append(table1_2018[55])
    table1_2017.append(table1_2017[55])  
    table1_ROOT_INC.append(table1_ROOT_INC[55])
    table1_ROOT_SEV.append(table1_ROOT_SEV[55])
    
#print('now 66 WEA is',table0_WEA[66])
    table1_WEA[55]='Three Hills'
    table1_WEA[66]='Delburne'

    print('------delete42    42')
    del table1_BEE[42]
    del table1_INC[42]
    del table1_SEV[42]
    
    del table1_WEA[42]
    
    del table1_2020[42]
        
    del table1_2019[42]
    del table1_2018[42]
    del table1_2017[42]
    del table1_ROOT_INC[42]
    del table1_ROOT_SEV[42]
    print('------delete46         46')

    print(table1_INC[44],table1_WEA[44],table1_INC[45],table1_WEA[45],table1_INC[46],table1_WEA[46],table1_INC[47],table1_WEA[47])
    del table1_BEE[45]
    del table1_INC[45]
    del table1_SEV[45]
    
    del table1_WEA[45]
    
    del table1_2020[45]
        
    del table1_2019[45]
    del table1_2018[45]
    del table1_2017[45]
    
    del table1_ROOT_INC[45]
    del table1_ROOT_SEV[45]
    print(table1_INC[44],table1_WEA[44],table1_INC[45],table1_WEA[45],table1_INC[46],table1_WEA[46],table1_INC[47],table1_WEA[47])
    print('---------')
    print('56 is not Delbrurne')
    print(table1_WEA[56])
    print(table1_WEA[58])
    table1_WEA[56]='Delburne'
    table1_WEA[58]='Delburne'
    print(table1_WEA[56])
    print(table1_WEA[58])
    return(table1_SEV,table1_INC,table1_WEA,names,table1_BEE,
           table1_2020,table1_2019,table1_2018,table1_2017,table1_ROOT_INC,table1_ROOT_SEV)


# In[18]:


#
class HMData(Dataset):
    def __init__(self,dataset,transform=None):
        self.c1=dataset
    def __len__(self):
        return len(self.c1)
    def __getitem__(self,i):
        final_data_c1=torch.cat((self.c1[i]['wea1'],self.c1[i]['wea2'],
                                 self.c1[i]['wea3'],self.c1[i]['wea4'],self.c1[i]['wea5'],
                                 self.c1[i]['wea6'],self.c1[i]['wea7'],self.c1[i]['wea8'],
                                 self.c1[i]['wea9'],self.c1[i]['wea10'],self.c1[i]['wea11'],
                                 self.c1[i]['wea12'],self.c1[i]['wea13']),1)
                     
        
        entryID=self.c1[i]['entryID']
   #     print('+1+1+1+1+1+1+1+1',entryID)
        label00=self.c1[i]['label'].value
        if label00 < 10:
            label = 0
        elif label00 >= 10:
            label=1
     #   print(self.c1[i]['wea1'],self.c1[i]['wea2'],
      #                           self.c1[i]['wea3'],self.c1[i]['wea4'],self.c1[i]['wea5'])            
        final_data_c1=torch.cat((self.c1[i]['wea1'],self.c1[i]['wea2'],
                                 self.c1[i]['wea3'],self.c1[i]['wea4'],
                                 self.c1[i]['wea5'],self.c1[i]['wea6'],
                                 self.c1[i]['wea7'],self.c1[i]['wea8'],
                                 self.c1[i]['wea9'],self.c1[i]['wea10'],
                                 self.c1[i]['wea11'],self.c1[i]['wea12'],
                                 self.c1[i]['wea13']),1)
        
        
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
    


# In[19]:


#
def read_data_from_dataset(train_dataset,use_init_clusters=True):
    print(len(train_dataset))
    for  XX in train_dataset:
        
        x_train = XX['input']
        y_train = XX['label']
        #print(len(x_train))
    for YY in train_dataset:
        x_test = YY['input']
        y_test = YY['label']
       # print(x_test.shape)
    
    print(y_train)
    print(y_test)
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    
    # make the min to zero of labels
    y_train, y_test = transform_labels(y_train, y_test)

    classes, classes_counts = np.unique(y_train, return_counts=True)

    if len(x_train.shape) == 2:  # if univariate 
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # maximum number of prototypes which is the minimum count of a class
    print(classes_counts.max() + 1)
    print(MAX_PROTOTYPES_PER_CLASS + 1)
    
    max_prototypes = min(classes_counts.max() + 1,
                         MAX_PROTOTYPES_PER_CLASS + 1)
    init_clusters = None

    return x_train, y_train, x_test, y_test, nb_classes, classes, max_prototypes, init_clusters


# In[20]:


#
def get_weights_average_selected(x_train, dist_pair_mat, distance_algorithm='dtw'):
    # get the distance function 
    dist_fun = utils.constants.DISTANCE_ALGORITHMS[distance_algorithm]
    # get the distance function params 
    dist_fun_params = utils.constants.DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]
    # get the number of dimenions 
    num_dim = x_train[0].shape[1]
    # number of time series 
    n = len(x_train)
    # maximum number of K for KNN 
    max_k = 5 
    # maximum number of sub neighbors 
    max_subk = 2
    # get the real k for knn 
    k = min(max_k,n-1)
    # make sure 
    subk = min(max_subk,k)
    # the weight for the center 
    weight_center = 0.5 
    # the total weight of the neighbors
    weight_neighbors = 0.3
    # total weight of the non neighbors 
    weight_remaining = 1.0- weight_center - weight_neighbors
    # number of non neighbors 
    n_others = n - 1 - subk
    # get the weight for each non neighbor 
    if n_others == 0 : 
        fill_value = 0.0
    else:
        fill_value = weight_remaining/n_others
    # choose a random time series 
    idx_center = random.randint(0,n-1)
    # get the init dba 
    init_dba = x_train[idx_center]
    # init the weight matrix or vector for univariate time series 
    weights = np.full((n,num_dim),fill_value,dtype=np.float64)
    # fill the weight of the center 
    weights[idx_center] = weight_center
    # find the top k nearest neighbors
    topk_idx = np.array(get_neighbors(x_train,init_dba,k,dist_fun,dist_fun_params,
                         pre_computed_matrix=dist_pair_mat, 
                         index_test_instance= idx_center))
    # select a subset of the k nearest neighbors 
    final_neighbors_idx = np.random.permutation(k)[:subk]
    # adjust the weight of the selected neighbors 
    weights[topk_idx[final_neighbors_idx]] = weight_neighbors / subk
    # return the weights and the instance with maximum weight (to be used as 
    # init for DBA )
    return weights, init_dba

def augment_train_set(x_train, y_train, classes, N, dba_iters=5, 
                      weights_method_name = 'aa', distance_algorithm='dtw',
                      limit_N = True):
    """
    This method takes a dataset and augments it using the method in icdm2017. 
    :param x_train: The original train set
    :param y_train: The original labels set 
    :param N: The number of synthetic time series. 
    :param dba_iters: The number of dba iterations to converge.
    :param weights_method_name: The method for assigning weights (see constants.py)
    :param distance_algorithm: The name of the distance algorithm used (see constants.py)
    """
    # get the weights function
    weights_fun = utils.constants.WEIGHTS_METHODS[weights_method_name]
    # get the distance function 
    dist_fun = utils.constants.DISTANCE_ALGORITHMS[distance_algorithm]
    # get the distance function params 
    dist_fun_params = utils.constants.DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]
    # synthetic train set and labels 
    synthetic_x_train = []
    synthetic_y_train = []
    # loop through each class
    for c in classes: 
        # get the MTS for this class 
        c_x_train = x_train[np.where(y_train==c)]

#        if len(c_x_train) == 1 :
            # skip if there is only one time series per set
#            continue

#        if limit_N == True:
            # limit the nb_prototypes
#            nb_prototypes_per_class = min(N, len(c_x_train))
#        else:
            # number of added prototypes will re-balance classes
        nb_prototypes_per_class = N + (N-len(c_x_train))

        # get the pairwise matrix 
        if weights_method_name == 'aa': 
            # then no need for dist_matrix 
            dist_pair_mat = None 
        else: 
            dist_pair_mat = calculate_dist_matrix(c_x_train,dist_fun,dist_fun_params)
        # loop through the number of synthtectic examples needed
        for n in range(nb_prototypes_per_class): 
            # get the weights and the init for avg method 
            weights, init_avg = weights_fun(c_x_train,dist_pair_mat,
                                            distance_algorithm=distance_algorithm)
            # get the synthetic data 
            synthetic_mts = dba(c_x_train, dba_iters, verbose=False, 
                            distance_algorithm=distance_algorithm,
                            weights=weights,
                            init_avg_method = 'manual',
                            init_avg_series = init_avg)  
            # add the synthetic data to the synthetic train set
            synthetic_x_train.append(synthetic_mts)
            # add the corresponding label 
            synthetic_y_train.append(c)
    # return the synthetic set 
    return np.array(synthetic_x_train), np.array(synthetic_y_train)


# In[21]:


##  这是培训集合，扩增后的保存  X
def save_syn_x_y(syn_x_train,syn_y_train, x_file, y_file):
  print('Liadubg syn X >>>>>>>>>>>>>>>>>>>>>>>> in load syn')
 
  print('syn_x shape',syn_x_train.shape)
  num_items = len(syn_x_train)
  print('length of x is', num_items)
  nrows = syn_x_train.shape[1]
  ncols = syn_x_train.shape[2]

  print('nrows is', nrows)
  print('ncols is', ncols)
  attr=collections.OrderedDict()
  f= open(x_file,'w')
  f.truncate()
  f.write('ID')
  f.write(',')

  f.write('tem_min')
  f.write(',')
  f.write('tem_max')
  f.write(',')
  f.write('hum_ave')
  f.write(',')
  f.write('prec_mm')
  f.write(',')
  f.write('wind')
  f.write(',')

  f.write('BEE')
  f.write(',')
  f.write('Rotation_2020')
  f.write(',')
  f.write('Rotation_2019')
  f.write(',')
  f.write('Rotation_2018')
  f.write(',')
  f.write('Rotation_2017')
  f.write(',')
  f.write('Roo_M_inc')
  f.write(',')
  f.write('Roo_M_sev')
  f.write(',')
  f.write('table_sev')

  f.write('\n')

  for i in range (0, num_items):
      for j in range (0, nrows):
          entry = 'entry' + str(i)
          f.write(entry)
          f.write(',')
          for k in range (0, ncols):
             # colum = 'wea' + str(k)
              f.write(str(syn_x_train[i][j][k]))
              if k ==12:
                 # print('ok')
                  f.write('\n')
              else: f.write(',')




  f.close()
  print('Liadubg syn Y >>>>>>>>>>>>>>>>>>>>>>>> in load syn')
  print('syn_y shape',syn_y_train.shape)
  num_items = len(syn_y_train)
  print('length of y is', num_items)
  print('n_item', num_items)
  attr=collections.OrderedDict()
  f= open(y_file,'w')
  f.truncate()
  f.write('ID')
  f.write(',')

  f.write('blk_inc')
  f.write('\n')

  for i in range (0, num_items):
      entry = 'entry' + str(i)
      f.write(entry)
      f.write(',')
      f.write(str(syn_y_train[i]))
      f.write('\n')




  f.close()


# In[22]:


#
def save_valid_x_y(syn_x_train,syn_y_train, x_file, y_file):
    print('Liadubg syn X >>>>>>>>>>>>>>>>>>>>>>>> in load syn')
 
    print('syn_x shape',syn_x_train.shape)
    num_items = len(syn_x_train)
    print('length of x is', num_items)
    nrows = syn_x_train.shape[2]
    ncols = syn_x_train.shape[3]

    print('nrows is', nrows)
    print('ncols is', ncols)
    attr=collections.OrderedDict()
    f= open(x_file,'w')
    f.truncate()
    f.write('ID')
    f.write(',')

    f.write('tem_min')
    f.write(',')
    f.write('tem_max')
    f.write(',')
    f.write('hum_ave')
    f.write(',')
    f.write('prec_mm')
    f.write(',')
    f.write('wind')
    f.write(',')

    f.write('BEE')
    f.write(',')
    f.write('Rotation_2020')
    f.write(',')
    f.write('Rotation_2019')
    f.write(',')
    f.write('Rotation_2018')
    f.write(',')
    f.write('Rotation_2017')
    f.write(',')
    f.write('Roo_M_inc')
    f.write(',')
    f.write('Roo_M_sev')
    f.write(',')
    f.write('table_sev')

    f.write('\n')

    for i in range (0, num_items):
        for j in range (0, nrows):
            entry = 'entry' + str(i)
            f.write(entry)
            f.write(',')
         #   print(i,j)
            for k in range (0, ncols):
            #    print(k)
               # colum = 'wea' + str(k)
                f.write(str((syn_x_train[i][0][j][k]).numpy()))
                if k ==12:
          #          print('ok')
                    f.write('\n')
                else: f.write(',')
    
    
    
    f.close()




    ## this is to save the validataion for Y##
    ## this is to save the validataion for Y##
    print('Liadubg validation 19 Y >>>>>>>>>>>>>>>>>>>>>>>> in load syn')

    num_items = len(syn_y_train)
    print('length of x is', num_items)
    nrows = syn_x_train.shape[2]
    ncols = syn_x_train.shape[3]

    print('nrows is', nrows)
    print('ncols is', ncols)
    attr=collections.OrderedDict()
    f= open(y_file,'w')
    f.truncate()
    f.write('ID')
    f.write(',')

    f.write('blk_inc')
    f.write('\n')

    for i in range (0, num_items):
        entry = 'entry' + str(i)
        f.write(entry)
        f.write(',')
        f.write(str(syn_y_train[i].numpy()))
        f.write('\n')





    f.close()


# In[23]:


table_SEV,table_INC,table_WEA,names,table_BEE,table_2020,table_2019,table_2018,table_2017,table_ROOT_INC,table_ROOT_SEV = clear_sheet_1()
table_INC = table_INC[2:]
table_SEV = table_SEV[2:]
table_WEA = table_WEA[2:]
names = names[2:]
table_BEE = table_BEE[2:]
table_2020 = table_2020[2:]
table_2019= table_2019[2:]
table_2018= table_2018[2:]
table_2017= table_2017[2:]
table_ROOT_INC= table_ROOT_INC[2:]
table_ROOT_SEV= table_ROOT_SEV[2:]

for i in range (0, len(table_BEE)):
    print(i)
    print(table_SEV[i], table_BEE[i].value,table_ROOT_SEV[i].value,table_ROOT_INC[i].value,table_INC[i].value,
          table_2020[i].value,table_2019[i].value,table_2018[i].value,
          table_2017[i].value,table_WEA[i]
          )
    
print(len(names))
print(len(table_INC))
print(table_INC)


# In[24]:


#
def load_attr_cross_validation(filename,names,table_INC,table_WEA,table_BEE,
              table_2020,table_2019,table_2018,table_2017,
             table_SEV,table_ROOT_INC,table_ROOT_SEV,train_kfold):
#    print('Data Reading>>>>>>>>>>>>>>>')
    print(table_ROOT_INC)
    filename_his = filename+'Blackleg.xlsx'

#    print('The path of histone_data',filename_his)
    data = xlrd.open_workbook(filename_his)
    names = data.sheet_names()  
    table=data.sheets()
    table1 = data.sheets()[1]
 ## this is to load weather information into Conditions[sheet][row][col]   
    nrows=table1.nrows-1
    len_sheets=len(data.sheets())-1
    ncols = table1.ncols-2
    attr_train=collections.OrderedDict()
    attr_test=collections.OrderedDict()
#    print('nrows is',nrows)
#    print('ncols is',ncols)
#    print('nsheets is',len_sheets)
    w=len(names)-1
#    print('number of sheets is ', w)
    num_items = len(table_INC)
 #   print('number of items is', num_items)
    count_train=0
    count_test = 0
    for i in range (0, num_items):
#        print('the number of items is i',num_items)
        for j in range(0,w):
    #        print('the number of sheets is j',w-1)
            if table_WEA[i] == names[j+1]:
   #             print('the number of rows is', nrows)
                wea1=torch.zeros(nrows,1)
                wea2=torch.zeros(nrows,1)
                wea3=torch.zeros(nrows,1)
                wea4=torch.zeros(nrows,1)
                wea5=torch.zeros(nrows,1)  
                wea6=torch.zeros(nrows,1)
                wea7=torch.zeros(nrows,1)
                wea8=torch.zeros(nrows,1)
                wea9=torch.zeros(nrows,1)  
                wea10=torch.zeros(nrows,1)
                
                wea10=torch.zeros(nrows,1)  
                wea11=torch.zeros(nrows,1)
                wea12=torch.zeros(nrows,1)
                wea13=torch.zeros(nrows,1)
                
                for k in range (0,nrows):
                    wea1[k][0]=float(table[j+1].row(k+1)[2].value)
     #       print(int(table[j+1].row(w+1)[2].value))
                    wea2[k][0]=float(table[j+1].row(k+1)[3].value)
                    wea3[k][0]=float(table[j+1].row(k+1)[4].value)
                    wea4[k][0]=float(table[j+1].row(k+1)[5].value)
                    wea5[k][0]=float(table[j+1].row(k+1)[6].value)
                    wea6[k][0]=float(table_BEE[i].value)
                    wea7[k][0]=float(table_2020[i].value)
                    wea8[k][0]=float(table_2019[i].value)
                    wea9[k][0]=float(table_2018[i].value)
                    wea10[k][0]=float(table_2017[i].value)
                    wea11[k][0]=float(table_ROOT_INC[i].value) 
                    wea12[k][0]=float(table_ROOT_SEV[i].value)
                    wea13[k][0]=float(table_SEV[i].value)
 
        ###############
                train = 1
                for m in range(0,len(train_kfold)):
                    #print(train_kfold[m])
                    if i == train_kfold[m]:
                        train = 0
                
                if train == 1:
                    attr_test[count_train]={
                        'entryID':i,
                        'label':table_INC[i],

                        'wea1':wea1,
                        'wea2':wea2,
                        'wea3':wea3,
                        'wea4':wea4,
                        'wea5':wea5,
                        'wea6':wea6,
                        'wea7':wea7,
                        'wea8':wea8,
                        'wea9':wea9,
                        'wea10':wea10,
                        'wea11':wea11,
                        'wea12':wea12,
                        'wea13':wea13
                        }
                 #   print('Test      entryID is', i, 'label is',table_INC[i])
                    count_train += 1
                else:
                    attr_train[count_test]={
                        'entryID':i,
                        'label':table_INC[i],

                        'wea1':wea1,
                        'wea2':wea2,
                        'wea3':wea3,
                        'wea4':wea4,
                        'wea5':wea5,
                        'wea6':wea6,
                        'wea7':wea7,
                        'wea8':wea8,
                        'wea9':wea9,
                        'wea10':wea10,
                        'wea11':wea11,
                        'wea12':wea12,
                        'wea13':wea13
                        }
                   # print('TRAIN      entryID is', i, 'label is',table_INC[i])
                    count_test +=1
        
       # print('count is ',count)
        
   #     print(count)
    return attr_train, attr_test


# In[26]:


filename_his = "/home/l/20211218 practice/svm/blackleg_3classes.csv"
print('The path of histone_data',filename_his)
with open(filename_his) as fi_his:
    csv_reader=csv.reader(fi_his)
    data=list(csv_reader)
    ncols=(len(data[0]))
fi_his.close()

nrows = (len(data)-1)
ncols = (len(data[1])-2)
print(nrows)
print(ncols)
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


# In[30]:



fold_num = [3,4,5]
for i in range(0,len(fold_num)):
    num_fold = fold_num[i]
    kf = StratifiedKFold(n_splits=num_fold,shuffle=True)
    for k,(train_kfold,test_kfold) in enumerate(kf.split(x_kfold,y_kfold)):
        

    # load original file and separate in to train and test dataset##
        filename = "/home/l/20211218 practice/Blackleg/"
        attr_train, attr_test = load_attr_cross_validation(filename,names,table_INC,table_WEA,table_BEE,
                      table_2020,table_2019,table_2018,table_2017,
                     table_SEV,table_ROOT_INC,table_ROOT_SEV,train_kfold)


    # process data for synthesisze and validation
        train_HMData = HMData(attr_train)
        print(train_HMData[1]['input'].shape)
        print('The length of total dataset',len(train_HMData))
        train_size = int(1 * len(train_HMData))
        test_size = len(train_HMData) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(train_HMData, 
                                                                      [train_size, test_size])
        print('The number of train dataset1 is:',len(train_dataset))


        test_HMData = HMData(attr_test)
        train_size = int(1 * len(test_HMData))
        test_size = len(test_HMData) - train_size
        test_dataset, test_dataset1 = torch.utils.data.random_split(test_HMData, 
                                                                      [train_size, test_size])
        print('The number of train dataset1 is:',len(test_dataset))


        batch_size = len(attr_train)
        print('Batch size is', batch_size)
    #synthesisze data prep
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=batch_size, 
                                                       shuffle=True)
    # validation data prep
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=batch_size, 
                                                       shuffle=True)


    # synthesize training data
        num_syn = [50,100,150,200,300]
        for n in range(0, len(num_syn)):
            num_synth = num_syn[n]
            print('the num of syn is', num_synth)     
            x_train, y_train, x_test, y_test,nb_classes, classes, max_prototypes, init_clusters = read_data_from_dataset(train_loader,use_init_clusters=True)
            x_train = x_train.to('cpu', torch.double).detach().numpy()
            x_train = x_train.reshape(x_train.shape[0],x_train.shape[2],x_train.shape[3])
            syn_train_set, distance_algorithm= augment_train_set(x_train, y_train, classes, num_synth,limit_N = True,
                          weights_method_name='as', 
                          distance_algorithm='dtw'),'dtw'
            print('The number of test dataset1 is:',len(test_dataset))



    # save syn and validation data into csv files         
            folder = "/home/l/20211218 practice/aaltd18-master1/creat_ALL_data_set/1_CNN_data_prepare_1000_problem_solved_2_class_2nd_try/"
            file_name_x = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_x_syn.csv'
            file_name_y = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_y_syn.csv'
            x_file = folder+file_name_x
            y_file  = folder+file_name_y
            syn_x_train, syn_y_train = syn_train_set
            print('saving file',x_file)
            save_syn_x_y(syn_x_train,syn_y_train, x_file, y_file)

            file_name_x = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_x_test.csv'
            file_name_y = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_y_test.csv'
            x_file = folder+file_name_x
            y_file  = folder+file_name_y
            for YY in test_loader:
                x_test = YY['input']
                y_test = YY['label']
                ID = YY['entryID']
            print('saving file',x_file)
            save_valid_x_y(x_test,y_test, x_file, y_file)   
            
            file_name_x = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_x_orig.csv'
            file_name_y = 'CNN_' + str(num_fold)  + '_fold' + '_of_' + str(k+1) + '_' + str(num_synth) + '_y_orig.csv'
            x_file = folder+file_name_x
            y_file  = folder+file_name_y
            for YY in train_loader:
                x_orig = YY['input']
                y_orig = YY['label']
                ID = YY['entryID']
            print('saving file',x_file)
            save_valid_x_y(x_orig,y_orig, x_file, y_file)   



