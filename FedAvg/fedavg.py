#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time
import matplotlib
import sys
import pylab
# matplotlib.use('Agg')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd
import math
import numpy as np
import random
import collections
import torch
import torch.nn.functional as F
import sympy as sy

from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from tqdm import tqdm
from torch import autograd
from tensorboardX import SummaryWriter
from sympy import solve
from sympy.abc import y
from scipy import optimize

from sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from options import args_parser
from Update import LocalUpdate
from FedNets import MLP1, CNNMnist, CNN_test, CNNCifar
from averaging import average_weights
from Calculate import minkowski_distance, mahala_distance, noise_add, sample_para


# if __name__ == '__main__':    
#     # return the available GPU
#     av_GPU = torch.cuda.is_available()
#     if  av_GPU == False:
#         exit('No available GPU')
#     # parse args
#     args = args_parser()
#     # define paths
#     path_project = os.path.abspath('..')

#     summary = SummaryWriter('local')
#     args.gpu = 0               # -1 (CPU only) or GPU = 0
#     args.lr = 0.02             # 0.001 for cifar dataset
#     args.model = 'mlp'         # 'mlp' or 'cnn'
#     args.dataset = 'mnist'     # 'mnist'
#     args.num_users = 50         # numb of users
#     args.num_Chosenusers = 50
#     args.epochs = 5            # numb of global iters
#     args.local_ep = 10          # numb of local iters
#     args.num_items_train = 800 # numb of local data size # 
#     args.num_items_test =  512
#     args.local_bs = 800        # Local Batch size (1200 = full dataset)
#                                # size of a user for mnist, 2000 for cifar)
#     args.degree_noniid = 1
                               
#     args.set_epochs = [100]
#     args.set_num_Chosenusers = [50]
    
#     args.set_degree_noniid = [0]
#     args.strict_iid = True
#     args.ratio_train = [0.5,0.75,1,1.25,1.5]
#     args.num_experiments = 1
#     args.clipthr = 20
    
#     args.iid = True
def main(args): 
    #####-Choose Variable-#####
    set_variable = args.set_num_Chosenusers
    set_variable0 = copy.deepcopy(args.set_epochs)
    set_variable1 = copy.deepcopy(args.set_degree_noniid)
    
    if not os.path.exists('./experiresult'):
        os.mkdir('./experiresult')
          
    # load dataset and split users
    dict_users,dict_users_train,dict_users_test = {},{},{}
    dataset_train,dataset_test = [],[]
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('./dataset/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        dataset_test = datasets.MNIST('./dataset/mnist/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
            # sample users
        if args.iid:
            dict_users = mnist_iid(args, dataset_train, args.num_users, args.num_items_train)
            # dict_users_test = mnist_iid(dataset_test, args.num_users, args.num_items_test) 
            dict_sever = mnist_iid(args, dataset_test, args.num_users, args.num_items_test)
        else:
            dict_users = mnist_noniid(args, dataset_train, args.num_users, args.num_items_train)
            dict_sever = mnist_noniid(args, dataset_test, args.num_users, args.num_items_test)
        
    elif args.dataset == 'cifar':
        dict_users_train, dict_sever = {},{}
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./dataset/cifar/', train=True, transform=transform, target_transform=None, download=True)
        dataset_test = copy.deepcopy(dataset_train)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users, args.num_items_train)
            num_train = int(0.6*args.num_items_train)
            for idx in range(args.num_users):
                dict_users_train[idx] = set(list(dict_users[idx])[:num_train])
                dict_sever[idx] = set(list(dict_users[idx])[num_train:])
        else:
            dict_users = cifar_noniid(args, dataset_train, args.num_users, args.num_items_train)
            dict_test = []
            num_train = int(0.6*args.num_items_train)
            for idx in range(args.num_users):
                dict_users_train[idx] = set(list(dict_users[idx])[:num_train])
                dict_sever[idx] = set(list(dict_users[idx])[num_train:])
        
    # sample users
    if args.iid:
        dict_users = mnist_iid(args,dataset_train, args.num_users, args.num_items_train)
        # dict_users_test = mnist_iid(dataset_test, args.num_users, args.num_items_test) 
        dict_sever = mnist_iid(args,dataset_test, args.num_users, args.num_items_test)
    else:
        dict_users = mnist_noniid(args, dataset_train, args.num_users, args.num_items_train)
        dict_sever = mnist_iid(args,dataset_test, args.num_users, args.num_items_test)          

    img_size = dataset_train[0][0].shape    
    
    for v in range(len(set_variable)):
        final_train_loss = [[0 for i in range(len(set_variable1))] for j in range(len(set_variable0))]
        final_train_accuracy = [[0 for i in range(len(set_variable1))] for j in range(len(set_variable0))]
        final_test_loss = [[0 for i in range(len(set_variable1))] for j in range(len(set_variable0))]
        final_test_accuracy = [[0 for i in range(len(set_variable1))] for j in range(len(set_variable0))]
        final_com_cons = [[0 for i in range(len(set_variable1))] for j in range(len(set_variable0))]
        args.num_Chosenusers = copy.deepcopy(set_variable[v])
        for s in range(len(set_variable0)):
            for j in range(len(set_variable1)): 
                args.epochs = copy.deepcopy(set_variable0[s])   
                args.degree_noniid = copy.deepcopy(set_variable1[j])
                print(args)
                loss_test, loss_train = [], []
                acc_test, acc_train = [], []          
                for m in range(args.num_experiments):
                    # build model
                    net_glob = None
                    if args.model == 'cnn' and args.dataset == 'mnist':
                        if args.gpu != -1:
                            torch.cuda.set_device(args.gpu)
                            # net_glob = CNNMnist(args=args).cuda()
                            net_glob = CNN_test(args=args).cuda()
                        else:
                            net_glob = CNNMnist(args=args)
                    elif args.model == 'mlp' and args.dataset == 'mnist':
                        len_in = 1
                        for x in img_size:
                            len_in *= x
                        if args.gpu != -1:
                            torch.cuda.set_device(args.gpu)
                            net_glob = MLP1(dim_in=len_in, dim_hidden=256, dim_out=args.num_classes).cuda()
                        else:
                            net_glob = MLP1(dim_in=len_in, dim_hidden=256, dim_out=args.num_classes)
                    elif args.model == 'cnn' and args.dataset == 'cifar':
                        if args.gpu != -1:
                            net_glob = CNNCifar(args).cuda()
                        else:
                            net_glob = CNNCifar(args)
                    else:
                        exit('Error: unrecognized model')
                    print("Nerual Net:",net_glob)
                
                    net_glob.train()  #Train() does not change the weight values
                    # copy weights
                    w_glob = net_glob.state_dict()       
                    w_size = 0
                    w_size_all = 0
                    for k in w_glob.keys():
                        size = w_glob[k].size()
                        if(len(size)==1):
                            nelements = size[0]
                        else:
                            nelements = size[0] * size[1]
                        w_size += nelements*4
                        w_size_all += nelements
                        # print("Size ", k, ": ",nelements*4)
                    print("Weight Size:", w_size, " bytes")
                    print("Weight & Grad Size:", w_size*2, " bytes")
                    print("Each user Training size:", 784* 8/8* args.local_bs, " bytes")
                    print("Total Training size:", 784 * 8 / 8 * 60000, " bytes")
                    # training
                    loss_avg_list, acc_avg_list, list_loss, loss_avg, com_cons = [], [], [], [], []  
                    ###  FedAvg Aglorithm  ###      
                    for iter in range(args.epochs):
                        print('\n','*' * 20,f'Epoch: {iter}','*' * 20)
                        if  args.num_Chosenusers < args.num_users:
                            chosenUsers = random.sample(range(args.num_users),args.num_Chosenusers)
                            chosenUsers.sort()
                        else:
                            chosenUsers = range(args.num_users)
                        print("\nChosen users:", chosenUsers)                
                        w_locals, w_locals_1ep, loss_locals, acc_locals = [], [], [], []

                        values_golbal = []
                        for i in w_glob.keys():
                            values_golbal += list(w_glob[i].view(-1).cpu().numpy())  
                        
                        for idx in range(len(chosenUsers)):
                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[chosenUsers[idx]], tb=summary)
                            w_1st_ep, w, loss, acc = local.update_weights(net=copy.deepcopy(net_glob))
                            w_locals.append(copy.deepcopy(w))
                            ### get 1st ep local weights ###
                            w_locals_1ep.append(copy.deepcopy(w_1st_ep))            
                            loss_locals.append(copy.deepcopy(loss))
                            # print("User ", chosenUsers[idx], " Acc:", acc, " Loss:", loss)
                            acc_locals.append(copy.deepcopy(acc))

                            # histogram for all clients
                            values_local = []
                            for i in w_glob.keys():
                                values_local += list(w[i].view(-1).cpu().numpy())
                            values_increment = [values_local[i]-values_golbal[i] for i in range(len(values_local))]                                     
                            value_sequence = sorted([d for d in values_increment], reverse=True)  # value sequence
                            hist, bin_edges = np.histogram(value_sequence, bins=100)
                            # valueCount = collections.Counter(hist)
                            # val, cnt = zip(*valueCount.items())
                            #print(hist, bin_edges)
                            # fig, ax = plt.subplots()
                            plt.close()
                            # plt.bar(range(len(hist)), hist, width=0.80, color='b')
                            # plt.close()
                            # plt.hist(value_sequence,bin_edges,color='b',alpha=0.8, rwidth=0.8)
                            plt.hist(value_sequence,bin_edges,color='steelblue',edgecolor = 'black',alpha=0.8)
                            plt.savefig('./histogra/histogra-{}-client-{}-iter-{}.pdf'.format(args.model,idx,iter))
                            plt.show()
                              
                        # malicious_users = [0, 3]
                        # w_locals = noise_add(args, w_locals, 0.001, malicious_users)                                            
                        ### update global weights ###                
                        # w_locals = users_sampling(args, w_locals, chosenUsers)
                        w_glob = average_weights(w_locals) 

                        # val_min_dist, val_mah_dist = [], []
                        # for i in range(len(w_locals)):
                        #     val_min_dist.append(minkowski_distance(w_locals[i],w_glob,1))
                        #     val_mah_dist.append(mahala_distance(w_locals[i],w_glob,w_locals,5))
                        # print('Minkowski distance:', val_mah_dist)
                        # print('Mahala distance:', val_min_dist)    
                        
                        # copy weight to net_glob
                        net_glob.load_state_dict(w_glob)
                        # global test
                        list_acc, list_loss = [], []
                        net_glob.eval()
                        for c in range(args.num_users):
                            net_local = LocalUpdate(args=args, dataset=dataset_test, idxs=dict_sever[idx], tb=summary)
                            acc, loss = net_local.test(net=net_glob)                    
                            # acc, loss = net_local.test_gen(net=net_glob, idxs=dict_users[c], dataset=dataset_test)
                            list_acc.append(acc)
                            list_loss.append(loss)
                        # print("\nEpoch: {}, Global test loss {}, Global test acc: {:.2f}%".\
                        #      format(iter, sum(list_loss) / len(list_loss),100. * sum(list_acc) / len(list_acc)))
                        # print loss
                        loss_avg = sum(loss_locals) / len(loss_locals)
                        acc_avg = sum(acc_locals) / len(acc_locals)
                        loss_avg_list.append(loss_avg)
                        acc_avg_list.append(acc_avg) 
                        print("\nTrain loss: {}, Train acc: {}".\
                              format(loss_avg_list[-1], acc_avg_list[-1]))
                        print("\nTest loss: {}, Test acc: {}".\
                              format(sum(list_loss) / len(list_loss), sum(list_acc) / len(list_acc)))
                        
                        # if (iter+1)%20==0:
                        #     torch.save(net_glob.state_dict(),'./Train_model/glob_model_{}epochs.pth'.format(iter))
                    
                    loss_train.append(loss_avg)
                    acc_train.append(acc_avg)               
                    loss_test.append(sum(list_loss) / len(list_loss))                
                    acc_test.append(sum(list_acc) / len(list_acc))
                    com_cons.append(iter+1)
                # plot loss curve
                final_train_loss[s][j] = copy.deepcopy(sum(loss_train) / len(loss_train))
                final_train_accuracy[s][j] = copy.deepcopy(sum(acc_train) / len(acc_train))
                final_test_loss[s][j] = copy.deepcopy(sum(loss_test) / len(loss_test))
                final_test_accuracy[s][j] = copy.deepcopy(sum(acc_test) / len(acc_test))
                final_com_cons[s][j] = copy.deepcopy(sum(com_cons) / len(com_cons))
    
            print('\nFinal train loss:', final_train_loss)
            print('\nFinal train accuracy:', final_train_accuracy)
            print('\nFinal test loss:', final_test_loss)
            print('\nFinal test accuracy:', final_test_accuracy)                    
        timeslot = int(time.time())
        data_test_loss = pd.DataFrame(index = set_variable0, columns = set_variable1, data = final_train_loss)
        data_test_loss.to_csv('./experiresult/'+'train_loss_{}_{}.csv'.format(set_variable[v],timeslot))
        data_test_loss = pd.DataFrame(index = set_variable0, columns = set_variable1, data = final_test_loss)
        data_test_loss.to_csv('./experiresult/'+'test_loss_{}_{}.csv'.format(set_variable[v],timeslot))
        data_test_acc = pd.DataFrame(index = set_variable0, columns = set_variable1, data = final_train_accuracy)
        data_test_acc.to_csv('./experiresult/'+'train_acc_{}_{}.csv'.format(set_variable[v],timeslot))
        data_test_acc = pd.DataFrame(index = set_variable0, columns = set_variable1, data = final_test_accuracy)
        data_test_acc.to_csv('./experiresult/'+'test_acc_{}_{}.csv'.format(set_variable[v],timeslot))
        data_test_acc = pd.DataFrame(index = set_variable0, columns = set_variable1, data = final_com_cons)
        data_test_acc.to_csv('./experiresult/'+'aggregation_consuming_{}_{}.csv'.format(set_variable[v],timeslot))
        
        plt.close()

        # dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeslot))
        # with open('./experiresult/new_fed_{}UEs_{}_{}_{}_C{}_iid{}_{}.csv'.\
        #   format(args.num_users, args.dataset,\
        #                 args.model,args.local_ep, iter, args.iid, timeslot),'w',encoding='utf-8') as f:
        #   f.write('Test_loss:')
        #   f.write(str(final_train_loss))
        #   f.write('\nTest_accuracy:')
        #   f.write(str(final_train_accuracy))
        #   f.write('\nTrain_loss:')
        #   f.write(str(final_test_loss))
        #   f.write('\nTrain_accuracy:')
        #   f.write(str(final_test_accuracy))
        
if __name__ == '__main__':    
    # return the available GPU
    av_GPU = torch.cuda.is_available()
    if  av_GPU == False:
        exit('No available GPU')
    # parse args
    args = args_parser()
    # define paths
    path_project = os.path.abspath('..')

    summary = SummaryWriter('local')
    args.gpu = 0               # -1 (CPU only) or GPU = 0
    args.lr = 0.02             # 0.001 for cifar dataset
    args.model = 'mlp'         # 'mlp' or 'cnn'
    args.dataset = 'mnist'     # 'mnist' or cifar
    args.num_users = 5         # numb of users
    args.num_Chosenusers = 5
    args.epochs = 10            # numb of global iters
    args.local_ep = 10          # numb of local iters
    args.num_items_train = 600 # numb of local data size # 
    args.num_items_test =  512
    args.local_bs = 800        # Local Batch size (1200 = full dataset)
                                # size of a user for mnist, 2000 for cifar)
    args.degree_noniid = 0.5
                               
    args.set_epochs = [20]
    args.set_num_Chosenusers = [10]
    
    args.set_degree_noniid = [0]
    args.strict_iid = True
    args.ratio_train = [0.5,0.75,1,1.25,1.5]
    args.num_experiments = 1
    
    args.iid = True
    
    main(args)