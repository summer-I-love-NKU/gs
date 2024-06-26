

import os

import torch
from torch import nn
import random
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Module
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

class Args:
    pass


class GCN(Module):
    def __init__(self,input_size,hidden_size,output_size,embedding_size,dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, embedding_size)
        self.linear=nn.Linear(embedding_size,output_size)
        self.dropout=dropout
        self.embedding=None

    def forward(self,x,edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        self.embedding=x
        x=self.linear(x)

        return F.log_softmax(x, dim=-1)
    def get_embedding(self):
        return self.embedding


def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def get_label_feature(args):
    if args.label_type=='000':
        pass
    elif args.label_type=='class':
        labels = torch.LongTensor(args.node_label_list)
    elif args.label_type == '':
        labels=None
    else:
        print('error label_type')
        exit()

    if args.features_type == 'eye':
        features = torch.eye(args.numNodes)
    elif args.features_type == 'eig':
        # eig函数的后缀功能：  h：对称矩阵 s：稀疏矩阵
        args.lm_eigvecs_maxiter=1000000
        args.lm_eigvecs_tol=1e-6
        _, Z = eigsh(args.Acsr, k=args.eigdim, which='LM', maxiter=args.lm_eigvecs_maxiter, tol=args.lm_eigvecs_tol)
        features = torch.FloatTensor(Z)
    else:
        print('error features_type')
        exit()

    # b if a is None else a, b
    # Out[5]: (1, 1)
    # b if a is None else (a, b)
    # Out[6]: 1
    return features if labels is None else (labels,features)

def args_gcn(args):
    args.device = 'cpu'
    args.hidden_size = 16
    args.lr = 0.01
    args.weight_decay = 5e-4
    args.epochs = 300
    args.patience = 100
    return args


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def create_gcncd(args):
    labels,features=get_label_feature(args)
    input_size = features.shape[1]
    output_size=labels.max().item()+1
    embedding_size=args.dimensions
    args=args_gcn(args)
    set_seed(args)

    len_labels=len(labels)
    train_mask=range(0,int(0.8*len_labels))
    valid_mask = range(int(0.8 * len_labels),int(1*len_labels))
    test_mask = range(int(0 * len_labels),  int(1*len_labels))
    # train_mask=range(0,300)
    # valid_mask=range(300,400)
    # test_mask=range(400,986)


    def f_nll_losss(output,labels):
        return F.nll_loss(output, labels)

    # def sim_cd(output,labels):
    #     d=np.zeros((labels.shape[0],labels.shape[0]))
    #     for i,v in enumerate(labels):
    #         d[i]=v==labels
    #     d[d == 0] = -2
    #     similarity = cosine_similarity(output.detach().numpy())
    #     to_maximize=torch.tensor(d * similarity)
    #     to_maximize = torch.sum(to_maximize)
    #     return -to_maximize


    loss_function=f_nll_losss

    model = GCN(input_size, args.hidden_size, output_size,embedding_size).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    min_loss=1e9
    patience_num=0
    best_output = None
    for epoch in range(args.epochs):
        # t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features,args.edge_index)
        loss_train = loss_function(output[train_mask],labels[train_mask])
        acc_train = accuracy(output[train_mask], labels[train_mask])
        # loss_train.requires_grad_(True)
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = model(features, args.edge_index)
        loss_valid = loss_function(output[valid_mask], labels[valid_mask])
        acc_valid = accuracy(output[valid_mask], labels[valid_mask])
        print('Epoch: {:04d}'.format(epoch + 1)+'\tloss_train: {:.4f}'.format(loss_train.item())+'\tacc_train: {:.4f}'.format(acc_train.item())+'\tloss_val: {:.4f}'.format(loss_valid.item())+'\tacc_val: {:.4f}'.format(acc_valid.item()))
        if loss_valid<min_loss:
            min_loss=loss_valid
            patience_num=0
            best_output=model.get_embedding()
        else:
            patience_num+=1
            if patience_num > args.patience:
                print('Epoch: {:04d}'.format(epoch + 1)+'  early stop!!!')
                break

    model.eval()
    output = model(features,args.edge_index)
    loss_test = F.nll_loss(output[test_mask], labels[test_mask])
    acc_test = accuracy(output[test_mask], labels[test_mask])
    print("Test set results:" +
                "\tloss= {:.4f}".format(loss_test.item()) +
                "\taccuracy= {:.4f}".format(acc_test.item()))

    best_output=best_output.detach().numpy()
    return best_output
