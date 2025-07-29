from re import I
import sys
import pickle
from numpy.core.numeric import identity
sys.path.append('../../')
import time
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from utils.pytorchtools import EarlyStopping
from utils.data import load_data
from utils.tools import func_args_parse,single_feat_net,vis_data_collector,blank_profile,writeIntoCsvLogger,count_torch_tensor
#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from GNN import myGAT,changedGAT,GAT,slotGAT,LabelPropagation,MLP
import dgl
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler
from sklearn.manifold import TSNE
#import wandb
import threading
from tqdm import tqdm
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
import torch.nn.functional as F
import dgl


import argparse, gc
import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
import argparse
import warnings

import numpy as np
import torch as th
import argparse
import numpy as np
import time
import random
import torch.nn.functional as F
import dgl
import torch
import time

import dgl
import dgl
from dgl.dataloading import NeighborSampler, DataLoader
import torch as th
import torch.nn.functional as F
from tqdm import tqdm
import os

#dgl库，内置数据集

import warnings
from sklearn.model_selection import train_test_split
num_heads = 8
lr = float(1e-3)
weight_decay = float(5e-4)
hidden_dim = 32
num_layers =2
slot_aggregator = "SA"
dl_mode = 'bi'
class_num=2

def score(logits, labels, thredhold):
    Softmax = th.nn.Softmax(dim=1)
    logits = Softmax(logits)
    prob = logits[:,1].cpu().detach().numpy()
    prediction = (prob>thredhold)
    true_labels = labels.long().cpu().numpy()  
    try:
        f1 = f1_score(true_labels, prediction)
    except:
        pass

    try:
        macro_f1 = f1_score(true_labels, prediction, average='macro')
    except:
        pass


    try:
        macro = precision_score(true_labels, prediction)
        
    except:
        pass
    try:

        roc_auc = roc_auc_score(true_labels,prob)
    except ValueError:
        roc_auc = 0.5
    try:
        recall=recall_score(true_labels,prediction)
    except:
        pass

    return f1, macro,roc_auc,recall

warnings.filterwarnings("ignore")
dev_id=1
device = th.device('cuda:1'if th.cuda.is_available() else 'cpu')
class earlyStopping(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.best_auc=None

    def step(self, f1, model):
        if self.best_auc is None:
            self.best_auc = f1
        elif f1 < self.best_auc :
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # model_name = benchmark+'_'+baselines+'.pt'
            model_save_path = f"/home/hongjiegu/projects/GRAVEL/baselines/checkpoint/iochg_slotgat.pt"
            torch.save(net.state_dict(), model_save_path) 
            self.best_auc = np.max((f1, self.best_auc))
            self.counter = 0
        return self.early_stop
def get_g(benchmark):
    edge = {}
    u = {}
    v = {}
    if benchmark == 'pdns':
        data_path = '/home/hongjiegu/projects/GRAVEL/dataset/pdns_net'
        domain = pd.read_csv("/home/hongjiegu/projects/GRAVEL/PDNS-Net/data/DNS_2m/domains2.csv")
        ip = pd.read_csv("/home/hongjiegu/projects/GRAVEL/PDNS-Net/data/DNS_2m/ips.csv")
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/pdns/domain_feats_rand.pt')
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/pdns/ip_feats_rand.pt')
        num_of_ntype = 2
        num_rels =6  
        for i in range(1, 4):
            print("load data_" + str(i))
            edge[i] = pd.read_csv(data_path + "/edge_" + str(i)+'.csv')
            u[i] = edge[i]["source"].values.astype(int)
            v[i] = edge[i]["target"].values.astype(int)
            u[i] = th.from_numpy(u[i])  # .to(device)
            v[i] = th.from_numpy(v[i])  # .to(device)

        print("-------------finish dataloader------")
        graph_data = {
            ('domain', '1', 'ip'): (u[1], v[1]),
            ('domain', '2', 'domain'): (u[2], v[2]),
            ('domain', '3', 'domain'): (u[3], v[3]),
            ('ip', '_1', 'domain'): (v[1], u[1]),
            ('domain', '_2', 'domain'): (v[2], u[2]),
            ('domain', '_3', 'domain'): (v[3], u[3]),


        }
        g = dgl.heterograph(graph_data)
        g.nodes['domain'].data['feat'] = th.tensor(np.array(domain_feats)).to(th.float32)
        g.nodes['ip'].data['feat'] = th.tensor(np.array(ip_feats)).to(th.float32)
    elif benchmark == 'minta':
        data_path = '/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl'
        domain = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/domain.csv')
        ip = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/ip.csv')
        host = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/host.csv')
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/feats/domain_feats_rand.pt')
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/feats/ip_feats_rand.pt')
        host_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/feats/host_feats_rand.pt')
        num_of_ntype = 3
        num_rels =8    

        for i in range(1, 5):
            print("load data_" + str(i))
            edge[i] = pd.read_csv(data_path + "/edge_" + str(i)+'.csv')
            u[i] = edge[i]["index_x"].values.astype(int)
            v[i] = edge[i]["index_y"].values.astype(int)
            u[i] = th.from_numpy(u[i])  # .to(device)
            v[i] = th.from_numpy(v[i])  # .to(device)
            if (u[i] < 0).any() or (v[i] < 0).any():
                print(f"edge_{i} contains negative node indices!")  
        print("-------------finish dataloader------")
        graph_data = {
            ('domain', '1', 'ip'): (u[1], v[1]),
            ('domain', '2', 'domain'): (u[2], v[2]),
            ('domain', '3', 'domain'): (u[3], v[3]),
            ('domain', '4', 'host'): (u[4], v[4]),
            ('ip', '_1', 'domain'): (v[1], u[1]),
            ('domain', '_2', 'domain'): (v[2], u[2]),
            ('domain', '_3', 'domain'): (v[3], u[3]),
            ('host', '_4', 'domain'): (v[4], u[4]),

        }
        g = dgl.heterograph(graph_data)
        g.nodes['domain'].data['feat'] = th.tensor(np.array(domain_feats)).to(th.float32)
        g.nodes['ip'].data['feat'] = th.tensor(np.array(ip_feats)).to(th.float32)
        g.nodes['host'].data['feat'] = th.tensor(np.array(host_feats)).to(th.float32)
    elif benchmark == 'iochg':
        data_path ='/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2022-1114-1605-a'
        # from ogb.nodeproppred import DglNodePropPredDataset
        domain = pd.read_csv(data_path + '/nodes/domain_new')
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/domain_feats_rand.pt')
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/ip_feats_rand.pt')
        url_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/url_feats_rand.pt')
        file_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/file_feats_rand.pt')
        num_of_ntype = 4
        num_rels = 20
        for i in range(1, 11):
            print("load data_" + str(i))
            edge[i] = pd.read_csv(data_path + "/edges/edges_" + str(i))
            u[i] = edge[i]["index_x"].values.astype(int)
            v[i] = edge[i]["index_y"].values.astype(int)
            u[i] = th.from_numpy(u[i])  # .to(device)
            v[i] = th.from_numpy(v[i])  # .to(device)

        print("-------------finish dataloader------")
        graph_data = {
            ('url', '1', 'file'): (u[1], v[1]),
            ('file', '2', 'file'): (u[2], v[2]),
            ('file', '3', 'ip'): (u[3], v[3]),
            ('file', '4', 'domain'): (u[4], v[4]),
            ('file', '5', 'domain'): (u[5], v[5]),
            ('file', '6', 'url'): (u[6], v[6]),
            ('domain', '7', 'domain'): (u[7], v[7]),
            ('domain', '8', 'ip'): (u[8], v[8]),
            ('domain', '9', 'domain'): (u[9], v[9]),
            ('domain', '10', 'url'): (u[10], v[10]),

            # 点-边-点 类型
            ('file', '_1', 'url'): (v[1], u[1]),
            ('file', '_2', 'file'): (v[2], u[2]),
            ('ip', '_3', 'file'): (v[3], u[3]),
            ('domain', '_4', 'file'): (v[4], u[4]),
            ('domain', '_5', 'file'): (v[5], u[5]),
            ('url', '_6', 'file'): (v[6], u[6]),
            ('domain', '_7', 'domain'): (v[7], u[7]),
            ('ip', '_8', 'domain'): (v[8], u[8]),
            ('domain', '_9', 'domain'): (v[9], u[9]),
            ('url', '_10', 'domain'): (v[10], u[10])
        }
        g = dgl.heterograph(graph_data)
        g.nodes['domain'].data['feat'] = th.tensor(np.array(domain_feats)).to(th.float32)
        g.nodes['ip'].data['feat'] = th.tensor(np.array(ip_feats)).to(th.float32)
        g.nodes['url'].data['feat'] = th.tensor(np.array(url_feats)).to(th.float32)
        g.nodes['file'].data['feat'] = th.tensor(np.array(file_feats)).to(th.float32)  

    domain['to_be_predicted'] = domain['label']
    domain.loc[domain['to_be_predicted']<2,'to_be_predicted']  = False
    domain.loc[domain['to_be_predicted']>2,'to_be_predicted']  = True   # != 2 ???
    domain.loc[domain['to_be_predicted']==2,'to_be_predicted']  = True
    #有标签数据的mask
    domain['with_label'] = domain['label']
    domain.loc[domain['with_label']<2,'with_label']  = True
    domain.loc[domain['with_label']>=2,'with_label']  = False
    with_label_mask = th.tensor(domain['with_label'].to_numpy(dtype=bool)).T
    domain_id = th.tensor(domain.index)
    train_set, test_set = train_test_split(domain_id[with_label_mask], test_size=0.2, random_state=47)
    test_set, valid_set = train_test_split(test_set, test_size=0.5, random_state=47)
    train_set = train_set.numpy()
    test_set = test_set.numpy()
    valid_set = valid_set.numpy()
    train_set = np.sort(train_set, axis=0)
    test_set = np.sort(test_set, axis=0)
    valid_set = np.sort(valid_set,axis=0)
    j, k, m = 0, 0, 0
    train_idx = []
    test_idx = []
    valid_idx = []

    for i in range(len(domain)):
        if j < train_set.size and i == train_set[j]:
            train_idx.append(True)
            test_idx.append(False)
            valid_idx.append(False)
            j += 1
        elif k < test_set.size and i == test_set[k]:
            test_idx.append(True)
            train_idx.append(False)
            valid_idx.append(False)
            k += 1
        elif m < valid_set.size and i == valid_set[m]:
            valid_idx.append(True)
            train_idx.append(False)
            test_idx.append(False)
            m += 1
        else:
            train_idx.append(False)
            test_idx.append(False)
            valid_idx.append(False)

    domain['with_label_train'] = train_idx
    domain['with_label_test'] = test_idx
    domain['with_label_valid'] = valid_idx
    g.nodes['domain'].data['label'] = th.tensor(domain['label'])  # .to(device)
    g.nodes['domain'].data['train_mask'] = th.tensor(domain['with_label_train']).T  # .to(device)
    g.nodes['domain'].data['test_mask'] = th.tensor(domain['with_label_test']).T  # .to(device)
    g.nodes['domain'].data['val_mask'] = th.tensor(domain['with_label_valid']).T  # .to(device)

    hg = dgl.to_homogeneous(g, ndata=['label'])

    hg.ndata['ntype'] = hg.ndata[dgl.NTYPE]
    hg.edata['etype'] = hg.edata[dgl.ETYPE]
    hg.ndata['type_id'] = hg.ndata[dgl.NID]


    node_feats = []
    for ntype in g.ntypes:
        feat = g.nodes[ntype].data['feat']
        node_feats.append(feat.share_memory_())

        node_ids = th.arange(hg.number_of_nodes())
    category_id = 0
    node_tids = hg.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]
    inv_target = th.empty(node_ids.shape,
                        dtype=node_ids.dtype)
    inv_target[target_idx] = th.arange(0, target_idx.shape[0],
                                    dtype=inv_target.dtype)
    target_idx = target_idx  # .to(device)

    labels = g.nodes['domain'].data['label']
    train_mask = g.nodes['domain'].data['train_mask']
    test_mask = g.nodes['domain'].data['test_mask']
    val_mask = g.nodes['domain'].data['val_mask']
    train_idx = g.nodes('domain')[train_mask]
    test_idx = g.nodes('domain')[test_mask]
    val_idx = g.nodes('domain')[val_mask]
    num_classes = 2

    return hg, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,g
g,  node_feats,num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg = get_g('minta')

g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)

features = node_feats
dataRecorder={"meta":{
    "getSAAttentionScore":True,
},"data":{},"status":"None"}
heads = [num_heads] * num_layers + [1]
GNN = slotGAT
vis_data_saver=vis_data_collector() 
in_dims = [features.shape[1] for features in node_feats]
num_ntypes=len(in_dims)
g.num_ntypes=num_ntypes

fargs,fkargs=func_args_parse(g,16, num_rels, in_dims, hidden_dim, num_classes, num_layers, heads, F.elu, 0.5,0.5, 0.05, True, 0.05,num_ntype=num_of_ntype, eindexer=None, aggregator=slot_aggregator ,SAattDim=3,dataRecorder=dataRecorder,vis_data_saver=vis_data_saver)
net_wrapper=single_feat_net
net=net_wrapper(GNN,*fargs,**fkargs)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
net = net.to(device)
fanouts = [20,20,20]
print(fanouts)
sampler = dgl.dataloading.NeighborSampler(fanouts,'in')

# 创建训练数据加载器
train_loader = DataLoader(
    g,
    target_idx[train_idx.to('cpu')],    # DGL的采样器需要节点在CPU上
    sampler,
    device=device,          # 最终数据转移到该设备
    batch_size=2048,         # 根据GPU内存调整
    shuffle=True,
    drop_last=False,
    num_workers=4           # 并行工作进程数
)
canonical_etypes = sorted(g.canonical_etypes)  # 获取所有边类型（三元组格式）
etype2idx = {etype: idx for idx, etype in enumerate(canonical_etypes)}  # 映射到整数ID
for etype in canonical_etypes:
    eids = g.edges(form='eid', etype=etype)  # 获取该类型所有边的ID
    g.edges[eids].data['etype'] = th.full((len(eids),), etype2idx[etype], dtype=th.int64)
e_feat = g.edata['etype']  # 从图中提取所有边的类型特征
e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)  # 转换为张量并移到GPU
# 修改训练循环
best_val_loss = float('inf')
node_feats = [feats.to(device) for feats in  node_feats]

def evaluate(model, graph, val_idx,target_idx,inv_target,e_feat,node_feats,labels):
    model.eval()
    eval_logits = []
    eval_seeds = []
    val_loader = DataLoader(
        graph,
        target_idx[val_idx.to('cpu')],
        sampler,  # 可以使用不同的采样器
        device=device,
        batch_size=2048,  # 验证可以使用更大的batch
        shuffle=False
    )
    
    total_loss = 0
    with th.no_grad():
        for input_nodes, output_nodes, blocks in val_loader:

            Softmax = nn.Softmax(dim=1)

            logits, _ = model(blocks, node_feats, e_feat)
            logits = Softmax(logits)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(inv_target[output_nodes.cpu()].cpu().detach())
    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)
    return eval_logits,eval_seeds


stopper = earlyStopping(5)

for epoch in tqdm(range(500)):
    net.train()
    total_loss = 0
    total_correct = 0
    
    for input_nodes, output_nodes, blocks in tqdm(train_loader):
        # 1. 准备批量数据 -------------------------------------------------
        # 获取当前批次的节点特征
        # input_features =[node_feats[blocks[0].srcdata[dgl.NID].cpu()].to(device)]
        # # 获取边特征
        # e_feat = e_feat[blocks[0].edata['_ID']]
        

        # 2. 前向传播 -----------------------------------------------------
        optimizer.zero_grad()
        
        logits, _ = net(blocks, node_feats, e_feat)
        
        # 3. 计算损失 -----------------------------------------------------
        # 只计算输出节点的损失
        batch_labels = labels[inv_target[output_nodes.cpu()]].to(device)
        loss = F.cross_entropy(logits, batch_labels)
        
        # 4. 反向传播 -----------------------------------------------------
        loss.backward()
        optimizer.step()
        
        # 5. 记录指标 -----------------------------------------------------
        total_loss += loss.item() * len(output_nodes)
        preds = logits.argmax(dim=1)
        total_correct += (preds == batch_labels).sum().item()
    
    # 计算epoch指标
    avg_loss = total_loss / len(train_idx)
    acc = total_correct / len(train_idx)
    
    # 打印训练指标
    print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    # 验证步骤（可选）
    if (epoch + 1) % 10 == 0:
        eval_logits,eval_seeds = evaluate(net, g, val_idx,target_idx,inv_target,e_feat,node_feats,labels)
        f1, precision, roc_auc, recall = score(eval_logits,labels[eval_seeds].to(device),0.5)
        print(f1,precision,recall)
# 评估函数示例


        early_stop = stopper.step(f1,net)
        if early_stop:
            break




