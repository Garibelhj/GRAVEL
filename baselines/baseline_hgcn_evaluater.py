
import argparse, gc
import numpy as np
import pandas as pd
import time
from functools import partial
import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F
# import dgl.multiprocessing as mp
# from dgl.multiprocessing import Queue
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import dgl
from dgl import DGLGraph
from functools import partial
#from utils.utils import earlyStopping
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
import sys
sys.path.append('/home/hongjiegu/projects/model')
sys.path.append('/home/hongjiegu/projects/GRAVEL/baselines')#dgl库，内置数据集
from HGAFA_V2 import RelGraphEmbedLayer
from HGT import HGT
import math
from dgl.nn.pytorch import TypedLinear
import tqdm
import os
import warnings
from sklearn.model_selection import train_test_split

import torch.multiprocessing as mp
import torch as th
import torch.nn as nn
import dgl
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import TypedLinear
import json
from sklearn.metrics import precision_recall_curve
import argparse
import numpy as np

import torch.nn.functional as F
import dgl

from rphgnn.callbacks import EarlyStoppingCallback, LoggingCallback, TensorBoardCallback

import argparse, gc
import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
import argparse
import warnings
import datetime
import shortuuid
import numpy as np
import torch as th
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import time
import random
import torch.nn.functional as F
import dgl
import torch
import torchmetrics
from rphgnn.layers.rphgnn_encoder import RpHGNNEncoder
import time

import dgl
from rphgnn.utils.torch_data_utils import NestedDataLoader

import os

#dgl库，内置数据集

import warnings
from sklearn.model_selection import train_test_split


from rphgnn.layers.rphgnn_pre import rphgnn_propagate_and_collect
from rphgnn.utils.nested_data_utils import gather_h_y, nested_gather, nested_map
# from utils import gen_norm,score
from SimpleHGN_standalone import SimpleHGN

warnings.filterwarnings("ignore")



def score(logits, labels, thredhold):
    Softmax = nn.Softmax(dim=1)
    # 确保labels和logits在同一设备上
    if labels.device != logits.device:
        labels = labels.to(logits.device)
    
    true_labels =labels[((labels==0)|(labels==1))]
    logits = logits[((labels==0)|(labels==1))]
    logits = Softmax(logits)
    prob = logits[:,1].cpu().detach().numpy()
    prediction = (prob>thredhold)
    true_labels = true_labels.long().cpu().numpy()  
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
    FDR,TPR,threshold = precision_recall_curve(true_labels,prob)
    FDR = 1-FDR
    return f1, macro,roc_auc,recall,FDR,TPR,threshold
def evaluate_simplehgn(model, embed_layer, val_loader,test_loader, node_feats, inv_target):
    model.eval()
    embed_layer.eval()
    val_logits = []
    val_seeds = []
    
    with th.no_grad():
        for input_nodes, seeds, blocks in val_loader:
            seeds = seeds.long().cpu()
            seeds = inv_target[seeds]
            
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'],
                                node_feats)
            logits = model.forward(blocks, feats)
            
            val_logits.append(logits['domain'].cpu())
            val_seeds.append(seeds.cpu())
    
    val_logits = th.cat(val_logits, dim=0)
    print(val_logits.shape)
    val_seeds = th.cat(val_seeds, dim=0)
    f1, precision,roc_auc,recall,fpr,tpr,thresholds = score(val_logits, labels[val_seeds],0.5)
    print(f1,roc_auc)
    val_logits = []
    val_seeds = []
    
    with th.no_grad():
        for input_nodes, seeds, blocks in test_loader:
            seeds = seeds.long().cpu()
            seeds = inv_target[seeds]
            
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'],
                                node_feats)
            logits = model.forward(blocks, feats)
            
            val_logits.append(logits['domain'].cpu())
            val_seeds.append(seeds.cpu())
    
    val_logits = th.cat(val_logits, dim=0)
    print(val_logits.shape)
    val_seeds = th.cat(val_seeds, dim=0)
    f1, precision,roc_auc,recall,fpr,tpr,thresholds = score(val_logits, labels[val_seeds],0.5)
    print(f1,roc_auc)
    return val_logits, val_seeds
def hgt_evaluater(model, embed_layer, val_loader,test_loader, node_feats, inv_target):

    


    model.eval()
    embed_layer.eval()

    eval_logits= []
    eval_seeds = []
    with th.no_grad():
        th.cuda.empty_cache()
        for sample_data in val_loader:
            inputs, seeds, blocks = sample_data
            seeds = seeds.long().cpu()
            seeds = inv_target[seeds]
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                    blocks[0].srcdata['ntype'],
                                    blocks[0].srcdata['type_id'],
                                    node_feats).to(device)
        logits = model(blocks, feats)
        eval_logits.append(logits.cpu().detach())
        eval_seeds.append(seeds.cpu().detach()) 

    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)

    f1, precision,roc_auc,recall,fpr,tpr,thresholds = score(eval_logits, labels[eval_seeds],0.5)
    print(f1,roc_auc)

    eval_logits= []
    eval_seeds = []
    with th.no_grad():
        th.cuda.empty_cache()
        for sample_data in test_loader:
            inputs, seeds, blocks = sample_data
            seeds = seeds.long().cpu()
            seeds = inv_target[seeds]
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                    blocks[0].srcdata['ntype'],
                                    blocks[0].srcdata['type_id'],
                                    node_feats).to(device)
        logits = model(blocks, feats)
        eval_logits.append(logits.cpu().detach())
        eval_seeds.append(seeds.cpu().detach()) 

    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)
    f1, precision,roc_auc,recall,fpr,tpr,thresholds = score(eval_logits, labels[eval_seeds],0.5)
    print(f1,roc_auc)
# In[4]:



def cal_malicious_logits_center(node_feats, g, labels,inv_target):
    """
    计算模型预测正确的label=1样本的恶意中心
    """
    fanouts = [20,20,20]
    embed_layer = th.load('/home/hongjiegu/projects/GRAVEL/checkpoint/pdns-gravel-encode-embed.pt')
    # create model
    # all model params are in device.
    model = th.load('/home/hongjiegu/projects/GRAVEL/checkpoint/pdns-gravel-encoder.pt')
    
    # 获取真实标签为1的domain节点
    node_ids = th.arange(g.number_of_nodes())
    domain_mask = (g.ndata['ntype'] == 0)  # 假设domain节点的类型id为0
    domain_node_ids = node_ids[domain_mask]
    malicious_mask = (labels[domain_node_ids] == 1)  # 真实标签为1的domain节点
    malicious_domain = domain_node_ids[malicious_mask]
    
    if len(malicious_domain) == 0:
        print("Warning: No malicious samples found!")
        return None
    
    # 获取模型预测结果
    Softmax = nn.Softmax(dim=1)
    model.eval()
    embed_layer.eval()
    
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    loader = dgl.dataloading.DataLoader(
        g,
        malicious_domain,
        sampler,
        batch_size=2048,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    all_logits = []
    all_hidden_status = []
    all_seeds = []
    
    with th.no_grad():
        th.cuda.empty_cache()
        for sample_data in tqdm.tqdm(loader):
            inputs, seeds, blocks = sample_data
            seeds = seeds.long()
            seeds = inv_target[seeds]

            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'],
                                node_feats)

            logits, hidden_status = model(blocks, feats)
            all_logits.append(logits)
            all_hidden_status.append(hidden_status)
            all_seeds.append(seeds)
    
    all_logits = th.cat(all_logits)
    all_hidden_status = th.cat(all_hidden_status)
    all_seeds = th.cat(all_seeds)
    
    # 获取预测概率
    prob = Softmax(all_logits)
    predictions = (prob[:, 1] > 0.5).long()  # 预测为恶性的样本
    
    # 找出预测正确的恶意样本（真实标签为1且预测为1）
    correct_prediction_mask = (predictions == 1)  # 预测为1的样本
    correctly_predicted_malicious_hidden = all_hidden_status[correct_prediction_mask]
    
    if len(correctly_predicted_malicious_hidden) == 0:
        print("Warning: No correctly predicted malicious samples found!")
        # 如果没有正确预测的恶意样本，使用所有恶意样本
        mean_center = th.mean(all_hidden_status, dim=0)
    else:
        # 计算正确预测的恶意样本的中心
        mean_center = th.mean(correctly_predicted_malicious_hidden, dim=0)
        print(f"Found {len(correctly_predicted_malicious_hidden)} correctly predicted malicious samples out of {len(all_hidden_status)} total malicious samples")
    
    return mean_center

def get_g(benchmark):
    edge = {}
    u = {}
    v = {}
    if benchmark == 'pdns':
        data_path = '/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl'
        domain = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/domain.csv")
        ip = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/ips.csv")
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/feats/domain_feats_rand.pt')
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/feats/ip_feats_rand.pt')
        num_of_ntype = 2
        num_rels =6  
        # 设置to_be_predicted：OOD malicious样本为True

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
        print(domain.label.unique())
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
        # 设置to_be_predicted：OOD malicious样本为True

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
        ood_malicious_domain = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/iochg-dgl/ood_malicious_2022_1114_1605.csv")
        domain = pd.read_csv(data_path + '/nodes/domain_new')
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/domain_feats_rand.pt')
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/ip_feats_rand.pt')
        url_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/url_feats_rand.pt')
        file_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/file_feats_rand.pt')
        num_of_ntype = 4
        num_rels = 20
        
        # 设置to_be_predicted：OOD malicious样本为True
        domain.to_be_predicted =  domain.content.isin(ood_malicious_domain.content)
        
        # 从domain.label=0的数据中选择和to_be_predicted相同数量的数据，将to_be_predicted设置为True
        ood_count = domain.to_be_predicted.sum()
        label_0_indices = domain[domain['label'] == 0].index
        if len(label_0_indices) >= ood_count:
            # 随机选择相同数量的label=0的数据
            selected_indices = np.random.choice(label_0_indices, size=int(ood_count), replace=False)
            domain.loc[selected_indices, 'to_be_predicted'] = True
        else:
            # 如果label=0的数据不够，则全部选择
            domain.loc[label_0_indices, 'to_be_predicted'] = True
        print(domain.to_be_predicted.sum())
        
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

    elif benchmark == 'iochg_small':
        data_path ='/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2023-1212-1537-zcf'
        ood_malicious_domain = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/iochg-dgl/ood_malicious_1537.csv")
        domain = pd.read_csv(data_path + '/nodes/domain_new')
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/domain_feats_rand.pt')
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/ip_feats_rand.pt')
        url_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/url_feats_rand.pt')
        file_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/file_feats_rand.pt')
        num_of_ntype = 4
        num_rels = 20
        

        
        for i in range(1, 11):
            print("load data_" + str(i))
            edge[i] = pd.read_csv(data_path + "/edges/edges_" + str(i)+'.csv')
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
    #有标签数据的mask

    



    g.nodes['domain'].data['label'] = th.tensor(domain['label'])  # .to(device)
    g.nodes['domain'].data['train_mask'] = th.tensor(domain['train_mask_finetune']).T  # .to(device)
    g.nodes['domain'].data['test_mask'] = th.tensor(domain['test_mask']).T  # .to(device)
    g.nodes['domain'].data['val_mask'] = th.tensor(domain['val_mask']).T  # .to(device)
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
                        dtype=th.long)
    inv_target[target_idx] = th.arange(0, target_idx.shape[0],
                                    dtype=th.long)
    target_idx = target_idx  # .to(device)
    labels = g.nodes['domain'].data['label']
    test_mask = g.nodes['domain'].data['test_mask']
    val_mask = g.nodes['domain'].data['val_mask']
    train_mask = g.nodes['domain'].data['train_mask']
    train_idx = g.nodes('domain')[train_mask]
    test_idx = g.nodes('domain')[test_mask]
    val_idx = g.nodes('domain')[val_mask]
    num_classes = 2

    return hg, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target,train_idx, val_idx, test_idx, labels,g




def rphgnn_evaluater():
    squash_k = 3
    inner_k = 2
    conv_filters = 2
    num_layers_list = [2, 0, 2]
    hidden_size = 512
    merge_mode = "concat"
    input_drop_rate = 0.1 
    drop_rate = 0.4
    use_pretrain_features = False
    input_random_projection_size = None
    random_projection_align = False
    target_feat_random_project_size = None
    add_self_group = False
    y = hg.nodes['domain'].data['label'].detach().cpu().numpy()
    embedding_size = None
    batch_size = 10000
    num_epochs = 10
    patience = 10
    validation_freq = 10
    convert_to_tensor = True
    hetero_graph = hg
    target_node_type ='domain'
    feature_node_types = [0]
    (train_index, valid_index, test_index)  = (train_idx,val_idx,test_idx)
    print(train_index.shape)
    print("train_rate = {}\tvalid_rate = {}\ttest_rate = {}".format(len(train_index) / len(y), len(valid_index) / len(y), len(test_index) / len(y)))
    num_classes = 2
    torch_y = th.tensor(y).long()
    train_mask = np.zeros([len(y)])
    train_mask[train_index] = 1.0
    torch_train_mask = th.tensor(train_mask).bool()
    label_target_h_list_list = [] 
    feat_target_h_list_list, target_sorted_keys = rphgnn_propagate_and_collect(hetero_graph, 
                            squash_k, 
                            inner_k, 
                            0.0,
                            target_node_type, 
                            use_input_features=True, squash_strategy="project_norm_sum", 
                            train_label_feat=None, 
                            norm="mean",
                            squash_even_odd="all",
                            collect_even_odd="all",
                            squash_self=False,
                            target_feat_random_project_size=target_feat_random_project_size,
                            add_self_group=add_self_group
                            )  


    feat_target_h_list_list = nested_map(feat_target_h_list_list, lambda x: x.to(th.float32).to("cpu"))
    target_h_list_list = feat_target_h_list_list + label_target_h_list_list


    time_dict = {
        "start": time.time()
    }

    time_dict["pre_compute"] = time.time() - time_dict["start"]
    print("pre_compute time: ", time_dict["pre_compute"])


    multi_label = len(y.shape) > 1

    accuracy_metric = torchmetrics.Accuracy("multilabel", num_labels=int(num_classes)) if multi_label else torchmetrics.Accuracy("multiclass" if multi_label else "multiclass", num_classes=int(num_classes)) 
    metrics_dict = {
        "accuracy": accuracy_metric,
        "micro_f1": torchmetrics.F1Score(task="multilabel", num_labels=int(num_classes), average="micro") if multi_label else torchmetrics.F1Score(task="multiclass", num_classes=int(num_classes), average="micro"),
        "macro_f1": torchmetrics.F1Score(task="multilabel", num_labels=int(num_classes), average="macro") if multi_label else torchmetrics.F1Score(task="multiclass", num_classes=int(num_classes), average="macro"),
    }
    metrics_dict = {metric_name: metric.to(device) for metric_name, metric in metrics_dict.items()}
    print("create model ====")
    model = RpHGNNEncoder(
        conv_filters, 
        [hidden_size] * num_layers_list[0],
        [hidden_size] * (num_layers_list[2] - 1) + [num_classes],
        merge_mode,
        input_shape=nested_map(target_h_list_list, lambda x: list(x.size())),
        input_drop_rate=input_drop_rate,
        drop_rate=drop_rate,
        activation="prelu",
        output_activation="identity",
        metrics_dict=metrics_dict,
        multi_label=multi_label, 
        loss_func= None,
        learning_rate=3e-3, 
        scheduler_gamma=None,
        train_strategy="common", 
        num_views=1, 
        cl_rate = None

        ).to(device)
    if benchmark == 'iochg_small':
        model_name = 'iochg'+'_'+baselines+'_fs.pt'
    else:
        model_name = benchmark+'_'+baselines+'_fs.pt'
    model_name = os.path.join('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint',model_name)  
    model.load_state_dict(torch.load(model_name))
    def eval_loader(model, data_loader):
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in data_loader:
                x, y = batch
                batch_logits = model(x)
                # 使用与训练过程中evaluate方法相同的预测逻辑
                if model.multi_label:
                    batch_y_pred = (torch.sigmoid(batch_logits) > 0.5).float()
                else:
                    batch_y_pred = torch.argmax(batch_logits, dim=-1)
                
                all_labels.append(y.cpu())
                all_preds.append(batch_y_pred.cpu())
        
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()

        f1 = f1_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = float('nan')
        return f1, auc
    train_h_list_list, train_y = nested_gather([target_h_list_list, torch_y], train_index)
    valid_h_list_list, valid_y = nested_gather([target_h_list_list, torch_y], valid_index)
    test_h_list_list, test_y = nested_gather([target_h_list_list, torch_y], test_index)



    valid_data_loader =NestedDataLoader(
        [valid_h_list_list, valid_y], 
        batch_size=batch_size, shuffle=False, device=device
    )
    test_data_loader = NestedDataLoader(
        [test_h_list_list, test_y], 
        batch_size=batch_size, shuffle=False, device=device
    )
    print("Evaluating on validation set...")
    val_f1, val_auc = eval_loader(model, valid_data_loader)
    print(f"Validation F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

    print("Evaluating on test set...")
    test_f1, test_auc = eval_loader(model, test_data_loader)
    print(f"Test F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
  
def slotgan_evaluater(model,val_loader,test_loader,inv_target,e_feat,node_feats,labels):
    
    model.eval()

    eval_logits= []
    eval_seeds = []
    with th.no_grad():
        th.cuda.empty_cache()
        for input_nodes, output_nodes, blocks in val_loader:

            Softmax = nn.Softmax(dim=1)

            logits, _ = model(blocks, node_feats, e_feat)
            logits = Softmax(logits)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(inv_target[output_nodes.cpu()].cpu().detach())

    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)

    f1, precision,roc_auc,recall,fpr,tpr,thresholds = score(eval_logits, labels[eval_seeds],0.5)
    print(f1,roc_auc)
    
    model.eval()

    eval_logits= []
    eval_seeds = []
    with th.no_grad():
        th.cuda.empty_cache()
        for input_nodes, output_nodes, blocks in test_loader:

            Softmax = nn.Softmax(dim=1)

            logits, _ = model(blocks, node_feats, e_feat)
            logits = Softmax(logits)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(inv_target[output_nodes.cpu()].cpu().detach())

    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)

    f1, precision,roc_auc,recall,fpr,tpr,thresholds = score(eval_logits, labels[eval_seeds],0.5)
    print(f1,roc_auc)

def config():
    parser = argparse.ArgumentParser(description='Select baselines and benchmark')
    #     parser.add_argument('-s', '--seed', type=int, default=2,
    #                         help='Random seed')
    parser.add_argument("--baselines", type=str, default='hgt',
                        help="dropout probability")
    parser.add_argument("--benchmark", type=str, default='iochg',
                        help="number of hidden units")
    parser.add_argument("--datatype", type=str, default='iid',
                        help="number of hidden units")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    # hgt = th.load('/home/hongjiegu/projects/IOC_TIME/checkpoint/Minta_HGT.pt').to(device)
    # hgt_embed  = th.load('/home/hongjiegu/projects/IOC_TIME/checkpoint/Minta_HGT_Embedding.pt').to(device)

    args = config()
    benchmark  = args.benchmark
    baselines = args.baselines
    datatype = args.datatype
    g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg = get_g(benchmark)

    fanouts = [20,20,20]
    print(fanouts)
    sampler = dgl.dataloading.NeighborSampler(fanouts,'in')
    test_loader = dgl.dataloading.DataLoader(
        g,
        target_idx[test_idx],  # 要训练的数据
        sampler,
        device=device,
        batch_size=2048,
        shuffle=True,
        drop_last=False,
    #         pin_memory =True,
        num_workers=4)

    val_loader = dgl.dataloading.DataLoader(
        g,
        target_idx[val_idx],  # 要训练的数据
        sampler,
        device=device,
        batch_size=2048,
        shuffle=True,
        drop_last=False,    
        num_workers=4)
    if baselines == 'rphgnn':
        rphgnn_evaluater()
    elif baselines == 'hgt':
        if benchmark != 'iochg_small':                    
            model_name = f"{benchmark}-{baselines}"
            model_path = os.path.join('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint', f"{model_name}_fs.pt")
            model = torch.load(model_path)
            embed_name = f"{benchmark}-{baselines}"
            embed_path = os.path.join('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint', f"{embed_name}_fs_Embedding.pt")
            embed_layer = torch.load(embed_path)
            hgt_evaluater(model, embed_layer, val_loader,test_loader, node_feats, inv_target)
        else:
            model_name = f"{'iochg'}-{baselines}"
            model_path = os.path.join('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint', f"{model_name}.pt")
            model = torch.load(model_path)
            embed_name = f"{'iochg'}-{baselines}"
            embed_path = os.path.join('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint', f"{embed_name}_Embedding.pt")
            embed_layer = torch.load(embed_path)
            hgt_evaluater(model, embed_layer, val_loader,test_loader, node_feats, inv_target)
    elif baselines == 'simplehgn':
        if benchmark != 'iochg_small':
            in_dims = [node_feats[i].shape[1] for i in range(len(node_feats))]

            model_name = f"{benchmark}-{baselines}"
            model_path = os.path.join('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint', f"{model_name}_fs.pt")

            embed_name = f"{benchmark}-{baselines}"
            embed_path = os.path.join('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint', f"{embed_name}_fs_Embedding.pt")
            model = torch.load(model_path)
            embed_layer = torch.load(embed_path)

        else:
            model_name = f"{'iochg'}-{baselines}{'n'}"
            model_path = os.path.join('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint', f"{model_name}.pt")
            model = torch.load(model_path)
            embed_name = f"{'iochg'}-{baselines}{'n'}"
            embed_path = os.path.join('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint', f"{embed_name}_Embedding.pt")
            embed_layer = torch.load(embed_path)
        evaluate_simplehgn(model, embed_layer, val_loader,test_loader, node_feats, inv_target)
    elif baselines == 'slotgat':
        canonical_etypes = sorted(g.canonical_etypes)  # 获取所有边类型（三元组格式）
        etype2idx = {etype: idx for idx, etype in enumerate(canonical_etypes)}  # 映射到整数ID
        for etype in canonical_etypes:
            eids = g.edges(form='eid', etype=etype)  # 获取该类型所有边的ID
            g.edges[eids].data['etype'] = th.full((len(eids),), etype2idx[etype], dtype=th.int64)
        e_feat = g.edata['etype']  # 从图中提取所有边的类型特征
        e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)  # 转换为张量并移到GPU
        # 修改训练循环
        node_feats = [feats.to(device) for feats in  node_feats]
        if benchmark != 'iochg_small':
            model_name = f"{benchmark}-{baselines}"
            model_path = os.path.join('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint', f"{model_name}_fs.pt")
            model = torch.load(model_path)
        else:
            model_name = f"{'iochg'}-{baselines}"
            model_path = os.path.join('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint', f"{model_name}.pt")
            model = torch.load(model_path)
        slotgan_evaluater(model, val_loader,test_loader,inv_target,e_feat,node_feats,labels)



# 