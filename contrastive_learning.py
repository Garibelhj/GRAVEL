  #-*-encoding:utf-8-*-

"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn
Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched

"""
import argparse, gc
import numpy as np
import pandas as pd
import time
from functools import partial
import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import dgl
from dgl import DGLGraph
from functools import partial
import sys
sys.path.append('/home/hongjiegu/projects/model')


from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve,accuracy_score
import matplotlib.pyplot as plt
#dgl库，内置数据集
#from ..model import linear
from HGAFA_V2 import RelGraphEmbedLayer
from HGAFA_V2 import RelGraphConv,EntityClassify

from sklearn.model_selection import train_test_split
from dgl import DropEdge
import tqdm
import os
import warnings
import torch.multiprocessing as mp
from dgl.nn import LabelPropagation
import math
warnings.filterwarnings("ignore")


TAU=0.1
data_path='/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2022-1114-1605-a'

domain = pd.read_csv(data_path + '/nodes/domain_new.csv')
file = pd.read_csv(data_path + '/nodes/file_nodes.csv')
url = pd.read_csv(data_path + '/nodes/url_nodes.csv')
ip = pd.read_csv(data_path + '/nodes/ip_new')
domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/domain_feats_rand.pt')
ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/ip_feats_rand.pt')
url_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/url_feats_rand.pt')
file_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/file_feats_rand.pt')
test_data = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/iochg-dgl/ood_malicious_2022_1114_1605.csv',header=0)

# data_path='/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2023-1212-1537-zcf'
# test_data = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/ood_malicious_1537.csv')

# # from ogb.nodeproppred import DglNodePropPredDataset
# domain = pd.read_csv(data_path + '/nodes/domain_new')
# domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/domain_feats_rand.pt')
# ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/ip_feats_rand.pt')
# url_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/url_feats_rand.pt')
# file_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/file_feats_rand.pt')
# file = pd.read_csv(data_path + '/nodes/file_nodes')
# url = pd.read_csv(data_path + '/nodes/url_nodes')
# ip = pd.read_csv(data_path + '/nodes/ip_new')
domain.to_be_predicted =  domain.content.isin(test_data.content)
#有标签数据的mask
domain['with_label'] = domain['label']
domain.loc[domain['with_label']<2,'with_label']  = True
domain.loc[domain['with_label']>=2,'with_label']  = True
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
domain.to_csv(data_path + '/nodes/domain_new',index=False)
file['label1'] = file['label']
file.loc[file['label1']==1,'label']  = 0
file.loc[file['label1']==0,'label']  = 1

#待预测的数据打上mask，to_be_predicted字段只有label=2时才为True
file['to_be_predicted'] = file['label']
file.loc[file['to_be_predicted']<2,'to_be_predicted']  = False
file.loc[file['to_be_predicted']>2,'to_be_predicted']  = False   # != 2 ???
file.loc[file['to_be_predicted']==2,'to_be_predicted']  = True
#有标签数据的mask
file['with_label'] = file['label']
file.loc[file['with_label']<2,'with_label']  = True
file.loc[file['with_label']>=2,'with_label']  = False
with_label_mask = th.tensor(file['with_label'].to_numpy(dtype=bool)).T
file_id = th.tensor(file.index)
train_set, test_set = train_test_split(file_id[with_label_mask], test_size=0.2, random_state=47)

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

for i in range(len(file)):
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

file['with_label_train'] = train_idx
file['with_label_test'] = test_idx
file['with_label_valid'] = valid_idx
file.to_csv(data_path + '/nodes/file_nodes',index=False)
url['label1'] = url['label']
url.loc[url['label1']==1,'label']  = 0
url.loc[url['label1']==0,'label']  = 1

#待预测的数据打上mask，to_be_predicted字段只有label=2时才为True
url['to_be_predicted'] = url['label']
url.loc[url['to_be_predicted']<2,'to_be_predicted']  = False
url.loc[url['to_be_predicted']>2,'to_be_predicted']  = False   # != 2 ???
url.loc[url['to_be_predicted']==2,'to_be_predicted']  = True
#有标签数据的mask
url['with_label'] = url['label']
url.loc[url['with_label']<2,'with_label']  = True
url.loc[url['with_label']>=2,'with_label']  = False
with_label_mask = th.tensor(url['with_label'].to_numpy(dtype=bool)).T
url_id = th.tensor(url.index)
train_set, test_set = train_test_split(url_id[with_label_mask], test_size=0.2, random_state=47)
test_set, valid_set = train_test_split(test_set, test_size=0.5, random_state=47)
train_set = train_set.numpy()
test_set = test_set.numpy()
valid_set = valid_set.numpy()
train_set = np.sort(train_set, axis=0)
test_set = np.sort(test_set, axis=0)
valid_set = np.sort(valid_set, axis = 0)
j, k, m = 0, 0, 0
train_idx = []
test_idx = []
valid_idx = []

for i in range(len(url)):
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
url['with_label_train'] = train_idx
url['with_label_test'] = test_idx
url['with_label_valid'] = valid_idx
url.to_csv(data_path + '/nodes/url_nodes',index=False)

def get_g(data_path):
    edge = {}
    u = {}
    v = {}
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
    print("-------------finish graph making------")


    g.nodes['domain'].data['label'] = th.tensor(domain['label'])  # .to(device)
    g.nodes['domain'].data['train_mask'] = th.tensor(domain['with_label_train']).T  # .to(device)
    g.nodes['domain'].data['test_mask'] = th.tensor(domain['to_be_predicted']).T  # .to(device)
    g.nodes['domain'].data['val_mask'] = th.tensor(domain['with_label_valid']).T  # .to(device)

    
    g.nodes['file'].data['label'] = th.tensor(file['label'])  # .to(device)    # wy, 4/3
    g.nodes['file'].data['train_mask'] = th.tensor(file['with_label_train']).T  # .to(device)     # wy, 4/3
    g.nodes['file'].data['test_mask'] = th.tensor(file['with_label_test']).T  # .to(device)     # wy, 4/3
    g.nodes['file'].data['val_mask'] = th.tensor(file['with_label_valid']).T
    
    g.nodes['url'].data['label'] = th.tensor(url['label'])  # .to(device)    # wy, 4/3
    g.nodes['url'].data['train_mask'] = th.tensor(url['with_label_train']).T  # .to(device)     # wy, 4/3
    g.nodes['url'].data['test_mask'] = th.tensor(url['with_label_test']).T  # .to(device)     # wy, 4/3
    g.nodes['url'].data['val_mask'] = th.tensor(url['with_label_valid']).T
    g.nodes['ip'].data['label'] = th.zeros(g.number_of_nodes('ip')).long()
    g.nodes['ip'].data['train_mask'] = th.full((g.number_of_nodes('ip'),), False)
    g.nodes['ip'].data['test_mask'] = th.full((g.number_of_nodes('ip'),), False)
    g.nodes['ip'].data['val_mask'] = th.full((g.number_of_nodes('ip'),), False)
    g.nodes['domain'].data['feats'] = th.tensor(np.array(domain_feats)).to(th.float32)
    g.nodes['ip'].data['feats'] = th.tensor(np.array(ip_feats)).to(th.float32)
    g.nodes['url'].data['feats'] = th.tensor(np.array(url_feats)).to(th.float32)
    g.nodes['file'].data['feats'] = th.tensor(np.array(file_feats)).to(th.float32)  

    # 为节点加入特征
    node_dict = {}
    edge_dict = {}
    for ntype in g.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in g.etypes:
        edge_dict[etype] = len(edge_dict)
        g.edges[etype].data['id'] = (th.ones(g.number_of_edges(etype), dtype=th.long) * edge_dict[etype])  # .to(device)
    
    # Random initialize input feature
    for ntype in g.ntypes:
        if ntype == 'domain' or ntype == 'url':
            continue
        emb = th.nn.Parameter(th.Tensor(g.number_of_nodes(ntype), 7), requires_grad=False)
        th.nn.init.xavier_uniform_(emb)
        g.nodes[ntype].data['inp'] = emb  # .to(device)

    print('feature engineering finished')
    for ntype in g.ntypes:
        emb = th.nn.Parameter(th.Tensor(g.number_of_nodes(ntype), 64), requires_grad=False)
        th.nn.init.xavier_uniform_(emb)
        g.nodes[ntype].data['inp'] = emb  # .to(device)
        
    feats = True
    node_feats = []
    for ntype in g.ntypes:
        if len(g.nodes[ntype].data) == 0 or feats is False:
            node_feats.append(g.number_of_nodes(ntype))
        else:
            #assert len(g.nodes[ntype].data) == 1
            feat = g.nodes[ntype].data.pop('feats')
            node_feats.append(feat.share_memory_())
    # 转为同构计算返回值
    # 保留每个节点的label


    hg = dgl.to_homogeneous(g,ndata=['label','test_mask','val_mask'])
    mask = hg.ndata['label'] <2
    mask[hg.ndata['test_mask']==True] = False
    mask[hg.ndata['val_mask']==True] = False

    _mask = ~mask
    hg.ndata['label'][_mask] = 0 
    _labels = hg.ndata['label']

    label_propagation = LabelPropagation(k=10, alpha=0.5, clamp=True, normalize=True)
    _new_labels = label_propagation(hg, _labels, mask)
    hg.ndata['label'] = _new_labels
    hg.ndata['ntype'] = hg.ndata[dgl.NTYPE]
    hg.edata['etype'] = hg.edata[dgl.ETYPE]
    hg.ndata['type_id'] = hg.ndata[dgl.NID]


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
    num_of_ntype = 4
    num_rels = 20
    return hg, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,g


def get_g2(benchmark,datatype):
    edge = {}
    u = {}
    v = {}
    if benchmark == 'pdns':
        data_path = '/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl'
        domain = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/domains2.csv")
        ip = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/ips.csv")
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/feats/domain_feats_rand.pt')
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/feats/ip_feats_rand.pt')

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
        g.nodes['domain'].data['label'] = th.tensor(domain['label'])  # .to(device)
        g.nodes['ip'].data['label'] = th.full(
            (g.number_of_nodes('ip'),), 2, dtype=th.long
        )
        g.nodes['ip'].data['test_mask'] = th.full((g.number_of_nodes('ip'),), False)
        g.nodes['ip'].data['val_mask'] = th.full((g.number_of_nodes('ip'),), False)

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
        g.nodes['domain'].data['label'] = th.tensor(domain['label'])  # .to(device)

        g.nodes['domain'].data['feat'] = th.tensor(np.array(domain_feats)).to(th.float32)
        g.nodes['ip'].data['label'] = th.full(
            (g.number_of_nodes('ip'),), 2, dtype=th.long
        )
        g.nodes['ip'].data['feat'] = th.tensor(np.array(ip_feats)).to(th.float32)
        g.nodes['host'].data['feat'] = th.tensor(np.array(host_feats)).to(th.float32)
        g.nodes['host'].data['label'] = th.full(
            (g.number_of_nodes('host'),), 2, dtype=th.long
        )
        g.nodes['ip'].data['test_mask'] = th.full((g.number_of_nodes('ip'),), False)
        g.nodes['ip'].data['val_mask'] = th.full((g.number_of_nodes('ip'),), False)
        g.nodes['host'].data['test_mask'] = th.full((g.number_of_nodes('host'),), False)
        g.nodes['host'].data['val_mask'] = th.full((g.number_of_nodes('host'),), False)
    elif benchmark == 'iochg':
        data_path ='/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2022-1114-1605-a'
        # from ogb.nodeproppred import DglNodePropPredDataset
        domain = pd.read_csv(data_path + '/nodes/domain_new')
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/domain_feats_rand.pt')
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/ip_feats_rand.pt')
        url_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/url_feats_rand.pt')
        file_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/file_feats_rand.pt')
        test_data = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/ood_malicious_2022_1114_1605.csv',header=0)

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


    #有标签数据的mask
    domain.loc[domain['label'] > 1, 'train_mask_zs'] = True
    g.nodes['domain'].data['label'] = th.tensor(domain['label'])  # .to(device)
    g.nodes['domain'].data['train_mask'] = th.tensor(domain['train_mask_zs']).T  # .to(devicez
    g.nodes['domain'].data['val_mask'] = th.tensor(domain['val_mask']).T  # .to(device)

    g.nodes['domain'].data['test_mask'] = th.tensor(domain['test_mask']).T

    hg = dgl.to_homogeneous(g,ndata=['label','test_mask','val_mask'])
    mask = hg.ndata['label'] <2
    mask[hg.ndata['test_mask']==True] = False
    mask[hg.ndata['val_mask']==True] = False

    _mask = ~mask
    hg.ndata['label'][_mask] = 0 
    _labels = hg.ndata['label']

    label_propagation = LabelPropagation(k=10, alpha=0.5, clamp=True, normalize=True)
    _new_labels = label_propagation(hg, _labels, mask)
    hg.ndata['label'] = _new_labels
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


def cal_malicious_logits_center(node_feats, g, labels):
    """
    计算模型预测正确的label=1样本的恶意中心
    """
    fanouts = [20,20,20]
    embed_layer = th.load('/home/hongjiegu/projects/GRAVEL/checkpoint/minta-gravel-encoder-embed.pt')
    # create model
    # all model params are in device.
    model = th.load('/home/hongjiegu/projects/GRAVEL/checkpoint/minta-gravel-encoder.pt')
    
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


def predict_all_nodes(node_feats,g,labels):
    node_ids = th.arange(g.number_of_nodes())
    domain_mask = (g.ndata['ntype'] == 0)  # 假设domain节点的类型id为0
    domain_node_ids = node_ids[domain_mask]
    Softmax = nn.Softmax(dim=1)
    logits_batch = []

    fanouts = [20,20,20]
    embed_layer = th.load('/home/hongjiegu/projects/GRAVEL/checkpoint/minta-gravel-encoder-embed.pt')
    # create model
    # all model params are in device.
    model = th.load('/home/hongjiegu/projects/GRAVEL/checkpoint/minta-gravel-encoder.pt')
    model.eval()
    embed_layer.eval()
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    loader = dgl.dataloading.DataLoader(
        g,
        domain_node_ids,
        sampler,
        batch_size=2048,
        shuffle=False,
        drop_last=False,
        num_workers= 0
        )     

    with th.no_grad():
        th.cuda.empty_cache()
            #neg_seeds.append(seeds)
        for sample_data in tqdm.tqdm(loader):
            inputs, seeds, blocks = sample_data
            seeds = seeds.long()
            seeds = inv_target[seeds]

            # for block in blocks:
            #     gen_norm(block)

            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'],
                                node_feats)

            logits,hidden_status = model(blocks, feats)
            prob = Softmax(logits)
            logits_batch.append(prob)
    logits_batch = th.cat(logits_batch)

    # 获取domain节点的真实标签
    
    # 创建最终的标签张量，初始化为预测的伪标签
    final_labels = logits_batch.clone()
    
    # 对于label=0或1的domain节点，使用真实标签
    # 将真实标签转换为one-hot编码格式
    for i, label in enumerate(labels):
        if label == 0:
            final_labels[i] = th.tensor([1.0, 0.0])  # [P(label=0), P(label=1)]
        elif label == 1:
            final_labels[i] = th.tensor([0.0, 1.0])  # [P(label=0), P(label=1)]
        # 对于label>2的节点，保持预测的伪标签不变

    return final_labels


class earlyStopping(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.best_auc=0
    def step(self, auc, model,embed_layer):
        if auc < self.best_auc :
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            th.save(model,'/home/hongjiegu/projects/GRAVEL/checkpoint/minta-gravel-contrastive.pt')
            th.save(embed_layer,'/home/hongjiegu/projects/GRAVEL/checkpoint/minta-gravel-contrastive-embed.pt')


            self.best_auc = np.max((auc, self.best_auc))
            self.counter = 0
        return self.early_stop
def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
# 计算每个点的度作为一个特征
def gen_norm(g):
    _, v, eid = g.all_edges(form='all')
    _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
    degrees = count[inverse_index]
    norm = th.ones(eid.shape[0], device=eid.device) / degrees
    norm = norm.unsqueeze(1)
    g.edata['norm'] = norm

    norm_w = th.ones(eid.shape[0], device=eid.device) / 2

    g.edata['norm_w'] = norm_w
    
    data=np.vstack((g.edata['norm_w'], g.edata['etype']))
    df = pd.DataFrame(np.transpose(data), columns=['norm_w','etype'])
    df.loc[df['etype']==16, 'norm_w']=0

    norm_w = th.tensor(np.array(df['norm_w']))
    norm_w = norm_w.unsqueeze(1)
    g.edata['norm_w'] = norm_w
def score(logits, labels, thredhold):
    Softmax = nn.Softmax(dim=1)
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
    try:
        accuracy = accuracy_score(true_labels, prediction)
    except:
        pass
    FDR,TPR,threshold = precision_recall_curve(true_labels,prob)
    FDR = 1-FDR
    return f1, macro,roc_auc,recall,FDR,TPR,threshold,accuracy
def evaluate(model, embed_layer,eval_loader, node_feats, inv_target):
    model.eval()
    embed_layer.eval()
    eval_logits = []
    eval_seeds = []

    with th.no_grad():
        th.cuda.empty_cache()
        # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
        for sample_data in tqdm.tqdm(eval_loader):
            inputs, seeds, blocks = sample_data
            seeds = seeds.long()
            seeds = inv_target[seeds]

            # for block in blocks:
            #     gen_norm(block)

            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'],
                                node_feats)

            logits,hidden_status = model(blocks, feats)

            eval_logits.append(hidden_status.cpu().detach())
            eval_seeds.append(seeds.cpu().detach())

    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)
    
    return eval_logits, eval_seeds

    
def my_loss(hidden_status,C_mal,pseudo_labels,seeds):
    C_mal = C_mal.to(device)

    pseudo_labels = pseudo_labels[:,1].to(device)
    pseudo_labels = pseudo_labels[seeds]

    squared_diff = (hidden_status-C_mal)**2
    R = squared_diff.sum(dim=1)
    y_i = 2*pseudo_labels-1
    global_distance = R**y_i
    global_distance = global_distance.mean()
    
    # 计算相同标签样本之间的平均距离并最小化
    batch_size = hidden_status.size(0)
    if batch_size > 1:
        # 分离label=1和label=0的样本
        label_1_mask = (pseudo_labels == 1)
        label_0_mask = (pseudo_labels == 0)
        
        label_1_samples = hidden_status[label_1_mask]
        label_0_samples = hidden_status[label_0_mask]
        
        # 计算label=1样本间的平均距离
        if len(label_1_samples) > 1:
            label_1_dist_matrix = th.cdist(label_1_samples, label_1_samples, p=2)
            # 获取上三角矩阵（避免重复计算）
            upper_tri = th.triu(label_1_dist_matrix, diagonal=1)
            # 计算平均距离
            label_1_mean_dist = th.mean(upper_tri)
        else: 
            label_1_mean_dist = th.tensor(0.0, device=device)
        
        # 计算label=0样本间的平均距离
        if len(label_0_samples) > 1:
            label_0_dist_matrix = th.cdist(label_0_samples, label_0_samples, p=2)
            # 获取上三角矩阵（避免重复计算）
            upper_tri = th.triu(label_0_dist_matrix, diagonal=1)
            # 计算平均距离
            label_0_mean_dist = th.mean(upper_tri)
        else:
            label_0_mean_dist = th.tensor(0.0, device=device)
        
        # 计算不同标签样本间的平均距离
        if len(label_1_samples) > 0 and len(label_0_samples) > 0:
            # 计算label=1和label=0样本间的距离矩阵
            cross_dist_matrix = th.cdist(label_1_samples, label_0_samples, p=2)
            # 计算平均距离
            cross_mean_dist = th.mean(cross_dist_matrix)
        else:
            cross_mean_dist = th.tensor(0.0, device=device)
        
        # 将平均距离作为损失项（最小化相同标签平均距离，最大化不同标签平均距离）
        cluster_loss = label_1_mean_dist + label_0_mean_dist - cross_mean_dist
        
        # 组合原始损失和聚类损失
        alpha = 0.5  # 聚类损失的权重
        total_loss = global_distance + alpha * cluster_loss
    else:
        total_loss = global_distance
    
    return total_loss
def score_auc(logits,label,C):
    C = C.to(device)
    logits = logits.to(device)
    labels = label[((label==0)|(label==1))]
    squared_diff = (logits-C)**2
    R = -squared_diff.sum(dim=1)
    R = R[((label==0)|(label==1))]
    R = R.cpu().detach()
    labels=labels.cpu()
    #fpr,tpr,thresholds = roc_curve(labels,R)
    precision,recall,threshold = precision_recall_curve(labels,R)
    roc_auc = roc_auc_score(labels,R)
    return roc_auc,precision,recall,threshold









if __name__ == '__main__':
    g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg = get_g2(
        'minta','iid')
    # g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg = get_g(data_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    gen_norm(g)
    pse_label = predict_all_nodes(node_feats,g,labels)
    # th.save(pse_label,'/home/hongjiegu/projects/GRAVEL/dataset/pse_label.pt')
    # pse_label =th.load('/home/hongjiegu/projects/GRAVEL/dataset/pse_label.pt')
    stopper = earlyStopping(5)
    C_mal = cal_malicious_logits_center(node_feats,g,labels)
    # th.save(C_mal,'/home/hongjiegu/projects/GRAVEL/dataset/C_mal.pt')
    # C_mal =th.load('/home/hongjiegu/projects/GRAVEL/dataset/C_mal.pt')
    node_tids = g.ndata[dgl.NTYPE]
    embedding = RelGraphEmbedLayer(device,
                                    num_of_ntype,
                                    node_feats,
                                    64,
                                    dgl_sparse=False).to(device)
    # create model
    # all model params are in device.
    model = EntityClassify(device,
                        g.number_of_nodes(),
                        64,
                        num_classes,
                        num_rels,
                        num_bases=2,
                        num_hidden_layers=2,
                        dropout=0,
                        use_self_loop=True,
                        layer_norm=False).to(device)
    dense_params = list(model.parameters())
    optimizer = th.optim.Adam(dense_params, lr=0.0001, weight_decay=1e-5)

    embs = list(embedding.embeds.parameters())    
    emb_optimizer = th.optim.Adam(embs, lr=0.0001) if len(embs) > 0 else None
    sampler = dgl.dataloading.NeighborSampler([20,20,20],'in')
    loader = dgl.dataloading.DataLoader(
        g,
        target_idx[train_idx],  # 要训练的数据
        sampler,
        # use_ddp=n_gpus > 1,
        device=device,
        batch_size=4096,
        shuffle=True,
        drop_last=False,
#         pin_memory =True,
        num_workers=4)

    # validation sampler
    val_loader = dgl.dataloading.DataLoader(
        g,
        target_idx[val_idx],
        sampler,
        # use_ddp=n_gpus > 1,
        device=device,
        batch_size=4096,
        shuffle=False,
        drop_last=False,
        num_workers=4)

    inv_target = inv_target.to(device)
    labels = labels.to(device)
    for epoch in tqdm.tqdm(range(1000)):
        model.train()
        embedding.train()

        for i, sample_data in enumerate(tqdm.tqdm(loader)):
            input_nodes, seeds, blocks = sample_data
            seeds = seeds.long()
            seeds = inv_target[seeds]
            t0 = time.time()

            feats = embedding(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'],
                                node_feats)
            logits,hidden_status = model(blocks,feats)
            loss = my_loss(hidden_status,C_mal,pse_label,seeds)
            optimizer.zero_grad()
            emb_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            emb_optimizer.step()
            if i % 100 == 0 :
                roc_auc,fpr,tpr,thresholds = score_auc(hidden_status, labels[seeds],C_mal)
                print('loss: {}'.format(loss.item()))

        val_logits, val_seeds = evaluate(model, embedding,val_loader, node_feats, inv_target)
        roc_auc,fdr,tpr,thresholds = score_auc(val_logits, labels[val_seeds],C_mal)
        print(roc_auc)
        early_stop = stopper.step(roc_auc,model=model,embed_layer=embedding)
        if early_stop:
            break
    
    # model = th.load('/data1/hongjiegu/checkpoint/HGSLA/CLV2_large.pt')
    # embedding = th.load('/data1/hongjiegu/checkpoint/HGSLA/CLV2_Embedding_large.pt')
    # node_ids = th.arange(g.number_of_nodes())
    # domain_mask = (g.ndata['ntype'] == 0)  # 假设domain节点的类型id为0
    # domain_node_ids = node_ids[domain_mask]
    # Softmax = nn.Softmax(dim=1)
    # logits_batch = []

    # fanouts = [20,20,20]
    # # create model
    # # all model params are in device.
    # model.eval()
    # embedding.eval()
    # sampler = dgl.dataloading.NeighborSampler(fanouts)
    # loader = dgl.dataloading.DataLoader(
    #     g,
    #     domain_node_ids,
    #     sampler,
    #     batch_size=8192,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers= 4
    #     )     

    # with th.no_grad():
    #     th.cuda.empty_cache()
    #         #neg_seeds.append(seeds)
    #     for sample_data in tqdm.tqdm(loader):
    #         inputs, seeds, blocks = sample_data
    #         seeds = seeds.long()
    #         seeds = inv_target[seeds]

    #         # for block in blocks:
    #         #     gen_norm(block)

    #         feats = embedding(blocks[0].srcdata[dgl.NID],
    #                             blocks[0].srcdata['ntype'],
    #                             blocks[0].srcdata['type_id'],
    #                             node_feats)
    #         print(feats)
    #         logits,hidden_status = model(blocks, feats)
    #         print(hidden_status)
    #         break
    #         logits_batch.append(hidden_status)
    # logits_batch = th.cat(logits_batch)
    
    # train_embs = logits_batch[target_idx[train_idx]]
    # test_embs = logits_batch[target_idx[test_idx]]
    # val_embs = logits_batch[target_idx[val_idx]]
    # inv_target = inv_target.to(device)
    # label = labels.to(device)
    # train_labels = label[inv_target[train_idx]]
    # test_labels = label[inv_target[test_idx]]
    # val_labels = label[inv_target[val_idx]]
    # print(train_idx)
    # class LogReg(nn.Module):
    #     def __init__(self, hid_dim, n_classes):
    #         super(LogReg, self).__init__()

    #         self.fc = nn.Linear(64, 2)

    #     def forward(self, x):
    #         ret = self.fc(x)
    #         return ret

    # accs = []
    # print(test_embs.shape)
    # # Step 5:  Linear evaluation ========================================================== #
    # model = LogReg(64, 2)
    # opt = th.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.01)
    # stopper = clf_earlyStopping(10)

    # model = model.to(device)
    # loss_fn = nn.CrossEntropyLoss()
    # print(train_embs)
    # for epoch in range(300):
    #     model.train()
    #     opt.zero_grad()
    #     logits = model(train_embs)
    #     loss = loss_fn(logits, train_labels)

    #     loss.backward()
    #     opt.step()
    #     acc = eval_clf(model,test_embs,val_embs)
    #     print(acc.item())
        # early_stop = stopper.step(acc.cpu(),clf=model)
        # if early_stop:
        #     break