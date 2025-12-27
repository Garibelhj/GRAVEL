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
from sklearn.metrics import precision_recall_curve
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

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES']='4'
device = th.device("cuda" if th.cuda.is_available() else "cpu")

TAU=0.1
# data_path='/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2022-1114-1605-a'
# domain = pd.read_csv(data_path + '/nodes/domain_new')
# file = pd.read_csv(data_path + '/nodes/file_nodes')
# url = pd.read_csv(data_path + '/nodes/url_nodes')
# ip = pd.read_csv(data_path + '/nodes/ip_new')
# domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/domain_feats_rand.pt')
# ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/ip_feats_rand.pt')
# url_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/url_feats_rand.pt')
# file_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/file_feats_rand.pt')
# test_data = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/iochg-dgl/ood_malicious_2022_1114_1605.csv')

# # data_path='/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2023-1212-1537-zcf'
# # test_data = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/ood_malicious_1537.csv')

# # # from ogb.nodeproppred import DglNodePropPredDataset
# # domain = pd.read_csv(data_path + '/nodes/domain_new')
# # domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/domain_feats_rand.pt')
# # ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/ip_feats_rand.pt')
# # url_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/url_feats_rand.pt')
# # file_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/file_feats_rand.pt')
# # file = pd.read_csv(data_path + '/nodes/file_nodes')
# # url = pd.read_csv(data_path + '/nodes/url_nodes')
# domain.to_be_predicted =  domain.content.isin(test_data.content)

# #有标签数据的mask
# domain['with_label'] = domain['label']
# domain.loc[domain['with_label']<2,'with_label']  = True
# domain.loc[domain['with_label']>=2,'with_label']  = False
# with_label_mask = th.tensor(domain['with_label'].to_numpy(dtype=bool)).T
# domain_id = th.tensor(domain.index)
# train_set, test_set = train_test_split(domain_id[with_label_mask], test_size=0.2, random_state=47)
# test_set, valid_set = train_test_split(test_set, test_size=0.5, random_state=47)
# train_set = train_set.numpy()
# test_set = test_set.numpy()
# valid_set = valid_set.numpy()
# train_set = np.sort(train_set, axis=0)
# test_set = np.sort(test_set, axis=0)
# valid_set = np.sort(valid_set,axis=0)
# j, k, m = 0, 0, 0
# train_idx = []
# test_idx = []
# valid_idx = []

# for i in range(len(domain)):
#     if j < train_set.size and i == train_set[j]:
#         train_idx.append(True)
#         test_idx.append(False)
#         valid_idx.append(False)
#         j += 1
#     elif k < test_set.size and i == test_set[k]:
#         test_idx.append(True)
#         train_idx.append(False)
#         valid_idx.append(False)
#         k += 1
#     elif m < valid_set.size and i == valid_set[m]:
#         valid_idx.append(True)
#         train_idx.append(False)
#         test_idx.append(False)
#         m += 1
#     else:
#         train_idx.append(False)
#         test_idx.append(False)
#         valid_idx.append(False)

# domain['with_label_train'] = train_idx
# domain['with_label_test'] = test_idx
# domain['with_label_valid'] = valid_idx

# domain.to_csv(data_path+'/nodes/domain_new',index=False)


# file['label1'] = file['label']
# file.loc[file['label1']==1,'label']  = 0
# file.loc[file['label1']==0,'label']  = 1

# #待预测的数据打上mask，to_be_predicted字段只有label=2时才为True
# file['to_be_predicted'] = file['label']
# file.loc[file['to_be_predicted']<2,'to_be_predicted']  = False
# file.loc[file['to_be_predicted']>2,'to_be_predicted']  = False   # != 2 ???
# file.loc[file['to_be_predicted']==2,'to_be_predicted']  = True
# #有标签数据的mask
# file['with_label'] = file['label']
# file.loc[file['with_label']<2,'with_label']  = True
# file.loc[file['with_label']>=2,'with_label']  = False
# with_label_mask = th.tensor(file['with_label'].to_numpy(dtype=bool)).T
# file_id = th.tensor(file.index)
# train_set, test_set = train_test_split(file_id[with_label_mask], test_size=0.2, random_state=47)

# test_set, valid_set = train_test_split(test_set, test_size=0.5, random_state=47)
# train_set = train_set.numpy()
# test_set = test_set.numpy()
# valid_set = valid_set.numpy()
# train_set = np.sort(train_set, axis=0)
# test_set = np.sort(test_set, axis=0)
# valid_set = np.sort(valid_set,axis=0)
# j, k, m = 0, 0, 0
# train_idx = []
# test_idx = []
# valid_idx = []

# for i in range(len(file)):
#     if j < train_set.size and i == train_set[j]:
#         train_idx.append(True)
#         test_idx.append(False)
#         valid_idx.append(False)
#         j += 1
#     elif k < test_set.size and i == test_set[k]:
#         test_idx.append(True)
#         train_idx.append(False)
#         valid_idx.append(False)
#         k += 1
#     elif m < valid_set.size and i == valid_set[m]:
#         valid_idx.append(True)
#         train_idx.append(False)
#         test_idx.append(False)
#         m += 1
#     else:
#         train_idx.append(False)
#         test_idx.append(False)
#         valid_idx.append(False)
# file['with_label_train'] = train_idx
# file['with_label_test'] = test_idx
# file['with_label_valid'] = valid_idx

# file.to_csv(data_path+'/nodes/file_nodes',index=False)

# url['label1'] = url['label']
# url.loc[url['label1']==1,'label']  = 0
# url.loc[url['label1']==0,'label']  = 1

# #待预测的数据打上mask，to_be_predicted字段只有label=2时才为True
# url['to_be_predicted'] = url['label']
# url.loc[url['to_be_predicted']<2,'to_be_predicted']  = False
# url.loc[url['to_be_predicted']>2,'to_be_predicted']  = False   # != 2 ???
# url.loc[url['to_be_predicted']==2,'to_be_predicted']  = True
# #有标签数据的mask
# url['with_label'] = url['label']
# url.loc[url['with_label']<2,'with_label']  = True
# url.loc[url['with_label']>=2,'with_label']  = False
# with_label_mask = th.tensor(url['with_label'].to_numpy(dtype=bool)).T
# url_id = th.tensor(url.index)
# train_set, test_set = train_test_split(url_id[with_label_mask], test_size=0.2, random_state=47)
# test_set, valid_set = train_test_split(test_set, test_size=0.5, random_state=47)
# train_set = train_set.numpy()
# test_set = test_set.numpy()
# valid_set = valid_set.numpy()
# train_set = np.sort(train_set, axis=0)
# test_set = np.sort(test_set, axis=0)
# valid_set = np.sort(valid_set, axis = 0)
# j, k, m = 0, 0, 0
# train_idx = []
# test_idx = []
# valid_idx = []

# for i in range(len(url)):
#     if j < train_set.size and i == train_set[j]:
#         train_idx.append(True)
#         test_idx.append(False)
#         valid_idx.append(False)
#         j += 1
#     elif k < test_set.size and i == test_set[k]:
#         test_idx.append(True)
#         train_idx.append(False)
#         valid_idx.append(False)
#         k += 1
#     elif m < valid_set.size and i == valid_set[m]:
#         valid_idx.append(True)
#         train_idx.append(False)
#         test_idx.append(False)
#         m += 1
#     else:
#         train_idx.append(False)
#         test_idx.append(False)
#         valid_idx.append(False)
# url['with_label_train'] = train_idx
# url['with_label_test'] = test_idx
# url['with_label_valid'] = valid_idx
# url.to_csv(data_path+'/nodes/url_nodes',index=False)

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

#     划分训练集和测试集
   

    g.nodes['domain'].data['label'] = th.tensor(domain['label'])  # .to(device)
    g.nodes['domain'].data['train_mask'] = th.tensor(domain['with_label_train']).T  # .to(device)
    g.nodes['domain'].data['test_mask'] = th.tensor(domain['to_be_predicted']).T  # .to(device)
    g.nodes['domain'].data['val_mask'] = th.tensor(domain['with_label_valid']).T  # .to(device)

    
    g.nodes['file'].data['label'] = th.tensor(file['label'])  # .to(device)    # wy, 4/3
    g.nodes['file'].data['train_mask'] = th.tensor(file['with_label_train']).T  # .to(device)     # wy, 4/3
    g.nodes['file'].data['test_mask'] = th.tensor(file['to_be_predicted']).T  # .to(device)     # wy, 4/3
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
            node_feats.append(512)
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
    node_tids = hg.ndata[dgl.NTYPE]
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
    #2分类


    return hg, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,g


def get_g2(benchmark,datatype):
    edge = {}
    u = {}
    v = {}
    if benchmark == 'pdns':
        data_path = '/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl'
        domain = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/domain.csv")
        ip = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/ips.csv")
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/feats/domain_feats_rand.pt')
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/feats/ip_feats_rand.pt')
        test_data = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/ood_pdns.csv')

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
        test_data = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/ood_minta.csv')

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
        url = pd.read_csv(data_path + '/nodes/url_nodes')
        file = pd.read_csv(data_path + '/nodes/file_nodes')
        ip = pd.read_csv(data_path + '/nodes/ip_new')
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/domain_feats_rand.pt')
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/ip_feats_rand.pt')
        url_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/url_feats_rand.pt')
        file_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/file_feats_rand.pt')
        test_data = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/iochg-dgl/ood_malicious_2022_1114_1605.csv',header=0)

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

        g.nodes['domain'].data['label'] = th.tensor(domain['label'])  # .to(device)


        
        g.nodes['file'].data['label'] = th.tensor(file['label'])  # .to(device)    # wy, 4/3
        g.nodes['file'].data['train_mask'] = th.tensor(file['train_mask']).T  # .to(device)     # wy, 4/3
        g.nodes['file'].data['test_mask'] = th.tensor(file['test_mask']).T  # .to(device)     # wy, 4/3
        g.nodes['file'].data['val_mask'] = th.tensor(file['val_mask']).T
        
        g.nodes['url'].data['label'] = th.tensor(url['label'])  # .to(device)    # wy, 4/3
        g.nodes['url'].data['train_mask'] = th.tensor(url['train_mask']).T  # .to(device)     # wy, 4/3
        g.nodes['url'].data['test_mask'] = th.tensor(url['test_mask']).T  # .to(device)     # wy, 4/3
        g.nodes['url'].data['val_mask'] = th.tensor(url['val_mask']).T
        g.nodes['ip'].data['label'] = th.zeros(g.number_of_nodes('ip')).long()
        g.nodes['ip'].data['train_mask'] = th.full((g.number_of_nodes('ip'),), False)
        g.nodes['ip'].data['test_mask'] = th.full((g.number_of_nodes('ip'),), False)
        g.nodes['ip'].data['val_mask'] = th.full((g.number_of_nodes('ip'),), False)
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
            th.save(model,'/home/hongjiegu/projects/GRAVEL/checkpoint/iochg_GRAVEL_wo_cl_ood.pt')
            th.save(embed_layer,'/home/hongjiegu/projects/GRAVEL/checkpoint/iochg_GRAVEL_wo_cl_embed_ood.pt')
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
    # print("v:")
    # print(v)
    # print("eid:")
    # print(eid)
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
#     g[g.edata['etype']==1].edata['norm_w'] = 0.3
#     g.edata['norm_w'] = 



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
    FDR,TPR,threshold = precision_recall_curve(true_labels,prob)
    FDR = 1-FDR
    return f1, macro,roc_auc,recall,FDR,TPR,threshold


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

            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(seeds.cpu().detach())

    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)
    
    return eval_logits, eval_seeds

def ood_loss(logits,labels):
    # logits: [batch_size, num_classes]
    probs = F.softmax(logits, dim=1)
    num_classes = logits.size(1)
    uniform = th.full_like(probs, 1.0 / num_classes)
    # KL散度，鼓励分布接近均匀
    return F.kl_div(probs.log(), uniform, reduction='batchmean')

def plot_test_probability_distribution(model, embed_layer, test_loader, 
                                     node_feats, inv_target, labels, save_path='test_probability_distribution.png'):
    """
    绘制模型在test_loader上的恶意概率分布直方图和ROC曲线（双子图）
    """
    model.eval()
    embed_layer.eval()
    
    all_logits = []
    all_seeds = []
    with th.no_grad():
        for sample_data in test_loader:
            inputs, seeds, blocks = sample_data
            seeds = seeds.long().cpu()
            seeds = inv_target[seeds]
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'],
                                node_feats)
            logits, hidden_status = model(blocks, feats)
            all_logits.append(logits.cpu().detach())
            all_seeds.append(seeds.cpu().detach())
    
    all_logits = th.cat(all_logits)
    all_seeds = th.cat(all_seeds)
    probs = F.softmax(all_logits, dim=1)
    malicious_probs = probs[:, 1].numpy()
    true_labels = labels[all_seeds].cpu().numpy()
    
    # 创建双子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：概率分布直方图
    ax1.hist(malicious_probs, bins=50, alpha=0.7, color='blue', edgecolor='black', label='All Test Samples')
    
    # 按标签分类绘制
    label_0_mask = (true_labels == 0)
    label_1_mask = (true_labels == 1)
    label_2_mask = (true_labels >= 2)
    
    if np.any(label_0_mask):
        ax1.hist(malicious_probs[label_0_mask], bins=30, alpha=0.6, color='green', 
                edgecolor='black', label='Benign (label=0)', histtype='step', linewidth=2)
    
    if np.any(label_1_mask):
        ax1.hist(malicious_probs[label_1_mask], bins=30, alpha=0.6, color='red', 
                edgecolor='black', label='Malicious (label=1)', histtype='step', linewidth=2)
    
    if np.any(label_2_mask):
        ax1.hist(malicious_probs[label_2_mask], bins=30, alpha=0.6, color='orange', 
                edgecolor='black', label='Unlabeled (label>=2)', histtype='step', linewidth=2)
    
    ax1.set_title('Test Set Malicious Probability Distribution', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Malicious Probability', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # 添加统计信息
    mean_prob = np.mean(malicious_probs)
    std_prob = np.std(malicious_probs)
    ax1.text(0.02, 0.98, f'Mean: {mean_prob:.3f}\nStd: {std_prob:.3f}\nSamples: {len(malicious_probs)}', 
            transform=ax1.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=12)
    
    ax1.axvline(mean_prob, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_prob:.3f}')
    
    # 右图：ROC曲线
    # 只使用有标签的样本（label=0和label=1）计算ROC
    labeled_mask = (true_labels == 0) | (true_labels == 1)
    if np.any(labeled_mask):
        labeled_probs = malicious_probs[labeled_mask]
        labeled_labels = true_labels[labeled_mask]
        
        # 计算ROC曲线
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(labeled_labels, labeled_probs)
        roc_auc = auc(fpr, tpr)
        
        # 绘制ROC曲线
        ax2.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate', fontsize=14)
        ax2.set_ylabel('True Positive Rate', fontsize=14)
        ax2.set_title('ROC Curve', fontsize=16, fontweight='bold')
        ax2.legend(loc="lower right", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 添加AUC信息
        ax2.text(0.02, 0.98, f'AUC: {roc_auc:.3f}\nLabeled samples: {len(labeled_probs)}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'No labeled samples\nfor ROC calculation', 
                transform=ax2.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8), fontsize=14)
        ax2.set_title('ROC Curve (No labeled data)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"测试集概率分布和ROC曲线图已保存到: {save_path}")
    print("\n=== 测试集概率分布统计摘要 ===")
    print(f"总样本数量: {len(malicious_probs)}")
    print(f"平均概率: {mean_prob:.4f}")
    print(f"标准差: {std_prob:.4f}")
    print(f"最小值: {np.min(malicious_probs):.4f}")
    print(f"最大值: {np.max(malicious_probs):.4f}")
    print(f"中位数: {np.median(malicious_probs):.4f}")
    
    # 按标签分类统计
    if np.any(label_0_mask):
        benign_probs = malicious_probs[label_0_mask]
        print(f"\nBenign样本 (label=0):")
        print(f"  样本数量: {len(benign_probs)}")
        print(f"  平均概率: {np.mean(benign_probs):.4f}")
        print(f"  标准差: {np.std(benign_probs):.4f}")
    
    if np.any(label_1_mask):
        malicious_probs_label1 = malicious_probs[label_1_mask]
        print(f"\nMalicious样本 (label=1):")
        print(f"  样本数量: {len(malicious_probs_label1)}")
        print(f"  平均概率: {np.mean(malicious_probs_label1):.4f}")
        print(f"  标准差: {np.std(malicious_probs_label1):.4f}")
    
    if np.any(label_2_mask):
        unlabeled_probs = malicious_probs[label_2_mask]
        print(f"\nUnlabeled样本 (label>=2):")
        print(f"  样本数量: {len(unlabeled_probs)}")
        print(f"  平均概率: {np.mean(unlabeled_probs):.4f}")
        print(f"  标准差: {np.std(unlabeled_probs):.4f}")
    
    # ROC统计信息
    if np.any(labeled_mask):
        print(f"\n=== ROC统计信息 ===")
        print(f"有标签样本数量: {len(labeled_probs)}")
        print(f"ROC AUC: {roc_auc:.4f}")
    
    return {
        'logits': all_logits,
        'seeds': all_seeds,
        'malicious_probs': malicious_probs,
        'true_labels': true_labels,
        'roc_auc': roc_auc if np.any(labeled_mask) else None
    }

def run(proc_id, n_gpus, n_cpus, args, devices, dataset, queue=None):
    dev_id = 0
    g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg = dataset
    gen_norm(g)     # test, wy, 2023/3/27
    stopper = earlyStopping(5)
    fanouts = [int(fanout) for fanout in args.fanout.split(',')]
    sampler = dgl.dataloading.NeighborSampler(fanouts,'in')
    loader = dgl.dataloading.DataLoader(
        g,
        target_idx[train_idx],  # 要训练的数据
        sampler,
        # use_ddp=n_gpus > 1,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
#         pin_memory =True,
        num_workers=args.num_workers)

    # validation sampler
    val_loader = dgl.dataloading.DataLoader(
        g,
        target_idx[val_idx],
        sampler,
        # use_ddp=n_gpus > 1,
        device=device,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers)

    # test sampler
    test_sampler = dgl.dataloading.NeighborSampler(fanouts,'in')
    test_loader = dgl.dataloading.DataLoader(
        g,
        target_idx[test_idx],
        test_sampler,
        device=device,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers)

    embed_layer = RelGraphEmbedLayer(device,
                                    num_of_ntype,
                                    node_feats,
                                    args.n_hidden,
                                    dgl_sparse=args.dgl_sparse).to(device)


    model = EntityClassify(dev_id,
                        g.number_of_nodes(),
                        args.n_hidden,
                        num_classes,
                        num_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers - 1,
                        dropout=args.dropout,
                        use_self_loop=args.use_self_loop,
#                            low_mem=args.low_mem,
                        layer_norm=args.layer_norm).to(device)
    
    pseudo_indices = set()  # 记录已添加的伪标签索引
    dense_params = list(model.parameters())
    optimizer = th.optim.Adam(dense_params, lr=args.lr, weight_decay=args.l2norm)
    embs = list(embed_layer.embeds.parameters())
    emb_optimizer = th.optim.Adam(embs, lr=args.sparse_lr) if len(embs) > 0 else None
    inv_target = inv_target.to(device)
    labels = labels.to(device)

    for epoch in range(args.n_epochs):

        model.train()
        embed_layer.train()

        for i, sample_data in enumerate(loader):
            input_nodes, seeds, blocks = sample_data
            seeds = seeds.long()
            seeds = inv_target[seeds]
            t0 = time.time()

            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'],
                                node_feats)
            logits,hidden_status = model(blocks,feats)
            norms = th.norm(logits, p=2, dim=1, keepdim=True) + 1e-7
            norms = TAU*norms
            logits = th.div(logits, norms)
            batch_labels = labels[seeds].to(device)
            valid_label_mask = (batch_labels == 0) | (batch_labels == 1)
            ood_mask = (batch_labels >= 2)


            # 计算分类损失（对有标签数据）
            if valid_label_mask.sum() > 0:
                classification_loss = F.cross_entropy(logits[valid_label_mask], batch_labels[valid_label_mask])
            else:
                classification_loss = th.tensor(0.0, device=device)
            
            # 计算OOD损失（对未标记数据）
            if ood_mask.sum() > 0:
                ood_loss_value = ood_loss(logits[ood_mask], batch_labels[ood_mask])
            else:
                ood_loss_value = th.tensor(0.0, device=device)
            
            # 总损失
            loss = classification_loss + ood_loss_value

            # loss = classification_loss
            t1 = time.time()
            optimizer.zero_grad()
            if emb_optimizer is not None:
                emb_optimizer.zero_grad()

            loss.backward()
            if emb_optimizer is not None:
                emb_optimizer.step()

            optimizer.step()

            if i % 100 == 0 and proc_id == 0:
                # 只对有标签数据计算指标
                batch_labels = labels[seeds].to(device)
                valid_label_mask = (batch_labels == 0) | (batch_labels == 1)
                
                if valid_label_mask.sum() > 0:
                    labeled_logits = logits[valid_label_mask]
                    labeled_labels = batch_labels[valid_label_mask]
                    f1, precision, roc_auc, recall,fpr,tpr,thresholds = score(labeled_logits, labeled_labels,0.5)
                else:
                    f1, precision, roc_auc, recall = 0.0, 0.0, 0.0, 0.0

                print(
                    "Train Loss: {:.4f} (Cls: {:.4f}, OOD: {:.4f}) | f1: {:.4f}| roc_auc: {:.4f} | precision: {:.4f}| recall: {:.4f}".
                    format(loss.item(), classification_loss.item(), ood_loss_value.item(), f1, roc_auc,precision,recall))


        def collect_eval():
            eval_logits = []
            eval_seeds = []
            for i in range(n_gpus):
                log = queue.get()
                eval_l, eval_s = log
                eval_logits.append(eval_l)
                eval_seeds.append(eval_s)
            eval_logits = th.cat(eval_logits)
            eval_seeds = th.cat(eval_seeds)
            
            # 过滤掉label>=2的样本，只对有标签数据计算损失
            eval_labels = labels[eval_seeds].cpu()
            valid_eval_mask = (eval_labels == 0) | (eval_labels == 1)
            
            if valid_eval_mask.sum() > 0:
                eval_loss = F.cross_entropy(eval_logits[valid_eval_mask], eval_labels[valid_eval_mask]).item()
                eval_acc = th.sum(eval_logits[valid_eval_mask].argmax(dim=1) == eval_labels[valid_eval_mask]).item() / valid_eval_mask.sum().item()
            else:
                eval_loss = 0.0
                eval_acc = 0.0

            return eval_loss, eval_acc

        if (queue is not None) or (proc_id == 0):
            val_logits, val_seeds = evaluate(model, embed_layer,val_loader,
                                            node_feats, inv_target)

            if queue is not None:
                queue.put((val_logits, val_seeds))

            # gather evaluation result from multiple processes
            if proc_id == 0:
                # 过滤掉label>=2的样本，只对有标签数据计算损失
                val_labels = labels[val_seeds].cpu()
                valid_val_mask = (val_labels == 0) | (val_labels == 1)
                
                if valid_val_mask.sum() > 0:
                    val_loss = F.cross_entropy(val_logits[valid_val_mask], val_labels[valid_val_mask]).item()
                    val_acc = th.sum(val_logits[valid_val_mask].argmax(dim=1) == val_labels[valid_val_mask]).item() / valid_val_mask.sum().item()
                else:
                    val_loss = 0.0
                    val_acc = 0.0
                
                val_loss, val_acc = collect_eval() if queue is not None else (val_loss, val_acc)
                val_seeds = val_seeds.to(device)
                val_logits = val_logits.to(device)
                
                # 只对有标签数据计算指标
                val_labels = labels[val_seeds]
                valid_val_mask = (val_labels == 0) | (val_labels == 1)
                
                if valid_val_mask.sum() > 0:
                    f1, precision,roc_auc,recall,fpr,tpr,thresholds = score(val_logits[valid_val_mask], val_labels[valid_val_mask],0.5)
                else:
                    f1, precision, roc_auc, recall = 0.0, 0.0, 0.0, 0.0

                early_stop = stopper.step(f1,model,embed_layer)
                print("ACC: {:.4f}|Validation loss: {:.4f}| Validation f1: {:.4f}| Validation roc_auc: {:.4f} | Validation precision: {:.4f}|Validation recall: {:.4f}".format(val_acc,val_loss, f1,  roc_auc, precision,recall))

                if early_stop:
                    break


        # === 新增：绘制恶意概率分布图 ===
        # 重新构建train/val/test loader用于分布图
        plot_test_probability_distribution(model, embed_layer, test_loader, 
                                            node_feats, inv_target, labels, save_path='/home/hongjiegu/projects/GRAVEL/plot/test_probability_distribution.png')

def main(args, devices):
    # # load graph data
    g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg = get_g2(
        'iochg','ood')
    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    # 在使用多 GPU 启动训练过程之前创建 csr/coo/csc 格式，创建系数矩阵。
    # 这样可以避免在每个子进程中创建某些格式，从而节省内存和 CPU。
    g.create_formats_()
    n_cpus = 4
    n_gpus = 0
    run(0, n_gpus, n_cpus, args, devices,
        (g, node_feats, num_of_ntype, num_classes, num_rels, target_idx,
        inv_target, train_idx, val_idx, test_idx, labels,hg), None)

def config():
    parser = argparse.ArgumentParser(description='HGAFA')
    #     parser.add_argument('-s', '--seed', type=int, default=2,
    #                         help='Random seed')
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=64,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=str, default='0',
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--sparse-lr", type=float, default=2e-3,
                        help="sparse embedding learning rate")
    parser.add_argument("--n-bases", type=int, default=2,
                        help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=100,
                        help="number of training epochs")
    # parser.add_argument("-d", "--dataset", type=str, required=True,
    #                     help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=1e-5,
                        help="l2 norm coef")
    parser.add_argument("--fanout", type=str, default="20,20,20",
                        help="Fan-out of neighbor sampling.")
    parser.add_argument("--use-self-loop", default=True, action='store_true',
                        help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Mini-batch size. ")
    parser.add_argument("--eval-batch-size", type=int, default=4096,
                        help="Mini-batch size. ")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for dataloader.")
    parser.add_argument("--low-mem", default=False, action='store_true',
                        help="Whether use low mem RelGraphCov")
    parser.add_argument("--dgl-sparse", default=False, action='store_true',
                        help='Use sparse embedding for node embeddings.')
    parser.add_argument("--embedding-gpu", default=True, action='store_true',
                        help='Store the node embeddings on the GPU.')
    parser.add_argument('--node-feats', default=True, action='store_true',
                        help='Whether use node features')
    parser.add_argument('--layer-norm', default=False, action='store_true',
                        help='Use layer norm')
    parser.set_defaults(validation=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = config()
    devices = list(map(int, args.gpu.split(',')))
    print(args)
    main(args, devices)