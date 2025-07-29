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
from dgl.dataloading import NeighborSampler, DataLoader

import dgl
from dgl import DGLGraph
from dgl.dataloading import NeighborSampler, DataLoader as DGLDataLoader
from functools import partial
#from utils.utils import earlyStopping
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
import sys
sys.path.append('/home/hongjiegu/projects/model')
sys.path.append('/home/hongjiegu/projects/GRAVEL/baselines')
from SlotGAT.utils.tools import func_args_parse,single_feat_net,vis_data_collector
#dgl库，内置数据集
from HGAFA_V2 import RelGraphEmbedLayer
from HGT import HGT
import math
from dgl.nn.pytorch import TypedLinear
from tqdm import tqdm
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
warnings.filterwarnings("ignore")

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
from dgl.nn import LabelPropagation

#dgl库，内置数据集

import warnings
from sklearn.model_selection import train_test_split
from SlotGAT.GNN import slotGAT


from rphgnn.layers.rphgnn_pre import rphgnn_propagate_and_collect
from rphgnn.utils.nested_data_utils import gather_h_y, nested_gather, nested_map
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from contrastive_learning import score,evaluate,earlyStopping

# RMR相关导入
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn

# SimpleHGN相关导入
from SimpleHGN_standalone import SimpleHGN
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES']='5'
device = th.device("cuda" if th.cuda.is_available() else "cpu")

TAU=0.1
data_path='/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2022-1114-1605-a'
domain = pd.read_csv(data_path + '/nodes/domain_new')
file = pd.read_csv(data_path + '/nodes/file_nodes')
url = pd.read_csv(data_path + '/nodes/url_nodes')
ip = pd.read_csv(data_path + '/nodes/ip_new')
domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/domain_feats_rand.pt')
ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/ip_feats_rand.pt')
url_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/url_feats_rand.pt')
file_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/file_feats_rand.pt')
test_data = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/iochg-dgl/ood_malicious_2022_1114_1605.csv')

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

    elif benchmark == 'iochg_small':
        data_path='/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2023-1212-1537-zcf'
        test_data = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/ood_malicious_1537.csv')

        # from ogb.nodeproppred import DglNodePropPredDataset
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
    g.nodes['domain'].data['label'] = th.tensor(domain['label'])  # .to(device)
    g.nodes['domain'].data['train_mask'] = th.tensor(domain['with_label_train']).T  # .to(devicez
    g.nodes['domain'].data['val_mask'] = th.tensor(domain['with_label_valid']).T  # .to(device)
    if datatype == 'iid':
        g.nodes['domain'].data['test_mask'] = th.tensor(domain['with_label_test']).T
    elif datatype == 'ood':
        domain.to_be_predicted =  domain.content.isin(test_data.domain)
        g.nodes['domain'].data['test_mask'] = th.tensor(domain.to_be_predicted).T

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
    def __init__(self, patience=10, benchmark=None, baselines=None, save_dir=None, zero_shot=False):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.best_auc = None
        self.benchmark = benchmark
        self.baselines = baselines
        self.zero_shot = zero_shot
        self.save_dir = save_dir or "/home/hongjiegu/projects/GRAVEL/baselines/checkpoint"

    def step(self, auc, model, embed_layer=None):
        if self.best_auc is None:
            self.best_auc = auc
        elif auc < self.best_auc:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 构建保存路径
            model_name = f"{self.benchmark}-{self.baselines}"
            if self.zero_shot:
                model_name += "_zero-shot"
            model_path = os.path.join(self.save_dir, f"{model_name}.pt")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            if embed_layer is not None:
                embed_name = f"{self.benchmark}-{self.baselines}"
                if self.zero_shot:
                    embed_name += "_zero-shot"
                embed_path = os.path.join(self.save_dir, f"{embed_name}_Embedding.pt")
                torch.save(embed_layer.state_dict(), embed_path)
            self.best_auc = max(auc, self.best_auc)
            self.counter = 0
        return self.early_stop
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

def cal_malicious_logits_center(node_feats, g, labels):
    """
    计算模型预测正确的label=1样本的恶意中心
    """
    fanouts = [20,20,20]
    embed_layer = th.load('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint/pdns-simplehgn_Embedding.pt')
    # create model
    # all model params are in device.
    model = th.load('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint/pdns-simplehgn.pt')
    
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
    embed_layer = th.load('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint/pdns-simplehgn_Embedding.pt')
    # create model
    # all model params are in device.
    model = th.load('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint/pdns-simplehgn.pt')
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

def evaluate_simplehgn(model, embed_layer, val_loader, node_feats, inv_target,device):
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
                                node_feats).to(device)
            logits = model.forward(blocks, feats)
            
            val_logits.append(logits['domain'].cpu())
            val_seeds.append(seeds.cpu())
    
    val_logits = th.cat(val_logits, dim=0)
    val_seeds = th.cat(val_seeds, dim=0)
    
    return val_logits, val_seeds
def ood_loss(logits,labels):
    # logits: [batch_size, num_classes]
    probs = F.softmax(logits, dim=1)
    num_classes = logits.size(1)
    uniform = th.full_like(probs, 1.0 / num_classes)
    # KL散度，鼓励分布接近均匀
    return F.kl_div(probs.log(), uniform, reduction='batchmean')


    
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

def simplehgnn_trainer(g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels, hg, device, benchmark, baselines, zero_shot=False):
    """SimpleHGN训练器 - 使用DGL block批量处理避免OOM"""
    print("Starting SimpleHGN training with batch processing...")
    print(f"Using device: {device}")
    print(f"Benchmark: {benchmark}")
    print(f"Zero-shot: {zero_shot}")

    # 将图移到设备上
    hg = hg.to(device)

    # 特征处理 - 确保特征数量与节点数量匹配
    h_dict = {}
    for i, ntype in enumerate(hg.ntypes):
        feat = node_feats[i].to(device)
        num_nodes = hg.num_nodes(ntype)
        if feat.shape[0] < num_nodes:
            padded = torch.zeros(num_nodes, feat.shape[1], dtype=feat.dtype, device=feat.device)
            padded[:feat.shape[0]] = feat
            h_dict[ntype] = padded
        else:
            h_dict[ntype] = feat[:num_nodes]

    # 获取标签和掩码
    labels = hg.nodes['domain'].data['label'].to(device)
    train_mask = hg.nodes['domain'].data['train_mask'].to(device)
    val_mask = hg.nodes['domain'].data['val_mask'].to(device)
    test_mask = hg.nodes['domain'].data['test_mask'].to(device)
    in_dims = [node_feats[i].shape[1] for i in range(len(node_feats))]

    # 模型参数
    model = SimpleHGN(
        edge_dim=64,
        num_etypes=num_rels,
        in_dim=in_dims,
        hidden_dim=64,
        num_classes=num_classes,
        num_layers=3,
        heads=[4, 4, 1],
        feat_drop=0.1,
        negative_slope=0.2,
        residual=True,
        beta=0.1,
        ntypes=hg.ntypes
    ).to(device)
    embed_layer = RelGraphEmbedLayer(   device,
                                        num_of_ntype,
                                        node_feats,
                                        16,
                                        dgl_sparse=False)
    dense_params = list(model.parameters())
    optimizer = torch.optim.Adam(dense_params, lr=0.001, weight_decay=1e-5)
    embs = list(embed_layer.embeds.parameters())
    emb_optimizer = torch.optim.Adam(embs, lr=0.001)
    stopper = earlyStopping(10, benchmark, "simplehgn", zero_shot=zero_shot)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Training samples: {train_mask.sum().item()}")
    print(f"Validation samples: {val_mask.sum().item()}")
    print(f"Test samples: {test_mask.sum().item()}")

    # 设置采样参数
    fanouts = [20, 20, 20]  # 每层的邻居采样数
    batch_size = 1024  # 根据GPU内存调整
    
    # 创建采样器
    sampler = dgl.dataloading.NeighborSampler(fanouts, 'in')

    # 创建数据加载器
    train_loader = dgl.dataloading.DataLoader(
        g,
        target_idx[train_idx],
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True
    )
    
    val_loader = dgl.dataloading.DataLoader(
        g,
        target_idx[val_idx],
        sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=True
    )
    
    test_loader = dgl.dataloading.DataLoader(
        g,
        target_idx[test_idx],
        sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=True
    )
    h_dict = th.cat(node_feats)
    # 训练循环
    for epoch in range(100):
        train_loss = train_epoch_simplehgn_batch(model, optimizer, emb_optimizer, train_loader, embed_layer, node_feats, labels, device, inv_target)
        val_logits, val_seeds = evaluate_simplehgn(model, embed_layer, val_loader,node_feats, inv_target.cpu(),device)
        f1, precision,roc_auc,recall,fpr,tpr,thresholds = score(val_logits, labels[val_seeds].cpu(),0.5)
        print( " Validation f1: {:.4f}| Validation roc_auc: {:.4f} | Validation precision: {:.4f}|Validation recall: {:.4f}".format(f1,  roc_auc,precision,recall))
        early_stop = stopper.step(f1,model,embed_layer)
        if early_stop:
            break

    # 最终测试
    print("\n" + "="*50)
    print("Final Testing")
    print("="*50)
    test_logits, test_seeds = evaluate_simplehgn(model, embed_layer, test_loader,node_feats, inv_target.cpu(),device)
    f1, precision,roc_auc,recall,fpr,tpr,thresholds = score(test_logits, labels[test_seeds],0.5)
    print(f"Test f1: {f1:.4f}")
    print(f"Test roc_auc: {roc_auc:.4f}")
    print(f"Test precision: {precision:.4f}")
    print(f"Test recall: {recall:.4f}")

    return {
        'f1': f1,
        'precision': precision,
        'roc_auc': roc_auc,
        'recall': recall
    }


def train_epoch_simplehgn_batch(model, optimizer, emb_optimizer, train_loader, embed_layer, node_feats, labels, device, inv_target):
    """SimpleHGN批量训练一个epoch"""
    model.train()
    embed_layer.train()
    total_loss = 0
    total_batches = 0
    

    for input_nodes, seeds, blocks in train_loader:

        seeds = seeds.long().cpu()
        seeds = inv_target[seeds]
        optimizer.zero_grad()
        emb_optimizer.zero_grad()

        # 准备批量数据 - 使用第一个block的输入节点
        feats = embed_layer(blocks[0].srcdata[dgl.NID],
                            blocks[0].srcdata['ntype'],
                            blocks[0].srcdata['type_id'],
                            node_feats).to(device)
        
        # 前向传播 - 传递blocks列表
        logits = model(blocks, feats)['domain']
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
        # 计算损失
        # loss = classification_loss + ood_loss_value
        loss = my_loss(hidden_status,C_mal,pse_label,seeds)

        loss.backward()
        optimizer.step()

        emb_optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    
    return total_loss / total_batches

g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg = get_g2('pdns','iid')

simplehgnn_trainer(g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels, hg, device, 'pdns', 'simplehgn_backbone', zero_shot=False)