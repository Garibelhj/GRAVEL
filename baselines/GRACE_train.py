
import argparse, gc
import numpy as np
import pandas as pd
import time
from functools import partial
import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import argparse
import warnings

import numpy as np
import torch as th
import torch.nn as nn
from aug import aug
import tqdm
from model import Grace

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
import dgl

import sys
sys.path.append('/home/hongjiegu/projects/IOC_TIME/model')
from sklearn.svm import LinearSVC

#dgl库，内置数据集
from model import MVGRL, LogReg

import warnings
from sklearn.model_selection import train_test_split

import torch as th
import dgl
import numpy as np
import torch as th
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power, inv

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import networkx as nx

from sklearn.preprocessing import MinMaxScaler

from dgl.nn import APPNPConv
warnings.filterwarnings("ignore")
dev_id=3
device = th.device('cuda:3'if th.cuda.is_available() else 'cpu')




def get_g(benchmark):
    edge = {}
    u = {}
    v = {}
    if benchmark == 'pdns':
        data_path = '/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl'
        ood_malicious_domain = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/ood_pdns.csv")
        domain = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/domains2.csv")
        ip = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/ips.csv")
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/feats/domain_feats_rand.pt')
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/feats/ip_feats_rand.pt')
        num_of_ntype = 2
        num_rels =6  
        # 设置to_be_predicted：OOD malicious样本为True
        domain['to_be_predicted'] = domain.domain.isin(ood_malicious_domain.domain)        
        # 从domain.label=0的数据中选择和to_be_predicted相同数量的数据，将to_be_predicted设置为True
        ood_count = domain.to_be_predicted.sum()
        print(ood_count)
        available_label_0_indices = domain[(domain['label'] == 0) & (domain['to_be_predicted'] == False)].index
        if len(available_label_0_indices) >= ood_count:
            selected_indices = np.random.choice(available_label_0_indices, size=int(ood_count), replace=False)
            domain.loc[selected_indices, 'to_be_predicted'] = True
        else:
            domain.loc[available_label_0_indices, 'to_be_predicted'] = True

        # 将ood_malicious_domain对应的domain的label设置
        print(domain.to_be_predicted.sum())
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
        ood_malicious_domain = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/ood_minta.csv")
        domain = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/domain.csv')
        print(domain.label.unique())
        ip = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/ip.csv')
        host = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/host.csv')
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/feats/domain_feats_rand.pt')
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/feats/ip_feats_rand.pt')
        host_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/feats/host_feats_rand.pt')
        num_of_ntype = 3
        num_rels =8    
        # 设置to_be_predicted：OOD malicious样本为True
        domain['to_be_predicted'] = domain.value.isin(ood_malicious_domain.domain)        
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

        # 将ood_malicious_domain对应的domain的label设置为1

        print(domain.to_be_predicted.sum())
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

    domain['with_label_valid'] = valid_idx
    
    # 将ood_malicious_domain对应的domain的label设置为1
    if benchmark == 'pdns':
        domain.loc[domain['domain'].isin(ood_malicious_domain['domain']), 'label'] = 1
    elif benchmark == 'minta':
        domain.loc[domain['value'].isin(ood_malicious_domain['domain']), 'label'] = 1
    elif benchmark == 'iochg':
        domain.loc[domain['content'].isin(ood_malicious_domain['content']), 'label'] = 1
    elif benchmark == 'iochg_small':
        domain.loc[domain['content'].isin(ood_malicious_domain['content']), 'label'] = 1
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

    g.nodes['domain'].data['label'] = th.tensor(domain['label'])  # .to(device)
    g.nodes['domain'].data['train_mask'] = th.tensor(domain['with_label_train']).T  # .to(device)
    g.nodes['domain'].data['test_mask'] = th.tensor(domain['to_be_predicted']).T  # .to(device)
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
                        dtype=th.long)
    inv_target[target_idx] = th.arange(0, target_idx.shape[0],
                                    dtype=th.long)
    target_idx = target_idx  # .to(device)
    labels = g.nodes['domain'].data['label']
    train_mask = g.nodes['domain'].data['train_mask']
    test_mask = g.nodes['domain'].data['test_mask']
    print(test_mask)
    val_mask = g.nodes['domain'].data['val_mask']
    train_idx = g.nodes('domain')[train_mask]
    test_idx = g.nodes('domain')[test_mask]
    val_idx = g.nodes('domain')[val_mask]
    num_classes = 2

    return hg, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,g


g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg = get_g('pdns')
node_ids = th.arange(g.number_of_nodes())
node_tids = g.ndata[dgl.NTYPE]
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
loader = dgl.dataloading.DataLoader(
    g,
    node_ids,  # 要训练的数据
    sampler,
    # use_ddp=n_gpus > 1,
    device=dev_id,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    #pin_memory =True,
    num_workers=0)

val_loader = dgl.dataloading.DataLoader(
    g,
    target_idx[val_idx],
    sampler,
    # use_ddp=n_gpus > 1,
    #device=dev_id,
    batch_size=1024,
    shuffle=False,
    drop_last=False,
    num_workers=0)
# test sampler
fanouts = [20,20,20]
test_sampler = dgl.dataloading.NeighborSampler(fanouts,'in')
test_loader = dgl.dataloading.DataLoader(
    g,
    target_idx[test_idx],
    test_sampler,
    # use_ddp=n_gpus > 1,
    device=dev_id,
    batch_size=16,
    shuffle=False,
    drop_last=False,
    #pin_memory =True,
    num_workers=0)


def count_parameters(model):
    return sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )


lr = 5e-4
hid_dim = 64
out_dim = 64

num_layers = 3
act_fn = ({"relu": nn.ReLU(), "prelu": nn.PReLU()})["relu"]

drop_edge_rate_1 = 0.2
drop_edge_rate_2 = 0.4
drop_feature_rate_1 = 0.3
drop_feature_rate_2 = 0.4
temp = 0.4
epochs = 20
wd = 1e-5



node_feats = th.cat(node_feats, dim=0).to(device)  # 假设特征的维度对齐
in_dim =16

# Step 3: Create model =================================================================== #
model = Grace(in_dim, hid_dim, out_dim, num_layers, act_fn, temp)
model = model.to(device)
print(f"# params: {count_parameters(model)}")
optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
best = 100
cnt_wait = 0
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    for i, sample_data in tqdm.tqdm(enumerate(loader)):
        input_nodes, seeds, blocks = sample_data
                        # 将种子节点映射回它们特定类型的id，以便它们可以用来查找各自的标签
        seeds = th.cat((input_nodes,seeds),0)
        feats = node_feats[seeds]

        _g = blocks[-1]
        # print(logits)
        graph1, feat1 = aug(_g, feats, drop_feature_rate_1, drop_edge_rate_1)
        graph2, feat2 = aug(_g, feats, drop_feature_rate_2, drop_edge_rate_2)

        graph1 = graph1.to(device)
        graph2 = graph2.to(device)

        feat1 = feat1.to(device)
        feat2 = feat2.to(device)

        loss = model(graph1, graph2, feat1, feat2)
        loss.backward()
        optimizer.step()
        if loss < best:
            best = loss
            cnt_wait = 0
            th.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == 10:
            print('Early stopping')
            break
    print(f"Epoch={epoch:03d}, loss={loss.item():.4f}")
    
g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg = get_g('pdns')

# # Step 5: Linear evaluation ============================================================== #
# print("=== Final ===")
# print(g)
graph = g.add_self_loop()
graph = graph.to(device)
feat  = th.cat(node_feats, dim=0).to(device)  # 假设特征的维度对齐

print(feat.shape)
embeds = model.get_embedding(graph, feat)

train_embs = embeds[train_idx]
test_embs = embeds[test_idx]
val_embs = embeds[val_idx]
inv_target = inv_target.to(device)
label = labels.to(device)
train_labels = label[inv_target[train_idx]]
test_labels = label[inv_target[test_idx]]
val_labels = label[inv_target[val_idx]]
accs = []
class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()

        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        ret = self.fc(x)
        return ret

accs = []

# Step 5:  Linear evaluation ========================================================== #
for _ in range(5):
    model = LogReg(64, 2)
    opt = th.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.01)

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(300):
        model.train()
        opt.zero_grad()
        logits = model(train_embs)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

model.eval()
logits = model(val_embs)
preds = th.argmax(logits, dim=1)
f1 = f1_score(val_labels.cpu(), preds.cpu())
auc = roc_auc_score(val_labels.cpu(),preds.cpu())
print(f1, auc)
logits = model(test_embs)
preds = th.argmax(logits, dim=1)
f1 = f1_score(test_labels.cpu(), preds.cpu())
auc = roc_auc_score(test_labels.cpu(),preds.cpu())
print(f1, auc)
