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
os.environ['CUDA_VISIBLE_DEVICES']='1'
device = th.device("cuda" if th.cuda.is_available() else "cpu")
TAU =0.6

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
            th.save(model,'/data1/hongjiegu/checkpoint/HGSLA/GRAVEL_small_end2end_v2_pdns.pt')
            th.save(embed_layer,'/data1/hongjiegu/checkpoint/HGSLA/GRAVEL_Embedding_small_end2end_v2_pdns.pt')
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



def ood_loss(logits,labels):
    # logits: [batch_size, num_classes]
    probs = F.softmax(logits, dim=1)
    num_classes = logits.size(1)
    uniform = th.full_like(probs, 1.0 / num_classes)
    # KL散度，鼓励分布接近均匀
    return F.kl_div(probs.log(), uniform, reduction='batchmean')

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

def generate_pseudo_labels(model, embed_layer, g, node_feats, inv_target, target_idx, test_idx, device, confidence_threshold=0):
    """Generate pseudo labels for unlabeled domains using current model"""
    model.eval()
    embed_layer.eval()
    
    # Get all domain nodes that need pseudo labels (test set)
    test_domain_ids = target_idx[test_idx]
    
    fanouts = [20, 20, 20]
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    loader = dgl.dataloading.DataLoader(
        g,
        test_domain_ids,
        sampler,
        batch_size=2048,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    # Create pseudo label tensor (initialize with zeros)
    num_domains = g.number_of_nodes()
    pseudo_label_tensor = th.zeros(num_domains, 2, device=device)  # [num_domains, 2] for binary classification
    
    with th.no_grad():
        for sample_data in tqdm.tqdm(loader, desc="Generating pseudo labels"):
            inputs, seeds, blocks = sample_data
            seeds = seeds.long()
            seeds = inv_target[seeds]
            
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                              blocks[0].srcdata['ntype'],
                              blocks[0].srcdata['type_id'],
                              node_feats)
            
            logits, hidden_status = model(blocks, feats)
            probs = F.softmax(logits, dim=1)
            
            # Get confidence scores
            confidence, predictions = th.max(probs, dim=1)
            
            # Only keep high-confidence predictions
            high_conf_mask = confidence > confidence_threshold
            if high_conf_mask.sum() > 0:
                # Use the probability distributions directly as pseudo labels
                pseudo_label_tensor[seeds[high_conf_mask]] = probs[high_conf_mask]
    
    return pseudo_label_tensor

def cal_malicious_logits_center_with_model(model, embed_layer, node_feats, g, labels, inv_target):
    """Calculate malicious and benign centers using specified model"""
    model.eval()
    embed_layer.eval()
    
    # 获取恶意和良性样本的掩码
    malicious_mask = (labels == 1)
    benign_mask = (labels == 0)
    
    node_ids = th.arange(g.number_of_nodes(), device=labels.device)
    domain_mask = (g.ndata['ntype'] == 0)
    domain_node_ids = node_ids[domain_mask]
    
    malicious_domain = domain_node_ids[malicious_mask]
    benign_domain = domain_node_ids[benign_mask]
    
    # 计算恶意中心
    malicious_logits_batch = []
    if len(malicious_domain) > 0:
        fanouts = [20,20,20]
        sampler = dgl.dataloading.NeighborSampler(fanouts)
        malicious_loader = dgl.dataloading.DataLoader(
            g, malicious_domain.cpu(), sampler, batch_size=2048, shuffle=False, drop_last=False, num_workers=0
        )     

        with th.no_grad():
            for sample_data in tqdm.tqdm(malicious_loader, desc="Computing malicious center"):
                inputs, seeds, blocks = sample_data
                seeds = seeds.long()
                seeds = inv_target[seeds]
                feats = embed_layer(blocks[0].srcdata[dgl.NID], blocks[0].srcdata['ntype'], blocks[0].srcdata['type_id'], node_feats)
                logits,hidden_status = model(blocks, feats)
                malicious_logits_batch.append(hidden_status)
        
    malicious_logits_batch = th.cat(malicious_logits_batch)
    C_mal = th.mean(malicious_logits_batch, dim=0)


    
    # 计算良性中心
    benign_logits_batch = []
    if len(benign_domain) > 0:
        fanouts = [20,20,20]
        sampler = dgl.dataloading.NeighborSampler(fanouts)
        benign_loader = dgl.dataloading.DataLoader(
            g, benign_domain.cpu(), sampler, batch_size=2048, shuffle=False, drop_last=False, num_workers=0
        )     

        with th.no_grad():
            for sample_data in tqdm.tqdm(benign_loader, desc="Computing benign center"):
                inputs, seeds, blocks = sample_data
                seeds = seeds.long()
                seeds = inv_target[seeds]
                feats = embed_layer(blocks[0].srcdata[dgl.NID], blocks[0].srcdata['ntype'], blocks[0].srcdata['type_id'], node_feats)
                logits,hidden_status = model(blocks, feats)
                benign_logits_batch.append(hidden_status)
        
    benign_logits_batch = th.cat(benign_logits_batch)
    C_benign = th.mean(benign_logits_batch, dim=0)


    return C_mal, C_benign


def run(proc_id, n_gpus, n_cpus, args, devices, dataset, queue=None):
    dev_id = 0
    g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg = dataset
    gen_norm(g)     # test, wy, 2023/3/27
    
    # 初始化伪标签为全零（第1阶段不使用center loss）
    num_domains = g.number_of_nodes()
    pseudo_labels = th.zeros(num_domains, 2, device=device)
    
    # 训练阶段控制
    training_phase = 1  # 1: 纯分类训练, 2: 引入center loss
    best_classification_f1 = 0.0
    best_classification_epoch = -1
    center_loss_introduced = False
    
    # 记录最佳性能用于伪标签更新
    best_performance = 0.0
    last_update_epoch = -1
    

    
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
    
    dense_params = list(model.parameters())
    optimizer = th.optim.Adam(dense_params, lr=args.lr, weight_decay=args.l2norm)
    embs = list(embed_layer.embeds.parameters())
    emb_optimizer = th.optim.Adam(embs, lr=args.sparse_lr) if len(embs) > 0 else None
    inv_target = inv_target.to(device)
    labels = labels.to(device)
    
    for epoch in range(args.n_epochs):
        print(f"\n=== Epoch {epoch} - Phase {training_phase} ===")
        
        # 阶段1：纯分类训练，达到最佳性能后进入阶段2
        if training_phase == 1 and not center_loss_introduced:
            print("Phase 1: Pure classification training (no center loss)")
            
            # 第1个epoch结束后生成初始伪标签（仅用于监控，不用于训练）
            if epoch == 1:
                print(f"Generating initial pseudo labels for monitoring (not used in training)")
                pseudo_labels = generate_pseudo_labels(
                    model, embed_layer, g, node_feats, inv_target, 
                    target_idx, train_idx, device, args.confidence_threshold
                )
                print(f"Generated {th.sum(pseudo_labels.sum(dim=1) > 0).item()} initial pseudo labels")
        
        # 阶段2：引入center loss后的训练
        elif training_phase == 2 and center_loss_introduced:
            print("Phase 2: Training with center loss")
            
            # 基于性能提升更新伪标签
            if epoch % args.pseudo_update_freq == 0:
                print(f"Checking for pseudo label updates...")
        
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
            
            # 只对label为0和1的域名计算cross-entropy损失
            batch_labels = labels[seeds].to(device)
            valid_label_mask = (batch_labels == 0) | (batch_labels == 1)
            ood_mask = (batch_labels >= 2)

            if valid_label_mask.sum() > 0:
                cls_loss = F.cross_entropy(logits[valid_label_mask], batch_labels[valid_label_mask])
            else:
                cls_loss = th.tensor(0.0, device=device, requires_grad=True)
            
            if ood_mask.sum() > 0:
                ood_loss_value = ood_loss(logits[ood_mask], batch_labels[ood_mask])
            else:
                ood_loss_value = th.tensor(0.0, device=device)
              
            
            # 阶段1：不使用center loss
            if training_phase == 1:
                center_loss = th.tensor(0.0, device=device, requires_grad=True)
            # 阶段2：使用center loss
            else:
                center_loss = my_loss(hidden_status,C_mal,pseudo_labels,seeds)
            
            # # 添加监督对比学习损失
            # supervised_contrastive_loss = local_loss(hidden_status, labels[seeds], 
            #                                        top_k_positive=args.top_k_positive, 
            #                                        top_k_negative=args.top_k_negative)
            
            # 总损失：阶段1只使用分类和对比损失，阶段2加入center loss
            if training_phase == 1:
                loss = cls_loss+ood_loss_value
            else:
                loss = cls_loss + center_loss
            
            optimizer.zero_grad()
            if emb_optimizer is not None:
                emb_optimizer.zero_grad()

            loss.backward()
            if emb_optimizer is not None:
                emb_optimizer.step()

            optimizer.step()

            if i % 100 == 0 and proc_id == 0:
                f1, precision, roc_auc, recall,fpr,tpr,thresholds = score(logits, labels[seeds].to(device),0.5)

                print(
                    "Epoch {} | Phase {} | Loss: {:.4f} | f1: {:.4f} | roc_auc: {:.4f} | precision: {:.4f} | recall: {:.4f} | cls_loss: {:.4f} | center_loss: {:.4f}".
                    format(epoch, training_phase, loss.item(), f1, roc_auc, precision, recall, cls_loss.item(), center_loss.item()))



        if (queue is not None) or (proc_id == 0):
            val_logits, val_seeds = evaluate(model, embed_layer,val_loader,
                                            node_feats, inv_target)

            if queue is not None:
                queue.put((val_logits, val_seeds))

            # gather evaluation result from multiple processes
            if proc_id == 0:
                val_labels = labels[val_seeds].cpu()
                valid_val_mask = (val_labels == 0) | (val_labels == 1)
                if valid_val_mask.sum() > 0:
                    val_loss = F.cross_entropy(val_logits[valid_val_mask], val_labels[valid_val_mask]).item()
                    val_acc = th.sum(val_logits[valid_val_mask].argmax(dim=1) == val_labels[valid_val_mask]).item() / valid_val_mask.sum().item()
                else:
                    val_loss = 0.0
                    val_acc = 0.0
                val_seeds = val_seeds.to(device)
                val_logits = val_logits.to(device)
                f1, precision,roc_auc,recall,fpr,tpr,thresholds = score(val_logits, labels[val_seeds],0.5)

                # 阶段1：检查是否达到最佳分类性能
                if training_phase == 1 and not center_loss_introduced:
                    if f1 > best_classification_f1:
                        best_classification_f1 = f1
                        best_classification_epoch = epoch
                        print(f"New best classification F1: {f1:.4f} at epoch {epoch}")
                        
                        # 保存最佳分类模型
                        th.save(model,'/data1/hongjiegu/checkpoint/HGSLA/best_classification_model.pt')
                        th.save(embed_layer,'/data1/hongjiegu/checkpoint/HGSLA/best_classification_embed.pt')
                        
                        # 检查是否应该进入阶段2
                    elif f1 <= best_classification_f1:  # 可配置的阈值
                        print(f"Classification performance threshold reached! F1: {f1:.4f}")
                        print("Switching to Phase 2: Introducing center loss")
                        
                        # 进入阶段2
                        training_phase = 2
                        center_loss_introduced = True
                        
                        # 用最佳分类模型计算黑中心和伪标签
                        print("Computing malicious center and pseudo labels with best classification model...")
                        
                        # 加载最佳分类模型
                        best_model = th.load('/data1/hongjiegu/checkpoint/HGSLA/best_classification_model.pt')
                        best_embed = th.load('/data1/hongjiegu/checkpoint/HGSLA/best_classification_embed.pt')
                        
                        # 计算黑中心
                        C_mal,C_benign = cal_malicious_logits_center_with_model(best_model, best_embed, node_feats, g, labels, inv_target)
                        th.save(C_mal,'/data1/hongjiegu/checkpoint/HGSLA/C_mal_best_classification.pt')
                        
                        # 生成伪标签
                        pseudo_labels = generate_pseudo_labels(
                            best_model, best_embed, g, node_feats, inv_target, 
                            target_idx, train_idx, device, args.confidence_threshold
                        )
                        print(f"Generated {th.sum(pseudo_labels.sum(dim=1) > 0).item()} pseudo labels")
                
                # 阶段2：基于性能提升更新伪标签
                elif training_phase == 2 and center_loss_introduced:
                    current_performance = f1
                    if current_performance > best_performance:
                        print(f"Performance improved from {best_performance:.4f} to {current_performance:.4f}, updating pseudo labels and center")
                        
                        # 更新黑中心
     
                        C_mal,C_benign = cal_malicious_logits_center_with_model(model, embed_layer, node_feats, g, labels, inv_target)
                        
                        # 更新伪标签
                        pseudo_labels = generate_pseudo_labels(
                            model, embed_layer, g, node_feats, inv_target, 
                            target_idx, train_idx, device, args.confidence_threshold
                        )
                        
                        best_performance = current_performance
                        print(f"Updated {th.sum(pseudo_labels.sum(dim=1) > 0).item()} pseudo labels")
                    else:
                        early_stop = stopper.step(f1,model,embed_layer)
                        if early_stop:
                            break
                print("ACC: {:.4f}|Validation loss: {:.4f}| Validation f1: {:.4f}| Validation roc_auc: {:.4f} | Validation precision: {:.4f}|Validation recall: {:.4f}".format(val_acc,val_loss, f1,  roc_auc, precision,recall))



def main(args, devices):
    # # load graph data
    g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg = get_g(
        'pdns')
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
    
    # EMA configuration parameters
    parser.add_argument("--ema-decay", type=float, default=0.99,
                        help="EMA decay rate for malicious center tracking")
    parser.add_argument("--ema-min-decay", type=float, default=0.9,
                        help="Minimum EMA decay rate")
    parser.add_argument("--use-ema", default=True, action='store_true',
                        help="Use EMA for end-to-end malicious center tracking")
    
    # Pseudo label configuration parameters
    parser.add_argument("--pseudo-update-freq", type=int, default=5,
                        help="Frequency of pseudo label updates (every N epochs)")
    parser.add_argument("--confidence-threshold", type=float, default=0,
                        help="Confidence threshold for pseudo label generation")
    parser.add_argument("--performance-threshold", type=float, default=0,
                        help="Performance improvement threshold for pseudo label updates")
    
    # Supervised contrastive learning parameters
    parser.add_argument("--contrastive-weight", type=float, default=1,
                        help="Weight for supervised contrastive learning loss")
    
    # Hard mining parameters for contrastive learning
    parser.add_argument("--top-k-positive", type=int, default=50,
                        help="Number of hardest positive pairs (largest distance) to select for contrastive learning")
    parser.add_argument("--top-k-negative", type=int, default=50,
                        help="Number of hardest negative pairs (smallest distance) to select for contrastive learning")
    

    
    parser.set_defaults(validation=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = config()

    devices = list(map(int, args.gpu.split(',')))
    print(args)
    main(args, devices)