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
def init_drop(dropout):
    if dropout > 0:
        return nn.Dropout(dropout)
    else:
        return lambda x: x



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
            model_path = os.path.join(self.save_dir, f"{model_name}_fs.pt")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model, model_path)
            if embed_layer is not None:
                embed_name = f"{self.benchmark}-{self.baselines}"
                if self.zero_shot:
                    embed_name += "_zero-shot"
                embed_path = os.path.join(self.save_dir, f"{embed_name}_fs_Embedding.pt")
                torch.save(embed_layer, embed_path)
            self.best_auc = max(auc, self.best_auc)
            self.counter = 0
        return self.early_stop

def get_g(benchmark, args=None):
    edge = {}
    u = {}
    v = {}
    
    # 根据benchmark设置数据路径和参数
    if benchmark == 'pdns':
        data_path = '/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl'
        domain = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/domain.csv")
        ip = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/ips.csv")
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/feats/domain_feats_rand.pt')
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/feats/ip_feats_rand.pt')
        test_data = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/ood_pdns.csv')
        num_of_ntype = 2
        num_rels = 6
        
        # 加载边数据
        for i in range(1, 4):
            print("load data_" + str(i))
            edge[i] = pd.read_csv(data_path + "/edge_" + str(i)+'.csv')
            u[i] = edge[i]["source"].values.astype(int)
            v[i] = edge[i]["target"].values.astype(int)
            u[i] = th.from_numpy(u[i])
            v[i] = th.from_numpy(v[i])

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
        g.nodes['domain'].data['label'] = th.tensor(domain['label'])
        g.nodes['ip'].data['label'] = th.full((g.number_of_nodes('ip'),), 2, dtype=th.long)
        g.nodes['ip'].data['test_mask'] = th.full((g.number_of_nodes('ip'),), False)
        g.nodes['ip'].data['val_mask'] = th.full((g.number_of_nodes('ip'),), False)
        g.nodes['domain'].data['test_mask'] = th.tensor(domain['test_mask']).T
        g.nodes['domain'].data['val_mask'] = th.tensor(domain['val_mask']).T

    elif benchmark == 'minta':
        data_path = '/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl'
        domain = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/domain.csv')
        ip = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/ip.csv')
        host = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/host.csv')
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/feats/domain_feats_rand.pt')
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/feats/ip_feats_rand.pt')
        host_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/feats/host_feats_rand.pt')
        test_data = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/ood_minta.csv')
        num_of_ntype = 3
        num_rels = 8
        print(domain.val_mask.sum())
        for i in range(1, 5):
            print("load data_" + str(i))
            edge[i] = pd.read_csv(data_path + "/edge_" + str(i)+'.csv')
            u[i] = edge[i]["index_x"].values.astype(int)
            v[i] = edge[i]["index_y"].values.astype(int)
            u[i] = th.from_numpy(u[i])
            v[i] = th.from_numpy(v[i])
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
        g.nodes['ip'].data['feat'] = th.tensor(np.array(ip_feats)).to(th.float32)
        g.nodes['host'].data['feat'] = th.tensor(np.array(host_feats)).to(th.float32)
        g.nodes['domain'].data['feat'] = th.tensor(np.array(domain_feats)).to(th.float32)
        g.nodes['domain'].data['label'] = th.tensor(domain['label'])
        g.nodes['domain'].data['test_mask'] = th.tensor(domain['test_mask']).T
        g.nodes['domain'].data['val_mask'] = th.tensor(domain['val_mask']).T
        g.nodes['ip'].data['label'] = th.full((g.number_of_nodes('ip'),), 2, dtype=th.long)
        g.nodes['host'].data['label'] = th.full((g.number_of_nodes('host'),), 2, dtype=th.long)
        g.nodes['ip'].data['test_mask'] = th.full((g.number_of_nodes('ip'),), False)
        g.nodes['ip'].data['val_mask'] = th.full((g.number_of_nodes('ip'),), False)
        g.nodes['host'].data['test_mask'] = th.full((g.number_of_nodes('host'),), False)
        g.nodes['host'].data['val_mask'] = th.full((g.number_of_nodes('host'),), False)
        
    elif benchmark in ['iochg', 'iochg_small']:
        # 统一处理iochg和iochg_small
        if benchmark == 'iochg':
            data_path = '/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2022-1114-1605-a'
            domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/domain_feats_rand.pt')
            ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/ip_feats_rand.pt')
            url_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/url_feats_rand.pt')
            file_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/file_feats_rand.pt')
            test_data = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/iochg-dgl/ood_malicious_2022_1114_1605.csv', header=0)
        else:  # iochg_small
            data_path = '/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2023-1212-1537-zcf'
            domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/domain_feats_rand.pt')
            ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/ip_feats_rand.pt')
            url_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/url_feats_rand.pt')
            file_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/file_feats_rand.pt')
            test_data = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/iochg-dgl/ood_malicious_1537.csv')

        domain = pd.read_csv(data_path + '/nodes/domain_new')
        file = pd.read_csv(data_path + '/nodes/file_nodes')
        url = pd.read_csv(data_path + '/nodes/url_nodes')
        ip = pd.read_csv(data_path + '/nodes/ip_new')
        
        num_of_ntype = 4
        num_rels = 20
        
        # 加载边数据
        for i in range(1, 11):
            print("load data_" + str(i))
            if benchmark == 'iochg':
                edge[i] = pd.read_csv(data_path + "/edges/edges_" + str(i))
            else:  # iochg_small
                edge[i] = pd.read_csv(data_path + "/edges/edges_" + str(i)+'.csv')
            u[i] = edge[i]["index_x"].values.astype(int)
            v[i] = edge[i]["index_y"].values.astype(int)
            u[i] = th.from_numpy(u[i])
            v[i] = th.from_numpy(v[i])

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
            # 反向边
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
        
        # 设置节点标签和掩码
        g.nodes['domain'].data['label'] = th.tensor(domain['label'])
        g.nodes['file'].data['label'] = th.tensor(file['label'])
        g.nodes['url'].data['label'] = th.tensor(url['label'])
        g.nodes['ip'].data['label'] = th.zeros(g.number_of_nodes('ip')).long()
        
        # 设置训练/测试/验证掩码
        g.nodes['domain'].data['test_mask'] = th.tensor(domain['test_mask']).T
        g.nodes['domain'].data['val_mask'] = th.tensor(domain['val_mask']).T
        
        g.nodes['file'].data['test_mask'] = th.tensor(file['test_mask']).T
        g.nodes['file'].data['val_mask'] = th.tensor(file['val_mask']).T
        
        g.nodes['url'].data['train_mask'] = th.tensor(url['with_label_train']).T
        g.nodes['url'].data['test_mask'] = th.tensor(url['with_label_test']).T
        g.nodes['url'].data['val_mask'] = th.tensor(url['with_label_valid']).T
        
        g.nodes['ip'].data['train_mask'] = th.full((g.number_of_nodes('ip'),), False)
        g.nodes['ip'].data['test_mask'] = th.full((g.number_of_nodes('ip'),), False)
        g.nodes['ip'].data['val_mask'] = th.full((g.number_of_nodes('ip'),), False)

    # 处理shot ratio采样（对所有数据集）


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

    shot_ratio = args.shot_ratio
    # 获取train_mask_finetune中ood_malicious_mask为True的索引
    finetune_ood_indices = domain[(domain['train_mask_finetune']) & (domain['ood_malicious_mask'])].index
    n_finetune_ood = len(finetune_ood_indices)
    sampled_count = int(n_finetune_ood * shot_ratio)
    np.random.seed(42)
    sampled_finetune_ood_indices = np.random.choice(finetune_ood_indices, sampled_count, replace=False)

    # 获取train_mask_finetune中ood_malicious_mask为False的索引（全部保留）
    finetune_non_ood_indices = domain[(domain['train_mask_finetune']) & (~domain['ood_malicious_mask'])].index

    # 新建mask，包含采样的ood数据和非ood数据
    domain['finetune_ood_mask'] = False
    domain.loc[sampled_finetune_ood_indices, 'finetune_ood_mask'] = True
    domain.loc[finetune_non_ood_indices, 'finetune_ood_mask'] = True
    g.nodes['domain'].data['train_mask'] = th.tensor(domain['finetune_ood_mask']).T
    # 打印统计信息
    print(f"train_mask_finetune中ood_malicious数量: {n_finetune_ood}")
    print(f"按shot_ratio={shot_ratio}采样后ood数量: {sampled_count}")
    print(f"train_mask_finetune中非ood数量: {len(finetune_non_ood_indices)}")
    print(f"新mask总数量: {domain['finetune_ood_mask'].sum()}")
        

    labels = g.nodes['domain'].data['label']
    train_mask = g.nodes['domain'].data['train_mask']
    test_mask = g.nodes['domain'].data['test_mask']
    val_mask = g.nodes['domain'].data['val_mask']
    train_idx = g.nodes('domain')[train_mask]
    test_idx = g.nodes('domain')[test_mask]
    val_idx = g.nodes('domain')[val_mask]
    num_classes = 2

    return hg, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,g
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

def config():
    parser = argparse.ArgumentParser(description='Select baselines and benchmark')
    #     parser.add_argument('-s', '--seed', type=int, default=2,
    #                         help='Random seed')
    parser.add_argument("--baselines", type=str, default='hgt',
                        help="dropout probability")
    parser.add_argument("--benchmark", type=str, default='iochg',
                        help="number of hidden units")
    parser.add_argument("--zero_shot", action='store_true', default=False,
                        help="whether to use zero-shot training (set ood_malicious_domain labels to 1)")

    args = parser.parse_args()
    return args
def rphgnn_trainer():
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
    num_epochs = 100
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


    model_name = os.path.join('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint',benchmark+'_'+baselines+'.pt')
    save_model_name = os.path.join('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint',benchmark+'_'+baselines+'_fs.pt')
    model.load_state_dict(torch.load(model_name))

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    uuid = "{}_{}".format(timestamp, shortuuid.uuid())
    tmp_output_fname = "{}.json.tmp".format(uuid)
    print("number of params:", sum(p.numel() for p in model.parameters()))

    data_loader_device = device

    def eval_loader(model, data_loader):
        model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for batch in data_loader:
                x, y = batch
                batch_logits = model(x)
                # 使用与训练过程中evaluate方法相同的预测逻辑
                if model.multi_label:
                    batch_y_pred = (torch.sigmoid(batch_logits) > 0.5).float()
                else:
                    batch_y_pred = torch.argmax(batch_logits, dim=-1)
                
                # 对于AUC计算，需要概率值
                if not model.multi_label:
                    batch_probs = torch.softmax(batch_logits, dim=-1)
                    # 取正类的概率
                    batch_probs = batch_probs[:, 1] if batch_probs.shape[1] == 2 else batch_probs[:, 1]
                else:
                    batch_probs = torch.sigmoid(batch_logits)
                
                all_labels.append(y.cpu())
                all_preds.append(batch_y_pred.cpu())
                all_probs.append(batch_probs.cpu())
        
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_probs = torch.cat(all_probs).numpy()
        
        f1 = f1_score(all_labels, all_preds)
        try:
            # 使用概率值计算AUC
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = float('nan')
        return f1, auc


    def train_and_eval():
        
        train_h_list_list, train_y = nested_gather([target_h_list_list, torch_y], train_index)
        valid_h_list_list, valid_y = nested_gather([target_h_list_list, torch_y], valid_index)
        test_h_list_list, test_y = nested_gather([target_h_list_list, torch_y], test_index)


        train_data_loader = NestedDataLoader(
            [train_h_list_list, train_y],
            batch_size=batch_size, shuffle=True, device=data_loader_device
        )




        valid_data_loader =NestedDataLoader(
            [valid_h_list_list, valid_y], 
            batch_size=batch_size, shuffle=False, device=data_loader_device
        )
        test_data_loader = NestedDataLoader(
            [test_h_list_list, test_y], 
            batch_size=batch_size, shuffle=False, device=data_loader_device
        )

        early_stop_strategy = "score"
        early_stop_metric_names = ["accuracy"]


        print("early_stop_metric_names = {}".format(early_stop_metric_names))

        early_stopping_callback = EarlyStoppingCallback(
            early_stop_strategy, early_stop_metric_names, validation_freq, patience, test_data_loader,
            model_save_path=save_model_name
        )
    

        model.fit(
            train_data=train_data_loader,
            epochs=num_epochs,
            validation_data=valid_data_loader,
            validation_freq=validation_freq,
            callbacks=[early_stopping_callback],
        )
# 在train_and_eval()最后加上：
        print("Evaluating on validation set...")
        val_f1, val_auc = eval_loader(model, valid_data_loader)
        print(f"Validation F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        print("Evaluating on test set...")
        test_f1, test_auc = eval_loader(model, test_data_loader)
        print(f"Test F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    train_and_eval()


def slotgat_trainer(g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg,device,benchmark,baselines,zero_shot=False):
    num_heads = 4  # 减少注意力头数
    lr = float(1e-3)
    weight_decay = float(5e-4)
    hidden_dim = 16  # 减少隐藏维度
    num_layers =2 
    slot_aggregator = "SA"
    dl_mode = 'bi'
    class_num=2
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
    net=th.load('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint/pdns-slotgat.pt')
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net = net.to(device)
    fanouts = [20,20,20]
    print(fanouts)
    sampler = dgl.dataloading.NeighborSampler(fanouts,'in')
    # 直接用全局earlyStopping
    stopper = earlyStopping(5, benchmark, baselines, zero_shot=zero_shot)
    # 创建训练数据加载器
    train_loader = DataLoader(
        g,
        target_idx[train_idx.to('cpu')],    # DGL的采样器需要节点在CPU上
        sampler,
        device=device,          # 最终数据转移到该设备
        batch_size=4096,         # 根据GPU内存调整
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
            batch_size=4096,  # 验证可以使用更大的batch
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
        eval_seeds = th.cat(eval_seeds, dim=0)
        return eval_logits,eval_seeds

    for epoch in tqdm(range(500)):
        net.train()
        total_loss = 0
        total_correct = 0
        
        for input_nodes, output_nodes, blocks in tqdm(train_loader):
            optimizer.zero_grad()
            logits, _ = net(blocks, node_feats, e_feat)
            batch_labels = labels[inv_target[output_nodes.cpu()]].to(device)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(output_nodes)
            preds = logits.argmax(dim=1)
            total_correct += (preds == batch_labels).sum().item()
        avg_loss = total_loss / len(train_idx)
        acc = total_correct / len(train_idx)
        print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
        eval_logits,eval_seeds = evaluate(net, g, val_idx,target_idx,inv_target,e_feat,node_feats,labels)
        f1, macro,roc_auc,recall,FDR,TPR,threshold = score(eval_logits,labels[eval_seeds].to(device),0.5)
        print(f1,macro,recall,roc_auc)
        early_stop = stopper.step(f1, net)
        if early_stop:
            break


def evaluate(model, embed_layer, val_loader, node_feats, inv_target):
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
            
            val_logits.append(logits.cpu())
            val_seeds.append(seeds.cpu())
    
    val_logits = th.cat(val_logits, dim=0)
    val_seeds = th.cat(val_seeds, dim=0)
    
    return val_logits, val_seeds


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
def hgt_trainer(g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg,zero_shot=False):
    model = th.load('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint/pdns-hgt.pt').to(device)
    fanouts = [20,20,20]
    sampler = dgl.dataloading.NeighborSampler(fanouts,'in')

    loader = dgl.dataloading.DataLoader(
        g,
        target_idx[train_idx],  # 要训练的数据
        sampler,
        batch_size=2048,
        shuffle=True,
        drop_last=False,
        num_workers=0)

    val_loader = dgl.dataloading.DataLoader(
        g,
        target_idx[val_idx],
        sampler,
        batch_size=2048,
        shuffle=False,
        drop_last=False,
        num_workers=0)

    test_sampler = dgl.dataloading.NeighborSampler(fanouts,'in')
    test_loader = dgl.dataloading.DataLoader(
        g,
        target_idx[test_idx],
        test_sampler,
        batch_size=2048,
        shuffle=False,
        drop_last=False,
        num_workers=0)
    labels = labels.to(device)
    model.to(device)
    embed_layer = th.load('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint/pdns-hgt_Embedding.pt').to(device)

    dense_params = list(model.parameters())
    optimizer = th.optim.Adam(dense_params, lr=0.001,weight_decay=1e-5)
    embs = list(embed_layer.embeds.parameters())
    emb_optimizer = th.optim.Adam(embs, lr=0.001)
    stopper = earlyStopping(5, benchmark, baselines, zero_shot=zero_shot)

    for epoch in range(100):
        model.train()
        embed_layer.train()
        for i, sample_data in enumerate(loader):
            input_nodes, seeds, blocks = sample_data
            seeds = seeds.long()
            seeds = inv_target[seeds]
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'],
                                node_feats).to(device)
            logits = model.forward(blocks,feats)
            loss = F.cross_entropy(logits, labels[seeds].to(device))
            optimizer.zero_grad()
            if emb_optimizer is not None:
                emb_optimizer.zero_grad()
            loss.backward()
            if emb_optimizer is not None:
                emb_optimizer.step()
            optimizer.step()
            if i % 100 == 0 :
                f1, precision, roc_auc, recall,fpr,tpr,thresholds = score(logits,labels[seeds].to(device),0.5)
                print(
                    "Train Loss: {:.4f}| f1: {:.4f}| roc_auc: {:.4f} | precision: {:.4f}| recall: {:.4f}".
                    format(loss.item(), f1, roc_auc,precision,recall))
        val_logits, val_seeds = evaluate(model, embed_layer, val_loader,node_feats, inv_target)
        f1, precision,roc_auc,recall,fpr,tpr,thresholds = score(val_logits, labels[val_seeds],0.5)
        print( " Validation f1: {:.4f}| Validation roc_auc: {:.4f} | Validation precision: {:.4f}|Validation recall: {:.4f}".format(f1,  roc_auc,precision,recall))
        early_stop = stopper.step(f1,model,embed_layer)
        if early_stop:
            break


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
    print(val_mask.sum())
    test_mask = hg.nodes['domain'].data['test_mask'].to(device)
    in_dims = [node_feats[i].shape[1] for i in range(len(node_feats))]

    # 模型参数
    model = th.load('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint/iochg-simplehgn.pt').to(device)
    embed_layer = th.load('/home/hongjiegu/projects/GRAVEL/baselines/checkpoint/iochg-simplehgn_Embedding.pt').to(device)
    dense_params = list(model.parameters())
    optimizer = torch.optim.Adam(dense_params, lr=0.001, weight_decay=1e-5)
    embs = list(embed_layer.embeds.parameters())
    emb_optimizer = torch.optim.Adam(embs, lr=0.001)
    stopper = earlyStopping(5, benchmark, "simplehgn", zero_shot=zero_shot)

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
        f1, precision,roc_auc,recall,fpr,tpr,thresholds = score(val_logits, labels[val_seeds],0.5)
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

        # 计算损失
        loss = F.cross_entropy(logits, labels[seeds].to(device))
        loss.backward()
        optimizer.step()

        emb_optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    
    return total_loss / total_batches



def config():
    parser = argparse.ArgumentParser(description='HGAFA')
    #     parser.add_argument('-s', '--seed', type=int, default=2,
    #                         help='Random seed')
    parser.add_argument("--benchmark", type=str, default='iochg',
                        help="benchmark dataset to use (iochg, iochg_small, minta, pdns)")
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
    parser.add_argument('--shot-ratio', type=float, default=1.0,
                        help='Shot ratio for OOD malicious samples (0.1-1.0)')
    parser.add_argument("--baselines", type=str, default='hgt',
                        help="dropout probability")
    parser.set_defaults(validation=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='6'
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    args = config()
    benchmark  = args.benchmark
    baselines = args.baselines
    g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg = get_g(benchmark,args)
    if baselines == 'slotgat':
        slotgat_trainer(g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg,device,benchmark,baselines)
    if baselines == 'hgt':
        hgt_trainer(g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg)
    if baselines == 'rphgnn':
        rphgnn_trainer()
    # if baselines == 'heco':
    #     heco_trainer(g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels, hg, device, benchmark, baselines)
    # if baselines == 'rmr':
    #     rmr_trainer(g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels, hg, device, benchmark, baselines)
    if baselines == 'simplehgn':
        simplehgnn_trainer(g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels, hg, device, benchmark, baselines)