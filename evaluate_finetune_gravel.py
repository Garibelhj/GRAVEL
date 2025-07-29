#!/usr/bin/env python3
"""
评估微调后的GRAVEL模型在不同benchmark上的性能
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
sys.path.append('/home/hongjiegu/projects/GRAVEL/baselines')
from baseline_trainer import simplehgnn_trainer

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

os.environ['CUDA_VISIBLE_DEVICES']='2'
device = th.device("cuda" if th.cuda.is_available() else "cpu")

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
        g.nodes['domain'].data['test_mask'] = th.tensor(domain['with_label_test']).T
        g.nodes['domain'].data['val_mask'] = th.tensor(domain['with_label_valid']).T

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
        g.nodes['domain'].data['test_mask'] = th.tensor(domain['with_label_test']).T
        g.nodes['domain'].data['val_mask'] = th.tensor(domain['with_label_valid']).T
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
        print(domain[domain.val_mask==True].label.value_counts())
        print(domain[domain.test_mask==True].label.value_counts())
        print(len(domain))
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
        g.nodes['domain'].data['train_mask'] = th.tensor(domain['with_label_train']).T
        g.nodes['domain'].data['test_mask'] = th.tensor(domain['with_label_test']).T
        g.nodes['domain'].data['val_mask'] = th.tensor(domain['with_label_valid']).T
        
        g.nodes['file'].data['train_mask'] = th.tensor(file['with_label_train']).T
        g.nodes['file'].data['test_mask'] = th.tensor(file['with_label_test']).T
        g.nodes['file'].data['val_mask'] = th.tensor(file['with_label_valid']).T
        
        g.nodes['url'].data['train_mask'] = th.tensor(url['with_label_train']).T
        g.nodes['url'].data['test_mask'] = th.tensor(url['with_label_test']).T
        g.nodes['url'].data['val_mask'] = th.tensor(url['with_label_valid']).T
        
        g.nodes['ip'].data['train_mask'] = th.full((g.number_of_nodes('ip'),), False)
        g.nodes['ip'].data['test_mask'] = th.full((g.number_of_nodes('ip'),), False)
        g.nodes['ip'].data['val_mask'] = th.full((g.number_of_nodes('ip'),), False)



    print(g)   

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

    g.nodes['domain'].data['label'] = th.tensor(domain['label'])
    g.nodes['domain'].data['train_mask'] = th.tensor(domain['train_mask_finetune']).T
    g.nodes['domain'].data['test_mask'] = th.tensor(domain['test_mask']).T
    g.nodes['domain'].data['val_mask'] = th.tensor(domain['val_mask']).T

    labels = g.nodes['domain'].data['label']
    train_mask = g.nodes['domain'].data['train_mask']
    test_mask = g.nodes['domain'].data['test_mask']
    val_mask = g.nodes['domain'].data['val_mask']
    train_idx = g.nodes('domain')[train_mask]
    test_idx = g.nodes('domain')[test_mask]
    val_idx = g.nodes('domain')[val_mask]
    num_classes = 2

    return hg, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,g

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

def evaluate(model, embed_layer, eval_loader, node_feats, inv_target):
    """评估模型在给定数据加载器上的性能"""
    model.eval()
    embed_layer.eval()
    eval_logits = []
    eval_seeds = []

    with th.no_grad():
        th.cuda.empty_cache()
        for sample_data in tqdm.tqdm(eval_loader):
            inputs, seeds, blocks = sample_data
            seeds = seeds.long().cpu()
            seeds = inv_target[seeds]

            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'],
                                node_feats)

            logits, hidden_status = model(blocks, feats)

            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(seeds.cpu().detach())

    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)
    
    return eval_logits, eval_seeds

def calculate_metrics(logits, labels, threshold=0.5):
    """计算各种评估指标"""
    Softmax = nn.Softmax(dim=1)
    probs = Softmax(logits)
    predictions = (probs[:, 1] > threshold).long()
    
    # 转换为numpy
    pred_np = predictions.numpy()
    true_np = labels.numpy()
    
    # 计算指标
    f1 = f1_score(true_np, pred_np, average='macro')
    precision = precision_score(true_np, pred_np, average='macro', zero_division=0)
    recall = recall_score(true_np, pred_np, average='macro', zero_division=0)
    
    # 计算AUC
    try:
        auc = roc_auc_score(true_np, probs[:, 1].numpy())
    except:
        auc = 0.0
    
    # 计算准确率
    accuracy = (pred_np == true_np).mean()
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'accuracy': accuracy
    }

def evaluate_benchmark(benchmark, args):
    """评估指定benchmark的性能"""
    print(f"Evaluating benchmark: {benchmark}")
    
    # 加载数据
    g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels, hg = get_g(benchmark, args)
    
    # 创建数据加载器
    gen_norm(g)
    fanouts = [int(fanout) for fanout in args.fanout.split(',')]
    sampler = dgl.dataloading.NeighborSampler(fanouts, 'in')
    
    # 验证集加载器
    val_loader = dgl.dataloading.DataLoader(
        g,
        target_idx[val_idx],
        sampler,
        device=device,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    
    # 测试集加载器
    test_loader = dgl.dataloading.DataLoader(
        g,
        target_idx[test_idx],
        sampler,
        device=device,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    
    # 加载模型
    embed_layer = th.load('/data1/hongjiegu/checkpoint/HGSLA/GRAVEL_Embedding_small_end2end_v2.pt').to(device)
    model = th.load('/data1/hongjiegu/checkpoint/HGSLA/GRAVEL_small_end2end_v2.pt').to(device)

    
    # 根据benchmark加载对应的预训练模型

    
    # 评估验证集
    print("Evaluating validation set...")
    val_logits, val_seeds = evaluate(model, embed_layer, val_loader, node_feats, inv_target)
    val_labels = labels[val_seeds]
    val_metrics = calculate_metrics(val_logits, val_labels)
    
    # 评估测试集
    print("Evaluating test set...")
    test_logits, test_seeds = evaluate(model, embed_layer, test_loader, node_feats, inv_target)
    test_labels = labels[test_seeds]
    test_metrics = calculate_metrics(test_logits, test_labels)
    
    # 打印结果
    print(f"\n=== {benchmark.upper()} Results ===")
    print(f"Validation Set:")
    for metric, value in val_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print(f"\nTest Set:")
    for metric, value in test_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    return {
        'benchmark': benchmark,
        'validation': val_metrics,
        'test': test_metrics
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate GRAVEL model on different benchmarks')
    parser.add_argument("--benchmark", type=str, required=True,
                        help="benchmark dataset to evaluate (iochg, iochg_small, minta, pdns)")
    parser.add_argument("--n-hidden", type=int, default=64, help="number of hidden units")
    parser.add_argument("--n-layers", type=int, default=3, help="number of propagation rounds")
    parser.add_argument("--n-bases", type=int, default=2, help="number of filter weight matrices")
    parser.add_argument("--dropout", type=float, default=0, help="dropout probability")
    parser.add_argument("--use-self-loop", default=True, action='store_true', help="include self feature")
    parser.add_argument("--layer-norm", default=False, action='store_true', help='Use layer norm')
    parser.add_argument("--dgl-sparse", default=False, action='store_true', help='Use sparse embedding')
    parser.add_argument("--fanout", type=str, default="20,20,20", help="Fan-out of neighbor sampling")
    parser.add_argument("--eval-batch-size", type=int, default=1024, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--shot-ratio", type=float, default=1.0, help='Shot ratio for OOD malicious samples')
    
    args = parser.parse_args()
    
    # 评估指定benchmark
    results = evaluate_benchmark(args.benchmark, args)
    
    print(f"\nEvaluation completed for {args.benchmark}!")

if __name__ == '__main__':
    main()