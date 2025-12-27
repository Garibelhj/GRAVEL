#-*-encoding:utf-8-*-
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import argparse
import numpy as np
import pandas as pd
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import warnings
# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from model.GRAVEL import RelGraphEmbedLayer
from model.GRAVEL import EntityClassify
import tqdm
from dgl.nn import LabelPropagation

warnings.filterwarnings("ignore")
device = th.device("cuda" if th.cuda.is_available() else "cpu")

TAU=0.6



def get_g(benchmark, args=None):
    edge = {}
    u = {}
    v = {}
    
    # 根据benchmark设置数据路径和参数
    if benchmark == 'pdns':
        data_path = os.path.join(SCRIPT_DIR, 'dataset', 'pdns-dgl')
        domain = pd.read_csv(os.path.join(data_path, 'domain.csv'))
        ip = pd.read_csv(os.path.join(data_path, 'ips.csv'))
        domain_feats = th.load(os.path.join(data_path, 'feats', 'domain_feats_rand.pt'))
        ip_feats = th.load(os.path.join(data_path, 'feats', 'ip_feats_rand.pt'))
        test_data = pd.read_csv(os.path.join(data_path, 'ood_pdns.csv'))
        num_of_ntype = 2
        num_rels = 6
        
        # 加载边数据
        for i in range(1, 4):
            print("load data_" + str(i))
            edge[i] = pd.read_csv(os.path.join(data_path, f'edge_{i}.csv'))
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
        data_path = os.path.join(SCRIPT_DIR, 'dataset', 'minta-dgl')
        domain = pd.read_csv(os.path.join(data_path, 'domain.csv'))
        ip = pd.read_csv(os.path.join(data_path, 'ip.csv'))
        host = pd.read_csv(os.path.join(data_path, 'host.csv'))
        domain_feats = th.load(os.path.join(data_path, 'feats', 'domain_feats_rand.pt'))
        ip_feats = th.load(os.path.join(data_path, 'feats', 'ip_feats_rand.pt'))
        host_feats = th.load(os.path.join(data_path, 'feats', 'host_feats_rand.pt'))
        test_data = pd.read_csv(os.path.join(data_path, 'ood_minta.csv'))
        num_of_ntype = 3
        num_rels = 8

        for i in range(1, 5):
            print("load data_" + str(i))
            edge[i] = pd.read_csv(os.path.join(data_path, f'edge_{i}.csv'))
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
            data_path = os.path.join(SCRIPT_DIR, 'dataset', 'iochg-dgl')
            domain_feats = th.load(os.path.join(data_path, 'feats', 'domain_feats_rand.pt'))
            ip_feats = th.load(os.path.join(data_path, 'feats', 'ip_feats_rand.pt'))
            url_feats = th.load(os.path.join(data_path, 'feats', 'url_feats_rand.pt'))
            file_feats = th.load(os.path.join(data_path, 'feats', 'file_feats_rand.pt'))
            test_data = pd.read_csv(os.path.join(data_path, 'ood_malicious_2022_1114_1605.csv'), header=0)
        else:  # iochg_small
            data_path = os.path.join(SCRIPT_DIR, 'dataset', 'iochg_small-dgl')
            domain_feats = th.load(os.path.join(data_path, 'feats', 'domain_feats_rand.pt'))
            ip_feats = th.load(os.path.join(data_path, 'feats', 'ip_feats_rand.pt'))
            url_feats = th.load(os.path.join(data_path, 'feats', 'url_feats_rand.pt'))
            file_feats = th.load(os.path.join(data_path, 'feats', 'file_feats_rand.pt'))
            test_data = pd.read_csv(os.path.join(SCRIPT_DIR, 'dataset', 'iochg-dgl', 'ood_malicious_1537.csv'))

        domain = pd.read_csv(os.path.join(data_path, 'nodes', 'domain_new.csv'))
        file = pd.read_csv(os.path.join(data_path, 'nodes', 'file_nodes.csv'))
        url = pd.read_csv(os.path.join(data_path, 'nodes', 'url_nodes.csv'))
        ip = pd.read_csv(os.path.join(data_path, 'nodes', 'ip_new.csv'))
        
        num_of_ntype = 4
        num_rels = 20
        
        # 加载边数据
        for i in range(1, 11):
            print("load data_" + str(i))
            edge[i] = pd.read_csv(os.path.join(data_path, 'edges', f'edges_{i}.csv'))
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

    # 处理shot ratio采样（对所有数据集）


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



class earlyStopping(object):
    def __init__(self, patience=10, checkpoint_dir=None, benchmark=None):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.best_auc = 0
        self.checkpoint_dir = checkpoint_dir or os.path.join(SCRIPT_DIR, 'checkpoint')
        self.benchmark = benchmark or 'finetune'
    def step(self, auc, model, embed_layer):
        if auc < self.best_auc:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            model_filename = os.path.join(self.checkpoint_dir, f'{self.benchmark}_finetune_model.pt')
            embed_filename = os.path.join(self.checkpoint_dir, f'{self.benchmark}_finetune_embed.pt')
            th.save(model, model_filename)
            th.save(embed_layer, embed_filename)
            self.best_auc = np.max((auc, self.best_auc))
            self.counter = 0
        return self.early_stop
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


def score(logits, labels, thredhold):
    Softmax = nn.Softmax(dim=1)
    true_labels =labels[((labels==0)|(labels==1))]
    logits = logits[((labels==0)|(labels==1))]
    logits = Softmax(logits)
    prob = logits[:,1].cpu().detach().numpy()
    
    # 处理 NaN 和 Inf 值
    # 检查是否有 NaN 或 Inf
    if np.isnan(prob).any() or np.isinf(prob).any():
        print(f"Warning: Found NaN or Inf in probabilities. Replacing with 0.5")
        prob = np.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
    
    prediction = (prob>thredhold)
    true_labels = true_labels.long().cpu().numpy()  
    try:
        f1 = f1_score(true_labels, prediction)
    except:
        f1 = 0.0
        pass

    try:
        macro_f1 = f1_score(true_labels, prediction, average='macro')
    except:
        macro_f1 = 0.0
        pass


    try:
        macro = precision_score(true_labels, prediction)
        
    except:
        macro = 0.0
        pass
    try:

        roc_auc = roc_auc_score(true_labels,prob)
    except ValueError:
        roc_auc = 0.5
    try:
        recall=recall_score(true_labels,prediction)
    except:
        recall = 0.0
        pass
    
    try:
        FDR,TPR,threshold = precision_recall_curve(true_labels,prob)
        FDR = 1-FDR
    except ValueError as e:
        print(f"Warning: precision_recall_curve failed: {e}")
        # 返回默认值
        FDR = np.array([0.0])
        TPR = np.array([1.0])
        threshold = np.array([0.5])
    
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
            _, seeds, blocks = sample_data
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



def run(proc_id, n_gpus, n_cpus, args, devices, dataset, queue=None):
    dev_id = 0
    g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels,hg = dataset

    gen_norm(g)

    checkpoint_dir = os.path.join(SCRIPT_DIR, 'checkpoint')
    stopper = earlyStopping(5, checkpoint_dir=checkpoint_dir, benchmark=args.benchmark)
    fanouts = [int(fanout) for fanout in args.fanout.split(',')]
    sampler = dgl.dataloading.NeighborSampler(fanouts,'in')
    loader = dgl.dataloading.DataLoader(
        g,
        target_idx[train_idx],  # 要训练的数据
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # validation sampler
    val_loader = dgl.dataloading.DataLoader(
        g,
        target_idx[val_idx],
        sampler,
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
    # 根据benchmark加载对应的预训练模型
    checkpoint_dir = os.path.join(SCRIPT_DIR, 'checkpoint')
    if args.benchmark == 'iochg':
        embed_layer = th.load(os.path.join(checkpoint_dir, 'iochg_pretrained_gravel_embed.pt')).to(device)
        model = th.load(os.path.join(checkpoint_dir, 'iochg_pretrained_gravel_model.pt')).to(device)
    elif args.benchmark == 'minta':
        embed_layer = th.load(os.path.join(checkpoint_dir, 'minta_pretrained_gravel_embed.pt')).to(device)
        model = th.load(os.path.join(checkpoint_dir, 'minta_pretrained_gravel_model.pt')).to(device)
    elif args.benchmark == 'pdns':
        embed_layer = th.load(os.path.join(checkpoint_dir, 'pdns_pretrained_gravel_embed.pt')).to(device)
        model = th.load(os.path.join(checkpoint_dir, 'pdns_pretrained_gravel_model.pt')).to(device)

    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")


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

            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'],
                                node_feats)
            logits, hidden_status = model(blocks, feats)
            loss = F.cross_entropy(logits, labels[seeds])
            optimizer.zero_grad()
            if emb_optimizer is not None:
                emb_optimizer.zero_grad()

            loss.backward()
            if emb_optimizer is not None:
                emb_optimizer.step()

            optimizer.step()

        val_logits, val_seeds = evaluate(model, embed_layer, val_loader, node_feats, inv_target)
        f1, macro,roc_auc,recall,FDR,TPR,threshold = score(val_logits, labels[val_seeds].cpu(),0.5)
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Best F1: {f1:.4f}")
        # print(f"Best threshold: {best_threshold:.4f}")
        early_stop = stopper.step(f1,model=model,embed_layer=embed_layer)
        if early_stop:
            break

def main(args, devices):
    g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels, hg = get_g(args.benchmark, args)

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
    parser.add_argument("--benchmark", type=str, default='pdns',
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
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Mini-batch size. ")
    parser.add_argument("--eval-batch-size", type=int, default=1024,
                        help="Mini-batch size. ")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for dataloader.")
    parser.add_argument("--dgl-sparse", default=False, action='store_true',
                        help='Use sparse embedding for node embeddings.')
    parser.add_argument('--layer-norm', default=False, action='store_true',
                        help='Use layer norm')
    parser.add_argument('--shot-ratio', type=float, default=1.0,
                        help='Shot ratio for OOD malicious samples (0.1-1.0)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = config()

    devices = list(map(int, args.gpu.split(',')))
    print(args)
    main(args, devices)