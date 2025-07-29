import wandb
import torch
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn
import torch as th
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
import os
os.environ['CUDA_VISIBLE_DEVICES']='1s'
device = th.device("cuda" if th.cuda.is_available() else "cpu")



def get_g(benchmark):
    edge = {}
    u = {}
    v = {}
    if benchmark == 'pdns':
        data_path = '/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl'
        ood_malicious_domain = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/ood_pdns.csv")
        domain = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/domains2.csv")
        ip = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/ips.csv")
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/feats/domain_feats_rand.pt', weights_only=True)
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/feats/ip_feats_rand.pt', weights_only=True)
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
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/feats/domain_feats_rand.pt', weights_only=True)
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/feats/ip_feats_rand.pt', weights_only=True)
        host_feats = th.load('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/feats/host_feats_rand.pt', weights_only=True)
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
        
        # 检查文件是否存在
        domain_file = data_path + '/nodes/domain_new'
        if not os.path.exists(domain_file):
            print(f"警告: 文件 {domain_file} 不存在，尝试添加.csv扩展名")
            domain_file = domain_file + '.csv'
        
        domain = pd.read_csv(domain_file)
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/domain_feats_rand.pt', weights_only=True)
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/ip_feats_rand.pt', weights_only=True)
        url_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/url_feats_rand.pt', weights_only=True)
        file_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2022-1114-1605/file_feats_rand.pt', weights_only=True)
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
            edge_file = data_path + "/edges/edges_" + str(i)
            # 尝试不同的文件扩展名
            if os.path.exists(edge_file + '.csv'):
                edge[i] = pd.read_csv(edge_file + '.csv')
            elif os.path.exists(edge_file):
                edge[i] = pd.read_csv(edge_file)
            else:
                print(f"警告: 边文件 {edge_file} 不存在，跳过")
                continue
                
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
        
        # 检查文件是否存在
        domain_file = data_path + '/nodes/domain_new'
        if not os.path.exists(domain_file):
            print(f"警告: 文件 {domain_file} 不存在，尝试添加.csv扩展名")
            domain_file = domain_file + '.csv'
        
        domain = pd.read_csv(domain_file)
        domain_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/domain_feats_rand.pt', weights_only=True)
        ip_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/ip_feats_rand.pt', weights_only=True)
        url_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/url_feats_rand.pt', weights_only=True)
        file_feats = th.load('/home/hongjiegu/projects/GRAVEL/feats/2023-1212-1537-zcf/file_feats_rand.pt', weights_only=True)
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
            edge_file = data_path + "/edges/edges_" + str(i)
            # 尝试不同的文件扩展名
            if os.path.exists(edge_file + '.csv'):
                edge[i] = pd.read_csv(edge_file + '.csv')
            elif os.path.exists(edge_file):
                edge[i] = pd.read_csv(edge_file)
            else:
                print(f"警告: 边文件 {edge_file} 不存在，跳过")
                continue
                
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
    with_label_mask = th.tensor(domain['with_label'].to_numpy(dtype=bool))
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
    with_label_mask = th.tensor(domain['with_label'].to_numpy(dtype=bool))
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
    g.nodes['domain'].data['train_mask'] = th.tensor(domain['with_label_train'])  # .to(device)
    g.nodes['domain'].data['test_mask'] = th.tensor(domain['to_be_predicted'])  # .to(device)
    g.nodes['domain'].data['val_mask'] = th.tensor(domain['with_label_valid'])  # .to(device)
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


class DGLGCNEncoder(torch.nn.Module):
    """DGL版本的GCN编码器，支持异构图"""
    def __init__(self, input_dim, hidden_dim, activation, num_layers, ntypes=None, etypes=None):
        super(DGLGCNEncoder, self).__init__()
        self.activation = activation()
        self.num_layers = num_layers
        self.ntypes = ntypes
        self.etypes = etypes
        
        # 为每种节点类型创建GCN层
        if ntypes and len(ntypes) > 1:
            # 异构图版本 - 使用HeteroGraphConv
            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                conv_dict = {}
                for etype in etypes:
                    if _ == 0:
                        conv_dict[etype] = dglnn.GraphConv(input_dim, hidden_dim, norm='both')
                    else:
                        conv_dict[etype] = dglnn.GraphConv(hidden_dim, hidden_dim, norm='both')
                self.layers.append(dglnn.HeteroGraphConv(conv_dict, aggregate='sum'))
        else:
            # 同构图版本
            self.layers = nn.ModuleList()
            self.layers.append(dglnn.GraphConv(input_dim, hidden_dim, norm='both'))
            for _ in range(num_layers - 1):
                self.layers.append(dglnn.GraphConv(hidden_dim, hidden_dim, norm='both'))

    def forward(self, g, feat_dict=None):
        if self.ntypes and len(self.ntypes) > 1 and feat_dict:
            # 异构图版本
            h_dict = feat_dict.copy()
            
            # 确保所有节点类型都有特征
            for ntype in self.ntypes:
                if ntype not in h_dict:
                    # 如果没有特征，创建随机特征
                    num_nodes = g.number_of_nodes(ntype)
                    feat_dim = list(h_dict.values())[0].shape[1] if h_dict else 256
                    h_dict[ntype] = torch.randn(num_nodes, feat_dim, device=g.device)
            
            for i, layer in enumerate(self.layers):
                h_dict = layer(g, h_dict)
                if i < len(self.layers) - 1:
                    h_dict = {k: self.activation(v) for k, v in h_dict.items()}
            return h_dict
        else:
            # 同构图版本
            h = feat_dict if feat_dict is not None else g.ndata['feat']
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
                if i < len(self.layers) - 1:
                    h = self.activation(h)
            return h


class DGLCOSTA(torch.nn.Module):
    """兼容DGL的COSTA模型"""
    def __init__(self, encoder, hidden_dim, proj_dim, device, ntypes=None, etypes=None):
        super(DGLCOSTA, self).__init__()
        self.encoder = encoder
        self.device = device
        self.hidden_dim = hidden_dim
        self.ntypes = ntypes
        self.etypes = etypes

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, g, feat_dict=None):
        # 原始图编码
        z = self.encoder(g, feat_dict)
        
        # 创建两个增强版本
        g1 = self._augment_graph(g)
        g2 = self._augment_graph(g)
        
        # 对增强图进行编码
        z1 = self.encoder(g1, feat_dict)
        z2 = self.encoder(g2, feat_dict)
        
        return z, z1, z2

    def _augment_graph(self, g):
        """图增强：随机删除边和特征掩码"""
        # 复制图
        g_aug = g.clone()
        
        # 特征掩码（在边删除之前进行）
        feat_mask_rate = 0.3
        if feat_mask_rate > 0:
            for ntype in g_aug.ntypes:
                if 'feat' in g_aug.nodes[ntype].data:
                    feat = g_aug.nodes[ntype].data['feat']
                    mask = torch.rand(feat.shape) > feat_mask_rate
                    g_aug.nodes[ntype].data['feat'] = feat * mask
        
        # 随机删除边（简化版本，避免复杂的边子图操作）
        edge_drop_rate = 0.2
        if edge_drop_rate > 0:
            # 暂时禁用边删除，因为它会导致节点数量变化
            # 这需要更复杂的处理来保持节点数量一致
            pass
        
        return g_aug

    def project(self, z):
        """投影头"""
        if isinstance(z, dict):
            # 异构图：对每种节点类型分别投影
            z_proj = {}
            for ntype, z_ntype in z.items():
                z_proj[ntype] = F.elu(self.fc1(z_ntype))
                z_proj[ntype] = self.fc2(z_proj[ntype])
            return z_proj
        else:
            # 同构图
            z = F.elu(self.fc1(z))
            return self.fc2(z)


class DualBranchContrast(torch.nn.Module):
    """双分支对比学习"""
    def __init__(self, loss, mode, intraview_negs=False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None):
        l1 = self.loss(anchor=h1, sample=h2)
        l2 = self.loss(anchor=h2, sample=h1)
        return (l1 + l2) * 0.5


class InfoNCE(object):
    """InfoNCE损失函数"""
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample):
        # 计算相似度
        if isinstance(anchor, dict) and isinstance(sample, dict):
            # 异构图：只对目标节点类型计算损失
            total_loss = 0
            count = 0
            for ntype in anchor.keys():
                if ntype in sample:
                    sim = F.cosine_similarity(anchor[ntype], sample[ntype], dim=1) / self.tau
                    exp_sim = torch.exp(sim)
                    log_prob = sim - torch.log(exp_sim.sum(dim=0, keepdim=True))
                    loss = -log_prob.mean()
                    total_loss += loss
                    count += 1
            return total_loss / count if count > 0 else torch.tensor(0.0)
        else:
            # 同构图
            sim = F.cosine_similarity(anchor, sample, dim=1) / self.tau
            exp_sim = torch.exp(sim)
            log_prob = sim - torch.log(exp_sim.sum(dim=0, keepdim=True))
            return -log_prob.mean()

    def __call__(self, anchor, sample):
        return self.compute(anchor, sample)


class DGLCOSTARunner:
    """DGL版本的COSTA运行器"""
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        
    def load_dgl_data(self, hg, node_feats, target_idx, labels, train_mask, val_mask, test_mask, target_ntype='domain'):
        """
        加载DGL数据并转换为COSTA格式
        
        Args:
            hg: DGL异构图
            node_feats: 节点特征字典 {ntype: features}
            target_idx: 目标节点索引
            labels: 节点标签
            train_mask, val_mask, test_mask: 掩码
            target_ntype: 目标节点类型
        """
        self.hg = hg.to(self.device)
        self.node_feats = {ntype: feat.to(self.device) for ntype, feat in node_feats.items()}
        self.target_idx = target_idx.to(self.device)
        self.labels = labels.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.val_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        self.target_ntype = target_ntype
        
        # 设置节点特征
        for ntype, feat in self.node_feats.items():
            self.hg.nodes[ntype].data['feat'] = feat
        
        print(f"数据加载完成: {self.hg.number_of_nodes()} 节点, {self.hg.number_of_edges()} 边")
        
    def train(self):
        """训练COSTA模型"""
        # 创建编码器
        ntypes = list(self.hg.ntypes)
        etypes = list(self.hg.etypes)
        input_dim = self.node_feats[ntypes[0]].shape[1]  # 假设所有节点类型特征维度相同
        
        encoder = DGLGCNEncoder(
            input_dim=input_dim,
            hidden_dim=self.config.get('num_hidden', 256),
            activation=torch.nn.ReLU,
            num_layers=self.config.get('num_layers', 2),
            ntypes=ntypes,
            etypes=etypes
        ).to(self.device)
        
        # 创建COSTA模型
        self.model = DGLCOSTA(
            encoder=encoder,
            hidden_dim=self.config.get('num_hidden', 256),
            proj_dim=self.config.get('num_proj_hidden', 256),
            device=self.device,
            ntypes=ntypes,
            etypes=etypes
        ).to(self.device)
        
        # 对比学习模型
        contrast_model = DualBranchContrast(
            loss=InfoNCE(tau=self.config.get('tau', 0.1)),
            mode='L2L',
            intraview_negs=True
        ).to(self.device)
        
        # 优化器
        optimizer = Adam(self.model.parameters(), lr=self.config.get('learning_rate', 0.001))
        
        # 训练循环
        num_epochs = self.config.get('num_epochs', 1)
        
        with tqdm(total=num_epochs, desc='Training COSTA') as pbar:
            for epoch in range(num_epochs):
                self.model.train()
                optimizer.zero_grad()
                
                # 前向传播
                z, z1, z2 = self.model(self.hg, self.node_feats)
                
                # 只对目标节点类型进行对比学习
                if isinstance(z, dict):
                    z_target = z[self.target_ntype]
                    z1_target = z1[self.target_ntype]
                    z2_target = z2[self.target_ntype]
                else:
                    z_target = z
                    z1_target = z1
                    z2_target = z2
                
                # 应用投影
                h1 = self.model.project(z1_target)
                h2 = self.model.project(z2_target)
                
                # 计算对比损失
                loss = contrast_model(h1, h2)
                
                loss.backward()
                optimizer.step()
                
                # 记录损失
                if wandb.run is not None:
                    wandb.log({'loss': loss.item()})
                
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()
                
                # 定期评估
                if self.config.get('test_every_epoch', False) and epoch % 10 == 0:
                    # 在训练过程中不进行完整评估，只记录损失
                    pass
        
        # 标记模型已训练
        self._model_trained = True
    
    def evaluate(self):
        """评估模型性能"""
        if not hasattr(self, 'model'):
            print("错误: 模型尚未初始化，请先调用train()或load_model()方法")
            return 0.0
            
        self.model.eval()
        with torch.no_grad():
            # 确保模型在评估模式下
            self.model.train(False)
            # 确保模型在正确的设备上
            self.model = self.model.to(self.device)
            z, _, _ = self.model(self.hg, self.node_feats)
            
            if isinstance(z, dict):
                z_target = z[self.target_ntype]
            else:
                z_target = z
            
            # 使用训练好的嵌入进行分类
        train_idx = self.train_mask.to(self.device)
        val_idx = self.val_mask.to(self.device)
        test_idx = self.test_mask.to(self.device)
        
        # 简单的线性分类器
        classifier = torch.nn.Linear(z_target.shape[1], 2).to(self.device)
        optimizer = Adam(classifier.parameters(), lr=0.01)
        
        # 训练分类器
        for epoch in range(50):
            classifier.train()
            optimizer.zero_grad()
            
            # 获取训练数据的嵌入
            z_train = z_target[train_idx].detach()  # 确保不需要梯度
            logits = classifier(z_train)
            loss = F.cross_entropy(logits, self.labels[train_idx].to(self.device))
            
            loss.backward()
            optimizer.step()
                
            # 评估
        classifier.eval()
        with torch.no_grad():
            # 验证集评估
            val_logits = classifier(z_target[val_idx])
            val_pred = torch.argmax(val_logits, dim=1)
            val_accuracy = (val_pred == self.labels[val_idx].to(self.device)).float().mean()

            # 计算验证集F1和AUC
            from sklearn.metrics import f1_score, roc_auc_score
            
            val_y_true = self.labels[val_idx].to(self.device).cpu().numpy()
            val_y_pred = val_pred.cpu().numpy()
            val_f1 = f1_score(val_y_true, val_y_pred, average='macro')
            
            val_probs = torch.softmax(val_logits, dim=1)
            val_y_probs = val_probs[:, 1].cpu().numpy()
            val_auc = roc_auc_score(val_y_true, val_y_probs)
            
            print(f"验证准确率: {val_accuracy.item():.4f}")
            print(f"验证F1分数: {val_f1:.4f}")
            print(f"验证AUC: {val_auc:.4f}")

            # 测试集评估
            test_logits = classifier(z_target[test_idx])
            test_pred = torch.argmax(test_logits, dim=1)
            test_accuracy = (test_pred == self.labels[test_idx].to(self.device)).float().mean()
            
            # 计算测试集F1和AUC
            test_y_true = self.labels[test_idx].to(self.device).cpu().numpy()
            test_y_pred = test_pred.cpu().numpy()
            test_f1 = f1_score(test_y_true, test_y_pred, average='macro')
            
            test_probs = torch.softmax(test_logits, dim=1)
            test_y_probs = test_probs[:, 1].cpu().numpy()
            test_auc = roc_auc_score(test_y_true, test_y_probs)
            
            print(f"测试准确率: {test_accuracy.item():.4f}")
            print(f"测试F1分数: {test_f1:.4f}")
            print(f"测试AUC: {test_auc:.4f}")
            
            return test_accuracy.item()
    
    def save_model(self, path):
        """保存模型权重"""
        if hasattr(self, 'model'):
            torch.save(self.model.state_dict(), path)
            print(f"模型已保存到: {path}")
        else:
            print("错误: 模型尚未训练，无法保存")
    
    def load_model(self, path):
        """加载模型权重"""
        # 如果模型不存在，先创建模型
        if not hasattr(self, 'model'):
            # 创建编码器
            ntypes = list(self.hg.ntypes)
            etypes = list(self.hg.etypes)
            input_dim = self.node_feats[ntypes[0]].shape[1]  # 假设所有节点类型特征维度相同
            
            encoder = DGLGCNEncoder(
                input_dim=input_dim,
                hidden_dim=self.config.get('num_hidden', 256),
                activation=torch.nn.ReLU,
                num_layers=self.config.get('num_layers', 2),
                ntypes=ntypes,
                etypes=etypes
            ).to(self.device)
            
            # 创建COSTA模型
            self.model = DGLCOSTA(
                encoder=encoder,
                hidden_dim=self.config.get('num_hidden', 256),
                proj_dim=self.config.get('num_proj_hidden', 256),
                device=self.device,
                ntypes=ntypes,
                etypes=etypes
            ).to(self.device)
            
            print("模型已初始化")
        
        # 加载权重
        self.model.load_state_dict(torch.load(path, weights_only=True))
        print(f"模型已从 {path} 加载")
        # 标记模型已加载
        self._model_trained = True


def convert_get_g_output_to_dgl_costa(hg, node_feats, target_idx, labels, train_mask, val_mask, test_mask, target_ntype='domain'):
    """
    将get_g函数的输出转换为DGL COSTA格式
    
    Args:
        hg: DGL同构图
        node_feats: 节点特征列表
        target_idx: 目标节点索引
        labels: 节点标签
        train_mask, val_mask, test_mask: 掩码
        target_ntype: 目标节点类型名称
    
    Returns:
        配置好的DGLCOSTARunner实例
    """
    # 创建节点特征字典 - 为所有节点类型创建特征
    node_feats_dict = {}
    ntypes = list(hg.ntypes)
    
    # 如果只有一个节点类型，使用第一个特征
    if len(ntypes) == 1:
        node_feats_dict[ntypes[0]] = node_feats[0]
    else:
        # 为每种节点类型分配特征
        for i, ntype in enumerate(ntypes):
            if i < len(node_feats):
                node_feats_dict[ntype] = node_feats[i]
            else:
                # 如果没有足够的特征，使用第一个特征并调整维度
                feat = node_feats[0]
                if ntype == target_ntype:
                    node_feats_dict[ntype] = feat
                else:
                    # 为非目标节点类型创建随机特征
                    node_feats_dict[ntype] = torch.randn(hg.number_of_nodes(ntype), feat.shape[1])
    
    # 创建运行器
    config = {
        'num_hidden': 256,
        'num_proj_hidden': 256,
        'num_layers': 2,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'tau': 0.1,
        'test_every_epoch': True
    }
    
    runner = DGLCOSTARunner(config, device='cpu')
    runner.load_dgl_data(hg, node_feats_dict, target_idx, labels, train_mask, val_mask, test_mask, target_ntype)
    
    return runner


# 使用示例
if __name__ == "__main__":
    # 示例：如何使用DGL版本的COSTA
    config = {
        'num_hidden': 256,
        'num_proj_hidden': 256,
        'num_layers': 2,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'tau': 0.1
    }
    
    # 在minta上训练模型
    print("=== 在minta数据集上训练模型 ===")
    hg, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels, g = get_g('iochg')
    
    # 转换为DGL COSTA格式
    runner = convert_get_g_output_to_dgl_costa(g, node_feats, target_idx, labels, train_idx, val_idx, test_idx)
    
    # 训练模型
    runner.train()
    
    # 保存训练好的模型
    model_save_path = '/tmp/costa_trained_model.pth'
    runner.save_model(model_save_path)
    
    # 在minta上评估模型
    print("=== 在minta数据集上评估模型 ===")
    minta_results = runner.evaluate()
    
    # 在iochg_small上评估模型
    print("=== 在iochg_small数据集上评估模型 ===")
    try:
        print("正在加载iochg_small数据...")
        hg_small, node_feats_small, num_of_ntype_small, num_classes_small, num_rels_small, target_idx_small, inv_target_small, train_idx_small, val_idx_small, test_idx_small, labels_small, g_small = get_g('iochg_small')
        print("iochg_small数据加载成功")
        
        # 创建新的runner实例，但使用相同的模型架构
        print("正在创建iochg_small的runner实例...")
        runner_small = convert_get_g_output_to_dgl_costa(g_small, node_feats_small, target_idx_small, labels_small, train_idx_small, val_idx_small, test_idx_small)
        print("runner实例创建成功")
        
        # 加载训练好的模型权重
        print("正在加载训练好的模型权重...")
        runner_small.load_model(model_save_path)
        print("模型权重加载成功")
        
        # 在iochg_small上评估
        print("开始在iochg_small上评估...")
        iochg_small_results = runner_small.evaluate()
        
        print(f"minta测试结果: {minta_results:.4f}")
        print(f"iochg_small测试结果: {iochg_small_results:.4f}")
        
    except Exception as e:
        print(f"在iochg_small上评估时出错: {e}")
        import traceback
        traceback.print_exc()
        print("请检查iochg_small的数据路径和文件格式")

    print("DGL COSTA模型训练和评估完成") 