## convert Minta
import json
import pandas as pd
from tqdm import tqdm
all_domain = []
all_ip = []
all_host = []
nodes_domain = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/MintA-main/myGraph_datasets/DNS/nodes.domain.csv')
domain_info = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/MintA-main/myGraph_datasets/DNS/extras.csv')
nodes_ip = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/MintA-main/myGraph_datasets/DNS/nodes.ip.csv')
nodes_host = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/MintA-main/myGraph_datasets/DNS/nodes.host.csv')
edges = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/MintA-main/myGraph_datasets/DNS/edges.csv')
edge_resolve = edges[edges.type=='resolves']
edge_similar = edges[edges.type=='similar']
edge_apex = edges[edges.type=='apex']
edge_query = edges[edges.type=='HAS_QUERY']
for i in edge_resolve.source:
    all_domain.append(i)
for i in edge_resolve.target:
    all_ip.append(i)
for i in edge_similar.source:
    all_domain.append(i)
for i in edge_similar.target:
    all_domain.append(i)
for i in edge_apex.source:
    all_domain.append(i)
for i in edge_apex.target:
    all_domain.append(i)
for i in edge_query.source:
    all_domain.append(i)
for i in edge_query.target:
    all_host.append(i)

all_domain = set(all_domain)
if 'd107051' in all_domain:
    print('ok')
all_ip = set(all_ip)
all_host = set(all_host)
# 将列表转换为 DataFrame，并作为一列
df_ip = pd.DataFrame(all_ip, columns=['node_id'])
df_ip = df_ip.reset_index(drop=True)
# 增加一个从 0 开始的整数索引列
df_ip['index'] = df_ip.index
df_host = pd.DataFrame(all_host, columns=['node_id'])
df_host = df_host.reset_index(drop=True)
# 增加一个从 0 开始的整数索引列
df_host['index'] = df_host.index



df_domain = pd.DataFrame(all_domain, columns=['node_id'])
df_domain = df_domain.reset_index(drop=True)

# 增加一个从 0 开始的整数索引列
df_domain['index'] = df_domain.index
merged_df = pd.merge(df_domain, domain_info, on='node_id', how='left')
merged_df = merged_df.drop(columns=['Unnamed: 0'])
merged_df['type'] = merged_df['type'].replace({
    'n_a': 2,
    'malicious': 1,
    'benign': 0
})
## 生成边对应的csv文件

merge_edge_resolve =  pd.merge(edge_resolve, merged_df, left_on='source',right_on='node_id', how='left')
merge_edge_resolve =  pd.merge(merge_edge_resolve, df_ip, left_on='target',right_on='node_id', how='left')

merge_edge_similar = pd.merge(edge_similar, merged_df, left_on='source',right_on='node_id', how='left')
merge_edge_similar = pd.merge(merge_edge_similar, merged_df, left_on='target',right_on='node_id', how='left')

merge_edge_apex = pd.merge(edge_apex, merged_df, left_on='source',right_on='node_id', how='left')
merge_edge_apex = pd.merge(merge_edge_apex, merged_df, left_on='target',right_on='node_id', how='left')

merge_edge_query = pd.merge(edge_query, merged_df, left_on='source',right_on='node_id', how='left')
merge_edge_query = pd.merge(merge_edge_query, df_host, left_on='target',right_on='node_id', how='left')

df_host.to_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/host.csv')
df_ip.to_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/ip.csv')
merged_df.to_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/domain.csv')
merge_edge_resolve.to_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/edge_1.csv')
merge_edge_similar.to_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/edge_2.csv')
merge_edge_apex.to_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/edge_3.csv')
merge_edge_query.to_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/edge_4.csv')

import dgl
import pandas as pd
import torch as th 
import numpy as np
th.set_num_threads(1)

def cal_ratio(tensor):
    count_0 = th.sum(tensor == 0).item()
    count_1 = th.sum(tensor == 1).item()
    count_2 = th.sum(tensor == 2).item()

    # 计算 tensor 中元素的总数
    total = tensor.numel()

    # 计算比例
    ratio_0 = count_0 / total
    ratio_1 = count_1 / total
    ratio_2 = count_2 / total
    return ratio_0,ratio_1,ratio_2
def cal_degrees(hg,label,dirct):
    labels = hg.ndata['label']
    label_2_nodes = th.nonzero(labels == label, as_tuple=False).squeeze()
    degrees = []

    for i in label_2_nodes:
        if dirct == 'in':
            degrees.append(hg.in_degrees(i).item())
        else:
            degrees.append(hg.out_degrees(i).item())
    # print(in_degrees)
    degrees_max = max(degrees)
    degrees_min = min(degrees)
    degrees_mean = np.mean(degrees)

    return degrees_max,degrees_min,degrees_mean
def neighbor_information(hg,label):
    labels = hg.ndata['label']
    label_2_nodes = th.nonzero(labels == label, as_tuple=False).squeeze()

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)

    dataloader = dgl.dataloading.DataLoader(
        hg, label_2_nodes, sampler,
        batch_size=64,
        shuffle=True,
        drop_last=False,
        num_workers=1)
    l1=[]
    l2=[]
    l3=[]
    for i, sample_data in enumerate(dataloader):
        input_nodes, seeds, blocks = sample_data
        r1,r2,r3 = cal_ratio(blocks[0].srcdata['label'])
        l1.append(r1)
        l2.append(r2)
        l3.append(r3)
    print(np.mean(l1),np.mean(l2),np.mean(l3))


data_path = '/home/hongjiegu/projects/GRAVEL/dataset/minta'
domain = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta/domain.csv')
ip = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta/ip.csv')
host = pd.read_csv('/home/hongjiegu/projects/GRAVEL/dataset/minta/host.csv')
edge = {}
u = {}
v = {}
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
g.nodes['domain'].data['label'] = th.tensor(domain['type'])  # .to(device)
g.nodes['host'].data['label'] = th.full((len(host),), 3, dtype=th.long)
g.nodes['ip'].data['label'] =  th.full((len(ip),), 3, dtype=th.long)
print(g)
hg = dgl.to_homogeneous(g, ndata=['label'])

hg.ndata['ntype'] = hg.ndata[dgl.NTYPE]
hg.edata['etype'] = hg.edata[dgl.ETYPE]
hg.ndata['type_id'] = hg.ndata[dgl.NID]
# 找到 label == 2 的节点
# in_degrees_max,in_degrees_min,in_degrees_mean = cal_degrees(hg,0,'in')
neighbor_information(hg,2)