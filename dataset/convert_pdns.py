import pandas as pd
import dgl
edges = []
for i in range(61):
    df = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/PDNS-Net-main/data/DNS_2m/timestamp_{}_edges.csv" .format(i))
    edges.append(df)
edges = pd.concat(edges, ignore_index=True)
edge_resolves = edges[edges.type=='resolves']
edge_resolves.to_csv('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/edge_1.csv')
edge_similar = edges[edges.type=='similar']
edge_similar.to_csv('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/edge_2.csv')
edge_apex = edges[edges.type=='apex']
edge_apex.to_csv('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/edge_3.csv')

domain = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/PDNS-Net-main/data/DNS_2m/domains2.csv")
ips = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/PDNS-Net-main/data/DNS_2m/ips.csv")
print(len(domain))
print(len(ips))

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
        r1,r2,r3 = cal_ratio(blocks[0].srcdata['ntype'])
        l1.append(r1)
        l2.append(r2)
        l3.append(r3)
    print(np.mean(l1),np.mean(l2),np.mean(l3))

domain = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/PDNS-Net-main/data/DNS_2m/domains2.csv")
ip = pd.read_csv("/home/hongjiegu/projects/GRAVEL/dataset/PDNS-Net-main/data/DNS_2m/ips.csv")
domain['type'] = domain['type'].replace({
    'n_a': 2,
    'malicious': 1,
    'benign': 0
})
print(domain.type.value_counts())
domain.to_csv('/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/domains2.csv')
data_path = '/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl'
edge = {}
u = {}
v = {}
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
g.nodes['domain'].data['label'] = th.tensor(domain['type'])  # .to(device)
g.nodes['ip'].data['label'] =  th.full((len(ip),), 3, dtype=th.long)
hg = dgl.to_homogeneous(g, ndata=['label'])

hg.ndata['ntype'] = hg.ndata[dgl.NTYPE]
hg.edata['etype'] = hg.edata[dgl.ETYPE]
hg.ndata['type_id'] = hg.ndata[dgl.NID]
in_degrees_max,in_degrees_min,in_degrees_mean = cal_degrees(hg,1,'in')
# neighbor_information(hg,0)
print(in_degrees_max,in_degrees_min,in_degrees_mean)