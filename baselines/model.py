import torch as th
import torch.nn as nn

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import AvgPooling
import torch.nn.functional as F

class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()

        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)

    def forward(self, h1, h2, h3, h4, c1, c2):
        c_x1 = c1.expand_as(h1).contiguous()
        c_x2 = c2.expand_as(h2).contiguous()

        # positive
        sc_1 = self.fn(h2, c_x1).squeeze(1)
        sc_2 = self.fn(h1, c_x2).squeeze(1)

        # negative
        sc_3 = self.fn(h4, c_x1).squeeze(1)
        sc_4 = self.fn(h3, c_x2).squeeze(1)

        logits = th.cat((sc_1, sc_2, sc_3, sc_4))

        return logits

class MVGRL(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MVGRL, self).__init__()

        self.encoder1 = GraphConv(in_dim, out_dim, norm='both', bias=True, activation=nn.PReLU())
        self.encoder2 = GraphConv(in_dim, out_dim, norm='none', bias=True, activation=nn.PReLU())
        self.pooling = AvgPooling()

        self.disc = Discriminator(out_dim)
        self.act_fn = nn.Sigmoid()

    def get_embedding(self, graph, diff_graph, feat, edge_weight):
        h1 = self.encoder1(graph, feat)
        h2 = self.encoder2(diff_graph, feat, edge_weight=edge_weight)

        return (h1 + h2).detach()

    def forward(self, graph, diff_graph, feat, shuf_feat, edge_weight):
        h1 = self.encoder1(graph, feat)
        h2 = self.encoder2(diff_graph, feat, edge_weight=edge_weight)

        h3 = self.encoder1(graph, shuf_feat)
        h4 = self.encoder2(diff_graph, shuf_feat, edge_weight=edge_weight)

        c1 = self.act_fn(self.pooling(graph, h1))
        c2 = self.act_fn(self.pooling(graph, h2))

        out = self.disc(h1, h2, h3, h4, c1, c2)

        return out
    
# Multi-layer Graph Convolutional Networks
class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn, num_layers=2):
        super(GCN, self).__init__()

        assert num_layers >= 2
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, out_dim * 2))
        for _ in range(self.num_layers - 2):
            self.convs.append(GraphConv(out_dim * 2, out_dim * 2))

        self.convs.append(GraphConv(out_dim * 2, out_dim))
        self.act_fn = act_fn

    def forward(self, graph, feat):
        for i in range(self.num_layers):
            feat = self.act_fn(self.convs[i](graph, feat))

        return feat


# Multi-layer(2-layer) Perceptron
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, in_dim)

    def forward(self, x):
        z = F.elu(self.fc1(x))
        return self.fc2(z)


class Grace(nn.Module):
    r"""
        GRACE model
    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.
    out_dim: int
        Output feature size.
    num_layers: int
        Number of the GNN encoder layers.
    act_fn: nn.Module
        Activation function.
    temp: float
        Temperature constant.
    """

    def __init__(self, in_dim, hid_dim, out_dim, num_layers, act_fn, temp):
        super(Grace, self).__init__()
        self.encoder = GCN(in_dim, hid_dim, act_fn, num_layers)
        self.temp = temp
        self.proj = MLP(hid_dim, out_dim)

    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = th.mm(z1, z2.t())
        return s

    def get_loss(self, z1, z2):
        # calculate SimCLR loss
        f = lambda x: th.exp(x / self.temp)

        refl_sim = f(self.sim(z1, z1))  # intra-view pairs
        between_sim = f(self.sim(z1, z2))  # inter-view pairs

        # between_sim.diag(): positive pairs
        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        loss = -th.log(between_sim.diag() / x1)

        return loss

    def get_embedding(self, graph, feat):
        # get embeddings from the model for evaluation
        h = self.encoder(graph, feat)

        return h.detach()

    def forward(self, graph1, graph2, feat1, feat2):
        # encoding
        h1 = self.encoder(graph1, feat1)
        h2 = self.encoder(graph2, feat2)

        # projection
        z1 = self.proj(h1)
        z2 = self.proj(h2)

        # get loss
        l1 = self.get_loss(z1, z2)
        l2 = self.get_loss(z2, z1)

        ret = (l1 + l2) * 0.5

        return ret.mean()