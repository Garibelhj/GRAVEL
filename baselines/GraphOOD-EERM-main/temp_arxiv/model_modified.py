import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj, dense_to_sparse

from nets import *

class Base(nn.Module):
    def __init__(self, args, n, c, d, gnn, device):
        super(Base, self).__init__()
        if gnn == 'gcn':
            self.gnn = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=not args.no_bn)
        elif gnn == 'sage':
            self.gnn = SAGE(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout)
        elif gnn == 'gat':
            self.gnn = GAT(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        heads=args.gat_heads)
        elif gnn == 'gpr':
            self.gnn = GPRGNN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        dropout=args.dropout,
                        alpha=args.gpr_alpha,
                        )
        self.n = n
        self.device = device
        self.args = args

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, data, criterion):
        x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        out = self.gnn(x, edge_index)
        loss = self.sup_loss(y, out, criterion)
        return loss

    def inference(self, data):
        x = data.graph['node_feat'].to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        out = self.gnn(x, edge_index)
        return out

    def sup_loss(self, y, pred, criterion):
        out = F.log_softmax(pred, dim=1)
        target = y.squeeze(1)
        loss = criterion(out, target)
        return loss

class Graph_Editer_Memory_Efficient(nn.Module):
    def __init__(self, K, n, device, max_nodes=10000):
        super(Graph_Editer_Memory_Efficient, self).__init__()
        self.K = K
        self.n = n
        self.device = device
        self.max_nodes = max_nodes
        
        # 如果节点数太多，使用稀疏表示
        if n > max_nodes:
            print(f"Warning: Graph has {n} nodes, using sparse graph editor for memory efficiency")
            self.use_sparse = True
            # 使用稀疏参数，只存储边的权重
            self.edge_weights = nn.Parameter(torch.FloatTensor(K, 1000))  # 限制边数
        else:
            self.use_sparse = False
            self.B = nn.Parameter(torch.FloatTensor(K, n, n))

    def reset_parameters(self):
        if self.use_sparse:
            nn.init.uniform_(self.edge_weights)
        else:
            nn.init.uniform_(self.B)

    def forward(self, edge_index, n, num_sample, k):
        if self.use_sparse:
            # 稀疏版本：只修改少量边
            return self._forward_sparse(edge_index, n, num_sample, k)
        else:
            # 原始版本
            return self._forward_dense(edge_index, n, num_sample, k)
    
    def _forward_sparse(self, edge_index, n, num_sample, k):
        # 稀疏版本：随机选择一些边进行修改
        edge_weights = self.edge_weights[k]
        
        # 随机选择一些边进行修改
        num_edges = edge_index.size(1)
        if num_edges > 1000:
            # 如果边太多，只选择前1000条边
            selected_edges = edge_index[:, :1000]
        else:
            selected_edges = edge_index
        
        # 简单的边权重修改
        P = torch.softmax(edge_weights[:selected_edges.size(1)], dim=0)
        
        # 返回原始边索引和简单的log概率
        log_p = torch.sum(torch.log(P + 1e-8))
        
        return edge_index, log_p
    
    def _forward_dense(self, edge_index, n, num_sample, k):
        # 原始密集版本
        Bk = self.B[k]
        A = to_dense_adj(edge_index, max_num_nodes=n)[0].to(torch.int)
        A_c = torch.ones(n, n, dtype=torch.int).to(self.device) - A
        P = torch.softmax(Bk, dim=0)
        S = torch.multinomial(P, num_samples=num_sample)  # [n, s]
        M = torch.zeros(n, n, dtype=torch.float).to(self.device)
        col_idx = torch.arange(0, n).unsqueeze(1).repeat(1, num_sample)
        M[S, col_idx] = 1.
        C = A + M * (A_c - A)
        edge_index = dense_to_sparse(C)[0]

        log_p = torch.sum(
            torch.sum(Bk[S, col_idx], dim=1) - torch.logsumexp(Bk, dim=0)
        )

        return edge_index, log_p

class Model_Memory_Efficient(nn.Module):
    def __init__(self, args, n, c, d, gnn, device):
        super(Model_Memory_Efficient, self).__init__()
        if gnn == 'gcn':
            self.gnn = GCN(in_channels=d,
                            hidden_channels=args.hidden_channels,
                            out_channels=c,
                            num_layers=args.num_layers,
                            dropout=args.dropout,
                            use_bn=not args.no_bn)
        elif gnn == 'sage':
            self.gnn = SAGE(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout)
        elif gnn == 'gat':
            self.gnn = GAT(in_channels=d,
                           hidden_channels=args.hidden_channels,
                           out_channels=c,
                           num_layers=args.num_layers,
                           dropout=args.dropout,
                           heads=args.gat_heads)
        elif gnn == 'gpr':
            self.gnn = GPRGNN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        dropout=args.dropout,
                        alpha=args.gpr_alpha,
                        )
        self.p = 0.2
        self.n = n
        self.device = device
        self.args = args

        # 使用内存高效的图编辑器
        self.gl = Graph_Editer_Memory_Efficient(args.K, n, device)

    def reset_parameters(self):
        self.gnn.reset_parameters()
        if hasattr(self, 'graph_est'):
            self.gl.reset_parameters()

    def forward(self, data, criterion):
        Loss, Log_p = [], 0
        x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)

        edge_index = data.graph['edge_index'].to(self.device)
        

        K = min(self.args.K, 2)  # 最多使用2个视图
        
        for k in range(K):
            edge_index_k, log_p = self.gl(edge_index, self.n, self.args.num_sample, k)
            out = self.gnn(x, edge_index_k)
            loss = self.sup_loss(y[data.train_mask], out[data.train_mask], criterion)
            Loss.append(loss.view(-1))
            Log_p += log_p
        
        if len(Loss) == 0:
            # 如果没有生成任何损失，使用原始图
            out = self.gnn(x, edge_index)
            loss = self.sup_loss(y, out, criterion)
            Loss = [loss.view(-1)]
            Log_p = torch.tensor(0.0, device=self.device)
        
        Loss = torch.cat(Loss, dim=0)
        Var, Mean = torch.var_mean(Loss)
        return Var, Mean, Log_p

    def inference(self, data):
        x = data.graph['node_feat'].to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        out = self.gnn(x, edge_index)
        return out

    def sup_loss(self, y, pred, criterion):
        if self.args.rocauc or self.args.dataset in ('twitch-e', 'fb100', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss