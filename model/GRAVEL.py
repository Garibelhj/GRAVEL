import os
# GPU设置应该在主脚本中设置，不应该在模块导入时设置
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sys
sys.path.append('../model/')
import torch as th
import torch.nn as nn
from parameterdict import ParameterDict
import dgl
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import TypedLinear  
import json
from dgl import DropEdge
import numpy as np
device = th.device("cuda" if th.cuda.is_available() else "cpu")



class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h

def initializer(emb):
    emb.uniform_(-1.0, 1.0)
    return emb

class RelGraphEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    storage_dev_id : int
        The device to store the weights of the layer.
    out_dev_id : int
        Device to return the output embeddings on.
    num_nodes : int
        Number of nodes.
    node_tides : tensor
        Storing the node type id for each node starting from 0
    num_of_ntype : int
        Number of node types
    input_size : list of int
        A list of input feature size for each node type. If None, we then
        treat certain input feature as an one-hot encoding feature.
    embed_size : int
        Output embed size
    dgl_sparse : bool, optional
        If true, use dgl.nn.NodeEmbedding otherwise use torch.nn.Embedding
    """
    def __init__(self,
                 storage_dev_id,
 
                 num_of_ntype,
                 input_size,
                 embed_size,
                 dgl_sparse=False):
        super(RelGraphEmbedLayer, self).__init__()
        self.storage_dev_id = device
        self.out_dev_id = device
        self.embed_size = embed_size
        self.dgl_sparse = dgl_sparse

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.mlp_embeds = nn.ModuleDict()
        self.node_embeds = nn.ModuleDict()
        self.num_of_ntype = num_of_ntype

        for ntype in range(num_of_ntype):
            if isinstance(input_size[ntype], int):
                if dgl_sparse:
                    self.node_embeds[str(ntype)] = dgl.nn.NodeEmbedding(input_size[ntype], embed_size, name=str(ntype),
                        init_func=initializer, device=self.storage_dev_id)
                else:
                    sparse_emb = th.nn.Embedding(input_size[ntype], embed_size, sparse=True)
                    if not self.storage_dev_id == th.device('cpu'):
                        sparse_emb.cuda(self.storage_dev_id)
                    nn.init.uniform_(sparse_emb.weight, -1.0, 1.0)
                    self.node_embeds[str(ntype)] = sparse_emb
            else:
                input_emb_size = input_size[ntype].shape[1]
                embed = nn.Parameter(th.empty([input_emb_size, self.embed_size],
                                              device=self.storage_dev_id))
                nn.init.xavier_uniform_(embed)
                self.embeds[str(ntype)] = embed
                # ********************
                # input_emb_size torch.Size([2620775, 12])
                # input_emb_size torch.Size([2292467, 30])
                # input_emb_size torch.Size([974434, 37])
                # input_emb_size torch.Size([2118354, 5])
                # ********************
                input_emb_size = input_size[ntype].shape[1]
                # print('input_emb_size', input_size[ntype].shape)
                self.mlp_embeds[str(ntype)] = nn.Sequential(
                    nn.Linear(input_emb_size, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(32, embed_size),
                    nn.BatchNorm1d(embed_size),
                )

    @property
    def dgl_emb(self):
        """
        """
        if self.dgl_sparse:
            embs = [emb for emb in self.node_embeds.values()]
            return embs
        else:
            return []

    def forward(self, node_ids, node_tids, type_ids, features):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        node_ids : tensor
            node type ids
        features : list of features
            list of initial features for nodes belong to different node type.
            If None, the corresponding features is an one-hot encoding feature,
            else use the features directly as input feature and matmul a
            projection matrix.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        embeds = th.empty(node_ids.shape[0], self.embed_size).to(device)
        self.node_embeds = self.node_embeds.to(device)
        # transfer input to the correct device
        type_ids = type_ids.to(device)
        node_tids = node_tids.to(device)

        # build locs first
        locs = [None for i in range(self.num_of_ntype)]
        for ntype in range(self.num_of_ntype):
            locs[ntype] = (node_tids == ntype).nonzero().squeeze(-1)
        for ntype in range(self.num_of_ntype):
            loc = locs[ntype]

            embeds[loc] = features[ntype].to(self.out_dev_id)[type_ids[loc]].to(self.out_dev_id) @ self.embeds[str(ntype)].to(self.out_dev_id)


        return embeds


class RelGraphConv(nn.Module):
    r"""Relational graph convolution layer from `Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__

    It can be described in as below:

    .. math::

       h_i^{(l+1)} = \sigma(\sum_{r\in\mathcal{R}}
       \sum_{j\in\mathcal{N}^r(i)}e_{j,i}W_r^{(l)}h_j^{(l)}+W_0^{(l)}h_i^{(l)})

    where :math:`\mathcal{N}^r(i)` is the neighbor set of node :math:`i` w.r.t. relation
    :math:`r`. :math:`e_{j,i}` is the normalizer. :math:`\sigma` is an activation
    function. :math:`W_0` is the self-loop weight.

    The basis regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)}V_b^{(l)}

    where :math:`B` is the number of bases, :math:`V_b^{(l)}` are linearly combined
    with coefficients :math:`a_{rb}^{(l)}`.

    The block-diagonal-decomposition regularization decomposes :math:`W_r` into :math:`B`
    number of block diagonal matrices. We refer :math:`B` as the number of bases.

    The block regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \oplus_{b=1}^B Q_{rb}^{(l)}

    where :math:`B` is the number of bases, :math:`Q_{rb}^{(l)}` are block
    bases with shape :math:`R^{(d^{(l+1)}/B)*(d^{l}/B)}`.

    Parameters
    ----------
    in_feat : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feat : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    num_rels : int
        Number of relations. .
    regularizer : str, optional
        Which weight regularizer to use "basis" or "bdd":

         - "basis" is short for basis-decomposition.
         - "bdd" is short for block-diagonal-decomposition.

        Default applies no regularization.
    num_bases : int, optional
        Number of bases. Needed when ``regularizer`` is specified. Default: ``None``.
    bias : bool, optional
        True if bias is added. Default: ``True``.
    activation : callable, optional
        Activation function. Default: ``None``.
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: float, optional
        Add layer norm. Default: ``False``

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import RelGraphConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = RelGraphConv(10, 2, 3, regularizer='basis', num_bases=2)
    >>> etype = th.tensor([0,1,2,0,1,2])
    >>> res = conv(g, feat, etype)
    >>> res
    tensor([[ 0.3996, -2.3303],
            [-0.4323, -0.1440],
            [ 0.3996, -2.3303],
            [ 2.1046, -2.8654],
            [-0.4323, -0.1440],
            [-0.1309, -1.0000]], grad_fn=<AddBackward0>)
    """
    def __init__(self,
                 in_feat,#输入维度h^l
                 out_feat,#输出维度h^(l+1)
                 num_rels,#边类型数量
                 regularizer=None,
                 num_bases=None,#w_r分解的数量，基数分解法
                 bias=True,#偏置
                 activation=None,#激活函数
                 self_loop=True,#是否将自身特征加入
                 dropout=0.0,
                 attention=True,
                 multi_atten_heads = 3,
                 layer_norm=True):#添加层正则化
        super().__init__()
        #根据类型进行线性变换。
        self.linear_r = TypedLinear(in_feat, in_feat, num_rels, regularizer, num_bases)
        self.linear_rels_r = TypedLinear(in_feat, out_feat, num_rels, regularizer, num_bases)

        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm
        self.attention = attention
        self.num_rels = num_rels
        self.linear = nn.Linear(in_feat,out_feat)
        self.label_bias = TypedLinear(2, out_feat, num_rels, regularizer, num_bases)
        #self.label_bias = nn.Linear(2, out_feat)
        self.sigmoid = nn.Sigmoid()
        self.gate_net = nn.Sequential(
            nn.Linear(2*in_feat, 4*in_feat),
            nn.ReLU(),
            nn.Dropout(dropout),  # 新增
            nn.Linear(4*in_feat, in_feat),
            nn.Sigmoid()
        )
        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # TODO(minjie): consider remove those options in the future to make
        #   the module only about graph convolution.
        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

        if self.attention:
            # self.fc = nn.Linear(in_feat, out_feat * num_heads, bias=False)
            # # create weight embeddings for each node for each relation
            self.atten_embeds = ParameterDict()
            self.atten_embeds_rels = ParameterDict()
            self.a = ParameterDict()
            self.a_rel = ParameterDict()
            self.h_bias_rels = ParameterDict()
            self.multi_atten_heads = multi_atten_heads
            # self.num_of_etype = num_rels
            # self.feat_drop = nn.Dropout(feat_drop)
            # self.attn_drop = nn.Dropout(attn_drop)

            for i in range(self.multi_atten_heads):
                # embed = nn.Parameter(th.empty([in_feat, out_feat]))
                # nn.init.xavier_uniform_(embed)
                embed = TypedLinear(in_feat, in_feat, num_rels, regularizer, num_bases)
                self.atten_embeds[str(i)] = embed

                a = nn.Parameter(th.Tensor(2*in_feat,1))
                nn.init.xavier_uniform_(a, gain=nn.init.calculate_gain('relu'))
                self.a[str(i)] = a

                a_rel = nn.Parameter(th.Tensor(2*out_feat,1))
                nn.init.xavier_uniform_(a_rel, gain=nn.init.calculate_gain('relu'))
                self.a_rel[str(i)] = a_rel


                embed_rels = TypedLinear(in_feat, out_feat, num_rels, regularizer, num_bases)
                self.atten_embeds_rels[str(i)] = embed_rels

                bias = nn.Parameter(th.Tensor(out_feat))
                nn.init.zeros_(bias)
                self.h_bias_rels[str(i)] = bias

            self.loop_atten = nn.Parameter(th.Tensor(in_feat, in_feat))
            nn.init.xavier_uniform_(self.loop_atten, gain=nn.init.calculate_gain('relu'))
            # self.atten_linear = TypedLinear(in_feat, out_feat, num_rels, regularizer, num_bases)
            negative_slope = 0.2
            self.leaky_relu = nn.LeakyReLU(negative_slope)
            self.tanh = nn.Tanh()



    def message_pass_atten(self, edges):
        """Message function."""
        # edges.src['h']-源节点特征
        # edges.data['etype']-边特征
        # label = edges.src['label']
        # label_bais = self.label_bias(label)
        m = self.linear_r(edges.src['h'], edges.data['etype'], self.presorted)

        if 'a' in edges.data:
            m = m * edges.data['a']
        # if 'norm' in edges.data:
        #     m_de = m * edges.data['norm']
        label_bias = self.label_bias(edges.src['label'],edges.data['etype'],self.presorted)
        # label_bias = self.label_bias(edges.src['label'])
        label_bias = self.sigmoid(label_bias)

        m_de = label_bias*m
        return {'m_atten_pass' : m_de,'pseudo_label':edges.src['label']}

    def message_atten(self, edges):
        """Message function."""
        # edges.src['h']-源节点特征
        # edges.data['etype']-边特征
        # m = self.linear_r(edges.src['h'], edges.data['etype'], self.presorted)
        # if self.attention:
        atten_l = self.atten_embeds[str(self.atten_k)](edges.src['h'], edges.data['etype'], self.presorted) @ self.loop_atten
        atten_r = self.atten_embeds[str(self.atten_k)](edges.dst['h'], edges.data['etype'], self.presorted) @ self.loop_atten
        # atten_l = th.mm(edges.src['h'], self.atten_embeds[str(self.atten_k)]) @ self.loop_atten
        # atten_r = th.mm(edges.dst['h'], self.atten_embeds[str(self.atten_k)]) @ self.loop_atten
        e = th.mm(th.cat((atten_l,atten_r),1), self.a[str(self.atten_k)])
        e_active = self.leaky_relu(e)
        atten =  e_active
        edges.data['e'] = e_active
        tensor = th.zeros((20,len(edges.data['etype'])),dtype=th.int32).to(device)
        for i in range(20):
            tensor[i] = i
        result = th.eq(tensor,edges.data['etype']).int()
        e_re = edges.data['e'].reshape(1,-1)
        result = th.mul(result,e_re)
        for i in result:
            alpha = F.softmax(i,dim=0)
            edges.data['e'] = alpha.unsqueeze(1)
        rel_in_feats_l=th.mul(edges.data['e'],edges.src['h'])
        rel_in_feats_r=th.mul(edges.data['e'],edges.dst['h'])
        linear_rel_in_feats_l = self.linear(rel_in_feats_l)
        linear_rel_in_feats_r = self.linear(rel_in_feats_r)
        e= th.mm(th.cat((linear_rel_in_feats_l,linear_rel_in_feats_r),1),self.a[str(self.atten_k)])
        e_active = self.leaky_relu(e)
        atten=e_active
        edges.data['e'] = e_active
        return {'m_atten' : atten}
# reduce函数
    def node_udf(self,nodes):
        # nodes.mailbox['m']: (batch_size, num_neighbors, feat_dim)
        # nodes.mailbox['label']: (batch_size, num_neighbors)
        m = nodes.mailbox['m_atten_pass']
        pseudo_label = nodes.mailbox['pseudo_label']
        
        # 获取恶意和正常邻居的mask
        black_mask = (pseudo_label[:, :, 1] >= 0.5).unsqueeze(-1)  # 取恶意概率
        white_mask = (pseudo_label[:, :, 0] >= 0.5).unsqueeze(-1)  # 取正常概率       
        
        # 使用mask分别获取黑邻居和白邻居的信息
        black_msgs = m * black_mask  # Shape: (150, 4, 64)
        white_msgs = m * white_mask  # Shape: (150, 4, 64)
        
        # 分别求和，得到两个64维度的向量
        black_sum = black_msgs.sum(1)  # Shape: (150, 64) - 黑邻居信息聚合
        white_sum = white_msgs.sum(1)  # Shape: (150, 64) - 白邻居信息聚合
        gate = self.gate_net(th.cat([black_sum, white_sum], dim=-1))
        h = gate * black_sum + (1 - gate) * white_sum
        
        # 先实现简单的求和效果，类似 fn.sum('m_atten_pass', 'h')
        
        return {'h': h}

    # def message_pass_rels(self, edges):
    #     """Message function."""
    #     # edges.src['h']-源节点特征
    #     # edges.data['etype']-边特征
    #     m = self.linear_rels_r(edges.dst['h_multi_atten'], edges.data['etype'], self.presorted)
    #     if 'a_rels' in edges.data:
    #         m = m * edges.data['a_rels']
    #     return {'m_pass' : m}

    # def massage_atten_rels(self, edges):
    #     m = self.atten_embeds_rels[str(self.atten_k)](edges.dst['h_multi_atten'], edges.data['etype'], self.presorted)
    #     # (m.size(),self.h_bias_rels[str(self.atten_k)].size())
    #     if 'norm' in edges.data:
    #         m_de = m * edges.data['norm']
    #     if 'norm_w' in edges.data:
    #         m_w = m * edges.data['norm_w'].to(th.float32)
    #     m = th.mm(th.cat((m_de,m_w),1), self.a_rel[str(self.atten_k)])
    #     m = m + self.h_bias_rels[str(self.atten_k)]
    #     e_active = self.tanh(m)
    #     # print(len(e_active))
    #     # print(len(edges))
    #     edges.data['e_rels'] = e_active
    #     return {'m_atten_rels' : m}
    # def get_edge_src_dst(self, edges):
    #     """Message function."""
    #     # edges.src['h']-源节点特征
    #     # edges.data['etype']-边特征
    #     m = edges.src['h']
    #     # GetG.setG(edges.data['a_rels'], edges.data['etype'], edges.src['ntype'], edges.src['type_id'],edges.dst['ntype'],edges.dst['type_id'])
    #     # print('get_edges')
    #     return {'_m': m}

    def forward(self, g, feat, etypes, norm=None, norm_w = None, *, presorted=False):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        etypes : torch.Tensor or list[int]
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        norm : torch.Tensor, optional
            An 1D tensor of edge norm value.  Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether the edges of the input graph have been sorted by their types.
            Forward on pre-sorted graph may be faster. Graphs created
            by :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for sorting edges manually.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{out})`.
        """

        _, v, eid = g.all_edges(form='all')
        # print(g.all_edges(form='all'))
  
        self.presorted = presorted
        with g.local_scope():
            g.srcdata['h'] = feat
            if norm is not None:
                g.edata['norm'] = norm
                # message passing
                # g.update_all(self.message, fn.sum('m', 'h'))
            if norm is not None:
                g.edata['norm_w'] = norm_w
                # message passing
                # g.update_all(self.message, fn.sum('m', 'h'))

            if self.attention:
                g.dstdata['h'] = feat[:g.num_dst_nodes()]
                # g.update_all(self.message, fn.sum('m', 'h'))
                for i in range(self.multi_atten_heads):
                    self.atten_k = i
                    g.update_all(self.message_atten, fn.sum('m_atten', 'h_atten'))
                    e = g.edata.pop('e')
                    g.edata['a'] = edge_softmax(g, e, norm_by='dst')
                    g.update_all(self.message_pass_atten, self.node_udf)
                    # g.update_all(self.message, fn.sum('m', 'h'))
                    if i == 0:
                        h_atten_sum = g.dstdata['h']
                    else:
                        h_atten_sum = h_atten_sum + g.dstdata['h']
                # g.dstdata['h_multi_atten'] = h_atten_sum / self.multi_atten_heads
                # for i in range(self.multi_atten_heads):
                #     self.atten_k = i
                #     g.update_all(self.massage_atten_rels, fn.sum('m_atten_rels', 'h_atten_rels'))
                #     e_rels = g.edata.pop('e_rels')
                #     g.edata['a_rels'] = edge_softmax(g, e_rels, norm_by='dst')
                #     g.update_all(self.message_pass_rels, fn.sum('m_pass', 'h'))
                #     if i == 0:
                #         h_atten_rels_sum = g.dstdata['h']
                #     else:
                #         h_atten_rels_sum = h_atten_rels_sum + g.dstdata['h']
                # h_atten_rels = h_atten_rels_sum / self.multi_atten_heads
                h_atten_rels = h_atten_sum / self.multi_atten_heads
            # g.edata['etype'] = etypes
            # apply bias and activation
            
            # h = g.dstdata['h_multi_atten']
            h = h_atten_rels
            # print(h_atten.size())
            # print(self.h_bias.size())
            if self.layer_norm:
                h = self.layer_norm_weight(h)
            if self.bias:
                h = h + self.h_bias
                
            if self.self_loop:
                h = h + feat[:g.num_dst_nodes()] @ self.loop_weight
            if self.activation:
                h = self.activation(h)
            # if self.attention:
            #     h = (h + h_atten_rels) / 2
            h = self.dropout(h)
            # g.update_all(self.get_edge_src_dst, fn.sum('_m', '_'))
            return h

class EntityClassify(nn.Module):
    """ Entity classification class for RGCN
    Parameters
    ----------
    device : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    h_dim : int
        Hidden dim size.
    out_dim : int
        Output dim size.
    num_rels : int
        Numer of relation types.
    num_bases : int, optional
        Number of bases. If is none, use number of relations.
        Default None
    num_hidden_layers : int, optional
        Number of hidden RelGraphConv Layer
        Default 1
    dropout : float, optional
        Dropout.
        Default 0
    use_self_loop : bool, optional
        Use self loop if True.
        Default True
    low_mem : bool, optional
        True to use low memory implementation of relation message passing function
        trade speed with memory consumption
        Default True
    layer_norm : bool, optional
        True to use layer norm.
        Default False
    """
    def __init__(self,
                device,
                num_nodes,
                h_dim,
                out_dim,
                num_rels,
                num_bases=2,
                num_hidden_layers=2,
                dropout=0,
                use_self_loop=True,
                low_mem=True,
                layer_norm=True):
        super(EntityClassify, self).__init__()
        self.device = device
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = 2 #None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.low_mem = low_mem
        self.layer_norm = layer_norm
        self.transform = DropEdge()
        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConv(
            self.h_dim, self.h_dim, self.num_rels, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,#low_mem=self.low_mem, 
            attention=True, dropout=self.dropout, layer_norm = layer_norm))
        # h2h
        for idx in range(self.num_hidden_layers):
            self.layers.append(RelGraphConv(
                self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,#low_mem=self.low_mem, 
                attention=True, dropout=self.dropout, layer_norm = layer_norm))
        # h2o
        self.linear = nn.Linear(self.h_dim, self.out_dim)

    def forward(self, blocks,feats, norm=None):

        h = feats
        for layer, block in zip(self.layers, blocks):
            block = block.to(device)
            h = layer(block, h, block.edata['etype'], block.edata['norm'], block.edata['norm_w'])
        hidden_status = self.linear(h)
        return hidden_status,h