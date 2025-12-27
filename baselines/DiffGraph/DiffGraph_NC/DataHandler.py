import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from params import args
import scipy.sparse as sp
import dgl
from Utils.TimeLogger import log
import torch as t
from sklearn.preprocessing import OneHotEncoder
import sys
import os
# Add parent directory to path to import get_g from baseline_trainer
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from baseline_trainer import get_g
device = "cuda:0" if t.cuda.is_available() else "cpu"

class DataHandler:
    def __init__(self):

        if args.data == 'DBLP':
            predir = './data/DBLP/'
        elif args.data == 'aminer':
            predir = './data/aminer/'
        elif args.data == 'Freebase':
            predir = './data/Freebase/'
        elif args.data in ['pdns', 'minta', 'iochg']:
            # GRAVEL datasets don't need predir
            predir = None
        else:
            predir = None
        self.predir = predir


    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        # make ui adj
        user,item = mat.shape[0],mat.shape[1]
        a = sp.csr_matrix((user, user))
        b = sp.csr_matrix((item, item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(device)

    def makeTorchuAdj(self, mat):
        """Create tensor-based adjacency matrix for user social graph.

        Args:
            mat: Adjacency matrix.

        Returns:
            Tensor-based adjacency matrix.
        """
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(device)
    def makeBiAdj(self, mat):
        n_user = mat.shape[0]
        n_item = mat.shape[1]
        a = sp.csr_matrix((n_user, n_user))
        b = sp.csr_matrix((n_item, n_item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = mat.tocoo()
        edge_src,edge_dst = mat.nonzero()
        ui_graph = dgl.graph(data=(edge_src, edge_dst),
                            idtype=t.int32,
                             num_nodes=mat.shape[0]
                             )

        return ui_graph


    def LoadData(self):
        if args.data == 'DBLP':
            features_list,apa_mat,ata_mat,ava_mat,train,val,test,labels = self.load_dblp_data()
            self.feature_list = t.FloatTensor(features_list).to(device)

            self.hete_adj1 = dgl.from_scipy(apa_mat).to(device)
            self.hete_adj2 = dgl.from_scipy(ata_mat).to(device)
            self.hete_adj3 = dgl.from_scipy(ava_mat).to(device)
            self.train_idx = train
            self.val_idx = val
            self.test_idx = test
            self.labels = labels
        
        

            # self.train_idx_generator = index_generator(batch_size=args.batch, indices=self.train_idx)
            # self.val_idx_generator = index_generator(batch_size=args.batch, indices=self.val_idx, shuffle=False)
            # self.test_idx_generator = index_generator(batch_size=args.batch, indices=self.test_idx, shuffle=False)
            self.he_adjs = [self.hete_adj1,self.hete_adj2,self.hete_adj3]
        if args.data == 'Freebase':
            features_list,mam_mat,mdm_mat,mwm_mat,train,val,test,labels = self.load_Freebase_data()
            self.feature_list = t.FloatTensor(features_list).to(device)

            self.hete_adj1 = dgl.from_scipy(mam_mat).to(device)
            self.hete_adj2 = dgl.from_scipy(mdm_mat).to(device)
            self.hete_adj3 = dgl.from_scipy(mwm_mat).to(device)
            self.train_idx = train
            self.val_idx = val
            self.test_idx = test
            self.labels = labels

            self.he_adjs = [self.hete_adj1,self.hete_adj2,self.hete_adj3]

        if args.data == 'aminer':
            features_list,pap_mat,prp_mat,pos_mat,train,val,test,labels = self.load_aminer_data()
            self.feature_list = t.FloatTensor(features_list).to(device)

            self.hete_adj1 = dgl.from_scipy(pap_mat).to(device)
            self.hete_adj2 = dgl.from_scipy(prp_mat).to(device)
            self.hete_adj3 = dgl.from_scipy(pos_mat).to(device)
            self.train_idx = train
            self.val_idx = val
            self.test_idx = test
            self.labels = labels

            self.he_adjs = [self.hete_adj1,self.hete_adj2,self.hete_adj3]

        # Support for GRAVEL datasets (pdns, minta, iochg) - Binary Classification
        if args.data in ['pdns', 'minta', 'iochg']:
            features_list, metapath_adjs, train, val, test, labels = self.load_gravel_data(args.data)
            self.feature_list = t.FloatTensor(features_list).to(device)
            
            # Convert adjacency matrices to DGL graphs
            self.he_adjs = [adj.to(device) for adj in metapath_adjs]
            
            self.train_idx = train
            self.val_idx = val
            self.test_idx = test
            self.labels = labels
           
          
        
    def load_dblp_data(self):
        features_a = sp.load_npz(self.predir + 'a_feat.npz').astype("float32")
        # features_1 = sp.load_npz(self.predir + '/features_1.npz').toarray()
        # features_2 = sp.load_npz(self.predir + '/features_2.npy')
        features_a = t.FloatTensor(preprocess_features(features_a))
        
        apa_mat=sp.load_npz(self.predir + "apa.npz")
        ata_mat=sp.load_npz(self.predir + "apcpa.npz")
        ava_mat=sp.load_npz(self.predir + "aptpa.npz")
        labels = np.load(self.predir + 'labels.npy')
        labels = encode_onehot(labels)
        labels= t.FloatTensor(labels).to(device)
        train = [np.load(self.predir + "train_" + str(i) + ".npy") for i in args.ratio]
        test = [np.load(self.predir + "test_" + str(i) + ".npy") for i in args.ratio]
        val = [np.load(self.predir + "val_" + str(i) + ".npy") for i in args.ratio]
        train = [t.LongTensor(i) for i in train]
        val = [t.LongTensor(i) for i in val]
        test = [t.LongTensor(i) for i in test]
        
        return features_a,apa_mat,ata_mat,ava_mat,train,val,test,labels
    
    def load_Freebase_data(self):
        type_num = [3492, 2502, 33401, 4459]
       
        # features_1 = sp.load_npz(self.predir + '/features_1.npz').toarray()
        # features_2 = sp.load_npz(self.predir + '/features_2.npy')
        features_m = sp.eye(type_num[0])
        features_m=t.FloatTensor(preprocess_features(features_m))
        mam = sp.load_npz(self.predir + "mam.npz")
        mdm = sp.load_npz(self.predir + "mdm.npz")
        mwm = sp.load_npz(self.predir + "mwm.npz")
        labels = np.load(self.predir + 'labels.npy')
        labels = encode_onehot(labels)
        labels= t.FloatTensor(labels).to(device)
        train = [np.load(self.predir + "train_" + str(i) + ".npy") for i in args.ratio]
        test = [np.load(self.predir + "test_" + str(i) + ".npy") for i in args.ratio]
        val = [np.load(self.predir + "val_" + str(i) + ".npy") for i in args.ratio]
        train = [t.LongTensor(i) for i in train]
        val = [t.LongTensor(i) for i in val]
        test = [t.LongTensor(i) for i in test]
        
        return features_m,mam,mdm,mwm,train,val,test,labels
    
    def load_aminer_data(self):
        type_num = [6564, 13329, 35890]
       
        # features_1 = sp.load_npz(self.predir + '/features_1.npz').toarray()
        # features_2 = sp.load_npz(self.predir + '/features_2.npy')
        features_p = sp.eye(type_num[0])
        features_p=t.FloatTensor(preprocess_features(features_p))
        pap = sp.load_npz(self.predir + "pap.npz")
        prp = sp.load_npz(self.predir + "prp.npz")
        pos = sp.load_npz(self.predir + "pos.npz")
        labels = np.load(self.predir + 'labels.npy')
        labels = encode_onehot(labels)
        labels= t.FloatTensor(labels).to(device)
        train = [np.load(self.predir + "train_" + str(i) + ".npy") for i in args.ratio]
        test = [np.load(self.predir + "test_" + str(i) + ".npy") for i in args.ratio]
        val = [np.load(self.predir + "val_" + str(i) + ".npy") for i in args.ratio]
        train = [t.LongTensor(i) for i in train]
        val = [t.LongTensor(i) for i in val]
        test = [t.LongTensor(i) for i in test]
        
        return features_p,pap,prp,pos,train,val,test,labels

    def load_gravel_data(self, benchmark):
        """Load GRAVEL datasets (pdns, minta, iochg) for BINARY classification.
        
        Args:
            benchmark: Dataset name ('pdns', 'minta', or 'iochg')
            
        Returns:
            Tuple of (features, metapath_adjs, train_idx, val_idx, test_idx, labels)
            
        Note:
            Labels are converted to binary: 0=benign, 1=malicious
            Nodes with label 2 (n_a/unlabeled) are excluded from train/val/test sets
        """
        # Load data using get_g function
        hg, node_feats, num_of_ntype, num_classes, num_rels, target_idx, inv_target, train_idx, val_idx, test_idx, labels, g = get_g(benchmark, zero_shot=False)
        
        # Extract domain node features (target nodes)
        domain_features = g.nodes['domain'].data['feat'].cpu().numpy()
        domain_features = preprocess_features(sp.csr_matrix(domain_features))
        
        # Convert labels to BINARY classification (0=benign, 1=malicious)
        # Original labels: 0=benign, 1=malicious, 2=n_a (unlabeled)
        labels_np = labels.cpu().numpy()
        
        # Create binary labels using one-hot encoding for 2 classes
        binary_labels = np.zeros((len(labels_np), 2))  # Shape: (num_nodes, 2)
        for i, label in enumerate(labels_np):
            if label == 0:  # benign
                binary_labels[i, 0] = 1
            elif label == 1:  # malicious
                binary_labels[i, 1] = 1
            # label == 2 (n_a) will remain [0, 0], we'll filter these out
        
        labels_tensor = t.FloatTensor(binary_labels).to(device)
        
        # Filter out unlabeled nodes (label==2) from train/val/test indices
        # Keep only nodes with binary labels (benign or malicious)
        labeled_mask = (labels_np == 0) | (labels_np == 1)
        
        def filter_indices(idx_tensor):
            """Filter indices to keep only labeled nodes"""
            idx_np = idx_tensor.cpu().numpy()
            # Keep only indices that correspond to labeled nodes
            filtered_idx = idx_np[labeled_mask[idx_np]]
            return t.LongTensor(filtered_idx)
        
        train_idx_filtered = filter_indices(train_idx)
        val_idx_filtered = filter_indices(val_idx)
        test_idx_filtered = filter_indices(test_idx)
        
        print(f"\n[{benchmark}] Binary classification data:")
        print(f"  - Original train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
        print(f"  - Filtered train/val/test: {len(train_idx_filtered)}/{len(val_idx_filtered)}/{len(test_idx_filtered)}")
        print(f"  - Benign: {np.sum(labels_np == 0)}, Malicious: {np.sum(labels_np == 1)}, Unlabeled: {np.sum(labels_np == 2)}")
        
        # Extract metapath-based adjacency matrices for domain nodes
        metapath_adjs = self.extract_metapath_adjacencies(g, benchmark)
        
        # Prepare train/val/test indices for multiple ratios
        # Replicate for consistency with DiffGraph's multiple ratio training
        train_list = [train_idx_filtered for _ in args.ratio]
        val_list = [val_idx_filtered for _ in args.ratio]
        test_list = [test_idx_filtered for _ in args.ratio]
        
        return domain_features, metapath_adjs, train_list, val_list, test_list, labels_tensor
    
    def extract_metapath_adjacencies(self, g, benchmark):
        """Extract metapath-based adjacency matrices from heterogeneous graph.
        
        For each dataset, we extract domain-domain adjacency matrices via different metapaths:
        - pdns: domain-ip-domain, domain-similar-domain, domain-apex-domain
        - minta: domain-ip-domain, domain-similar-domain, domain-apex-domain
        - iochg: domain-ip-domain, domain-domain (direct), domain-url-domain
        
        Args:
            g: DGL heterogeneous graph
            benchmark: Dataset name
            
        Returns:
            List of DGL graphs representing metapath-based adjacencies
        """
        metapath_adjs = []
        num_domains = g.num_nodes('domain')
        
        if benchmark == 'pdns':
            # Metapath 1: domain-ip-domain (via edge type '1' and '_1')
            metapath_adj1 = self.compute_metapath_adj(g, [('domain', '1', 'ip'), ('ip', '_1', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj1)
            
            # Metapath 2: domain-similar-domain (edge type '2')
            metapath_adj2 = self.compute_metapath_adj(g, [('domain', '2', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj2)
            
            # Metapath 3: domain-apex-domain (edge type '3')
            metapath_adj3 = self.compute_metapath_adj(g, [('domain', '3', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj3)
            
        elif benchmark == 'minta':
            # Metapath 1: domain-ip-domain
            metapath_adj1 = self.compute_metapath_adj(g, [('domain', '1', 'ip'), ('ip', '_1', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj1)
            
            # Metapath 2: domain-similar-domain
            metapath_adj2 = self.compute_metapath_adj(g, [('domain', '2', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj2)
            
            # Metapath 3: domain-apex-domain
            metapath_adj3 = self.compute_metapath_adj(g, [('domain', '3', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj3)
            
        elif benchmark == 'iochg':
            # Metapath 1: domain-ip-domain
            metapath_adj1 = self.compute_metapath_adj(g, [('domain', '8', 'ip'), ('ip', '_8', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj1)
            
            # Metapath 2: domain-domain (edge type '7')
            metapath_adj2 = self.compute_metapath_adj(g, [('domain', '7', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj2)
            
            # Metapath 3: domain-domain (edge type '9')
            metapath_adj3 = self.compute_metapath_adj(g, [('domain', '9', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj3)
        
        return metapath_adjs
    
    def compute_metapath_adj(self, g, metapath, num_nodes):
        """Compute adjacency matrix for a given metapath.
        
        Args:
            g: DGL heterogeneous graph
            metapath: List of edge types forming the metapath
            num_nodes: Number of target nodes (domains)
            
        Returns:
            DGL graph representing the metapath adjacency
        """
        if len(metapath) == 1:
            # Direct edge between same node type
            etype = metapath[0]
            if etype in g.canonical_etypes:
                src, dst = g.edges(etype=etype)
                # Create scipy sparse matrix
                adj = sp.coo_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                   shape=(num_nodes, num_nodes))
            else:
                # If edge type doesn't exist, create empty adjacency
                adj = sp.coo_matrix((num_nodes, num_nodes))
        else:
            # Multi-hop metapath
            adj = None
            for i, etype in enumerate(metapath):
                if etype not in g.canonical_etypes:
                    # Edge type doesn't exist, return empty
                    return dgl.from_scipy(sp.coo_matrix((num_nodes, num_nodes)))
                
                src, dst = g.edges(etype=etype)
                src_type = etype[0]
                dst_type = etype[2]
                
                if i == 0:
                    # First hop: domain to intermediate
                    num_intermediate = g.num_nodes(dst_type)
                    curr_adj = sp.coo_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                            shape=(num_nodes, num_intermediate))
                    adj = curr_adj
                else:
                    # Subsequent hops
                    num_dst = g.num_nodes(dst_type)
                    curr_adj = sp.coo_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                            shape=(adj.shape[1], num_dst))
                    adj = adj.dot(curr_adj)
            
            # Convert to binary adjacency
            adj = (adj > 0).astype(np.float32)
        
        # Add self-loops
        adj = adj + sp.eye(num_nodes)
        adj = (adj > 0).astype(np.float32)
        
        # Convert to DGL graph
        return dgl.from_scipy(adj.tocoo())



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()
def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0



        for i, label in enumerate(labels_np):
            if label == 0:  # benign
                binary_labels[i, 0] = 1
            elif label == 1:  # malicious
                binary_labels[i, 1] = 1
            # label == 2 (n_a) will remain [0, 0], we'll filter these out
        
        labels_tensor = t.FloatTensor(binary_labels).to(device)
        
        # Filter out unlabeled nodes (label==2) from train/val/test indices
        # Keep only nodes with binary labels (benign or malicious)
        labeled_mask = (labels_np == 0) | (labels_np == 1)
        
        def filter_indices(idx_tensor):
            """Filter indices to keep only labeled nodes"""
            idx_np = idx_tensor.cpu().numpy()
            # Keep only indices that correspond to labeled nodes
            filtered_idx = idx_np[labeled_mask[idx_np]]
            return t.LongTensor(filtered_idx)
        
        train_idx_filtered = filter_indices(train_idx)
        val_idx_filtered = filter_indices(val_idx)
        test_idx_filtered = filter_indices(test_idx)
        
        print(f"\n[{benchmark}] Binary classification data:")
        print(f"  - Original train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
        print(f"  - Filtered train/val/test: {len(train_idx_filtered)}/{len(val_idx_filtered)}/{len(test_idx_filtered)}")
        print(f"  - Benign: {np.sum(labels_np == 0)}, Malicious: {np.sum(labels_np == 1)}, Unlabeled: {np.sum(labels_np == 2)}")
        
        # Extract metapath-based adjacency matrices for domain nodes
        metapath_adjs = self.extract_metapath_adjacencies(g, benchmark)
        
        # Prepare train/val/test indices for multiple ratios
        # Replicate for consistency with DiffGraph's multiple ratio training
        train_list = [train_idx_filtered for _ in args.ratio]
        val_list = [val_idx_filtered for _ in args.ratio]
        test_list = [test_idx_filtered for _ in args.ratio]
        
        return domain_features, metapath_adjs, train_list, val_list, test_list, labels_tensor
    
    def extract_metapath_adjacencies(self, g, benchmark):
        """Extract metapath-based adjacency matrices from heterogeneous graph.
        
        For each dataset, we extract domain-domain adjacency matrices via different metapaths:
        - pdns: domain-ip-domain, domain-similar-domain, domain-apex-domain
        - minta: domain-ip-domain, domain-similar-domain, domain-apex-domain
        - iochg: domain-ip-domain, domain-domain (direct), domain-url-domain
        
        Args:
            g: DGL heterogeneous graph
            benchmark: Dataset name
            
        Returns:
            List of DGL graphs representing metapath-based adjacencies
        """
        metapath_adjs = []
        num_domains = g.num_nodes('domain')
        
        if benchmark == 'pdns':
            # Metapath 1: domain-ip-domain (via edge type '1' and '_1')
            metapath_adj1 = self.compute_metapath_adj(g, [('domain', '1', 'ip'), ('ip', '_1', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj1)
            
            # Metapath 2: domain-similar-domain (edge type '2')
            metapath_adj2 = self.compute_metapath_adj(g, [('domain', '2', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj2)
            
            # Metapath 3: domain-apex-domain (edge type '3')
            metapath_adj3 = self.compute_metapath_adj(g, [('domain', '3', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj3)
            
        elif benchmark == 'minta':
            # Metapath 1: domain-ip-domain
            metapath_adj1 = self.compute_metapath_adj(g, [('domain', '1', 'ip'), ('ip', '_1', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj1)
            
            # Metapath 2: domain-similar-domain
            metapath_adj2 = self.compute_metapath_adj(g, [('domain', '2', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj2)
            
            # Metapath 3: domain-apex-domain
            metapath_adj3 = self.compute_metapath_adj(g, [('domain', '3', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj3)
            
        elif benchmark == 'iochg':
            # Metapath 1: domain-ip-domain
            metapath_adj1 = self.compute_metapath_adj(g, [('domain', '8', 'ip'), ('ip', '_8', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj1)
            
            # Metapath 2: domain-domain (edge type '7')
            metapath_adj2 = self.compute_metapath_adj(g, [('domain', '7', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj2)
            
            # Metapath 3: domain-domain (edge type '9')
            metapath_adj3 = self.compute_metapath_adj(g, [('domain', '9', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj3)
        
        return metapath_adjs
    
    def compute_metapath_adj(self, g, metapath, num_nodes):
        """Compute adjacency matrix for a given metapath.
        
        Args:
            g: DGL heterogeneous graph
            metapath: List of edge types forming the metapath
            num_nodes: Number of target nodes (domains)
            
        Returns:
            DGL graph representing the metapath adjacency
        """
        if len(metapath) == 1:
            # Direct edge between same node type
            etype = metapath[0]
            if etype in g.canonical_etypes:
                src, dst = g.edges(etype=etype)
                # Create scipy sparse matrix
                adj = sp.coo_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                   shape=(num_nodes, num_nodes))
            else:
                # If edge type doesn't exist, create empty adjacency
                adj = sp.coo_matrix((num_nodes, num_nodes))
        else:
            # Multi-hop metapath
            adj = None
            for i, etype in enumerate(metapath):
                if etype not in g.canonical_etypes:
                    # Edge type doesn't exist, return empty
                    return dgl.from_scipy(sp.coo_matrix((num_nodes, num_nodes)))
                
                src, dst = g.edges(etype=etype)
                src_type = etype[0]
                dst_type = etype[2]
                
                if i == 0:
                    # First hop: domain to intermediate
                    num_intermediate = g.num_nodes(dst_type)
                    curr_adj = sp.coo_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                            shape=(num_nodes, num_intermediate))
                    adj = curr_adj
                else:
                    # Subsequent hops
                    num_dst = g.num_nodes(dst_type)
                    curr_adj = sp.coo_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                            shape=(adj.shape[1], num_dst))
                    adj = adj.dot(curr_adj)
            
            # Convert to binary adjacency
            adj = (adj > 0).astype(np.float32)
        
        # Add self-loops
        adj = adj + sp.eye(num_nodes)
        adj = (adj > 0).astype(np.float32)
        
        # Convert to DGL graph
        return dgl.from_scipy(adj.tocoo())



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()
def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0



        for i, label in enumerate(labels_np):
            if label == 0:  # benign
                binary_labels[i, 0] = 1
            elif label == 1:  # malicious
                binary_labels[i, 1] = 1
            # label == 2 (n_a) will remain [0, 0], we'll filter these out
        
        labels_tensor = t.FloatTensor(binary_labels).to(device)
        
        # Filter out unlabeled nodes (label==2) from train/val/test indices
        # Keep only nodes with binary labels (benign or malicious)
        labeled_mask = (labels_np == 0) | (labels_np == 1)
        
        def filter_indices(idx_tensor):
            """Filter indices to keep only labeled nodes"""
            idx_np = idx_tensor.cpu().numpy()
            # Keep only indices that correspond to labeled nodes
            filtered_idx = idx_np[labeled_mask[idx_np]]
            return t.LongTensor(filtered_idx)
        
        train_idx_filtered = filter_indices(train_idx)
        val_idx_filtered = filter_indices(val_idx)
        test_idx_filtered = filter_indices(test_idx)
        
        print(f"\n[{benchmark}] Binary classification data:")
        print(f"  - Original train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
        print(f"  - Filtered train/val/test: {len(train_idx_filtered)}/{len(val_idx_filtered)}/{len(test_idx_filtered)}")
        print(f"  - Benign: {np.sum(labels_np == 0)}, Malicious: {np.sum(labels_np == 1)}, Unlabeled: {np.sum(labels_np == 2)}")
        
        # Extract metapath-based adjacency matrices for domain nodes
        metapath_adjs = self.extract_metapath_adjacencies(g, benchmark)
        
        # Prepare train/val/test indices for multiple ratios
        # Replicate for consistency with DiffGraph's multiple ratio training
        train_list = [train_idx_filtered for _ in args.ratio]
        val_list = [val_idx_filtered for _ in args.ratio]
        test_list = [test_idx_filtered for _ in args.ratio]
        
        return domain_features, metapath_adjs, train_list, val_list, test_list, labels_tensor
    
    def extract_metapath_adjacencies(self, g, benchmark):
        """Extract metapath-based adjacency matrices from heterogeneous graph.
        
        For each dataset, we extract domain-domain adjacency matrices via different metapaths:
        - pdns: domain-ip-domain, domain-similar-domain, domain-apex-domain
        - minta: domain-ip-domain, domain-similar-domain, domain-apex-domain
        - iochg: domain-ip-domain, domain-domain (direct), domain-url-domain
        
        Args:
            g: DGL heterogeneous graph
            benchmark: Dataset name
            
        Returns:
            List of DGL graphs representing metapath-based adjacencies
        """
        metapath_adjs = []
        num_domains = g.num_nodes('domain')
        
        if benchmark == 'pdns':
            # Metapath 1: domain-ip-domain (via edge type '1' and '_1')
            metapath_adj1 = self.compute_metapath_adj(g, [('domain', '1', 'ip'), ('ip', '_1', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj1)
            
            # Metapath 2: domain-similar-domain (edge type '2')
            metapath_adj2 = self.compute_metapath_adj(g, [('domain', '2', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj2)
            
            # Metapath 3: domain-apex-domain (edge type '3')
            metapath_adj3 = self.compute_metapath_adj(g, [('domain', '3', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj3)
            
        elif benchmark == 'minta':
            # Metapath 1: domain-ip-domain
            metapath_adj1 = self.compute_metapath_adj(g, [('domain', '1', 'ip'), ('ip', '_1', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj1)
            
            # Metapath 2: domain-similar-domain
            metapath_adj2 = self.compute_metapath_adj(g, [('domain', '2', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj2)
            
            # Metapath 3: domain-apex-domain
            metapath_adj3 = self.compute_metapath_adj(g, [('domain', '3', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj3)
            
        elif benchmark == 'iochg':
            # Metapath 1: domain-ip-domain
            metapath_adj1 = self.compute_metapath_adj(g, [('domain', '8', 'ip'), ('ip', '_8', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj1)
            
            # Metapath 2: domain-domain (edge type '7')
            metapath_adj2 = self.compute_metapath_adj(g, [('domain', '7', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj2)
            
            # Metapath 3: domain-domain (edge type '9')
            metapath_adj3 = self.compute_metapath_adj(g, [('domain', '9', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj3)
        
        return metapath_adjs
    
    def compute_metapath_adj(self, g, metapath, num_nodes):
        """Compute adjacency matrix for a given metapath.
        
        Args:
            g: DGL heterogeneous graph
            metapath: List of edge types forming the metapath
            num_nodes: Number of target nodes (domains)
            
        Returns:
            DGL graph representing the metapath adjacency
        """
        if len(metapath) == 1:
            # Direct edge between same node type
            etype = metapath[0]
            if etype in g.canonical_etypes:
                src, dst = g.edges(etype=etype)
                # Create scipy sparse matrix
                adj = sp.coo_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                   shape=(num_nodes, num_nodes))
            else:
                # If edge type doesn't exist, create empty adjacency
                adj = sp.coo_matrix((num_nodes, num_nodes))
        else:
            # Multi-hop metapath
            adj = None
            for i, etype in enumerate(metapath):
                if etype not in g.canonical_etypes:
                    # Edge type doesn't exist, return empty
                    return dgl.from_scipy(sp.coo_matrix((num_nodes, num_nodes)))
                
                src, dst = g.edges(etype=etype)
                src_type = etype[0]
                dst_type = etype[2]
                
                if i == 0:
                    # First hop: domain to intermediate
                    num_intermediate = g.num_nodes(dst_type)
                    curr_adj = sp.coo_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                            shape=(num_nodes, num_intermediate))
                    adj = curr_adj
                else:
                    # Subsequent hops
                    num_dst = g.num_nodes(dst_type)
                    curr_adj = sp.coo_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                            shape=(adj.shape[1], num_dst))
                    adj = adj.dot(curr_adj)
            
            # Convert to binary adjacency
            adj = (adj > 0).astype(np.float32)
        
        # Add self-loops
        adj = adj + sp.eye(num_nodes)
        adj = (adj > 0).astype(np.float32)
        
        # Convert to DGL graph
        return dgl.from_scipy(adj.tocoo())



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()
def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0



        for i, label in enumerate(labels_np):
            if label == 0:  # benign
                binary_labels[i, 0] = 1
            elif label == 1:  # malicious
                binary_labels[i, 1] = 1
            # label == 2 (n_a) will remain [0, 0], we'll filter these out
        
        labels_tensor = t.FloatTensor(binary_labels).to(device)
        
        # Filter out unlabeled nodes (label==2) from train/val/test indices
        # Keep only nodes with binary labels (benign or malicious)
        labeled_mask = (labels_np == 0) | (labels_np == 1)
        
        def filter_indices(idx_tensor):
            """Filter indices to keep only labeled nodes"""
            idx_np = idx_tensor.cpu().numpy()
            # Keep only indices that correspond to labeled nodes
            filtered_idx = idx_np[labeled_mask[idx_np]]
            return t.LongTensor(filtered_idx)
        
        train_idx_filtered = filter_indices(train_idx)
        val_idx_filtered = filter_indices(val_idx)
        test_idx_filtered = filter_indices(test_idx)
        
        print(f"\n[{benchmark}] Binary classification data:")
        print(f"  - Original train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
        print(f"  - Filtered train/val/test: {len(train_idx_filtered)}/{len(val_idx_filtered)}/{len(test_idx_filtered)}")
        print(f"  - Benign: {np.sum(labels_np == 0)}, Malicious: {np.sum(labels_np == 1)}, Unlabeled: {np.sum(labels_np == 2)}")
        
        # Extract metapath-based adjacency matrices for domain nodes
        metapath_adjs = self.extract_metapath_adjacencies(g, benchmark)
        
        # Prepare train/val/test indices for multiple ratios
        # Replicate for consistency with DiffGraph's multiple ratio training
        train_list = [train_idx_filtered for _ in args.ratio]
        val_list = [val_idx_filtered for _ in args.ratio]
        test_list = [test_idx_filtered for _ in args.ratio]
        
        return domain_features, metapath_adjs, train_list, val_list, test_list, labels_tensor
    
    def extract_metapath_adjacencies(self, g, benchmark):
        """Extract metapath-based adjacency matrices from heterogeneous graph.
        
        For each dataset, we extract domain-domain adjacency matrices via different metapaths:
        - pdns: domain-ip-domain, domain-similar-domain, domain-apex-domain
        - minta: domain-ip-domain, domain-similar-domain, domain-apex-domain
        - iochg: domain-ip-domain, domain-domain (direct), domain-url-domain
        
        Args:
            g: DGL heterogeneous graph
            benchmark: Dataset name
            
        Returns:
            List of DGL graphs representing metapath-based adjacencies
        """
        metapath_adjs = []
        num_domains = g.num_nodes('domain')
        
        if benchmark == 'pdns':
            # Metapath 1: domain-ip-domain (via edge type '1' and '_1')
            metapath_adj1 = self.compute_metapath_adj(g, [('domain', '1', 'ip'), ('ip', '_1', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj1)
            
            # Metapath 2: domain-similar-domain (edge type '2')
            metapath_adj2 = self.compute_metapath_adj(g, [('domain', '2', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj2)
            
            # Metapath 3: domain-apex-domain (edge type '3')
            metapath_adj3 = self.compute_metapath_adj(g, [('domain', '3', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj3)
            
        elif benchmark == 'minta':
            # Metapath 1: domain-ip-domain
            metapath_adj1 = self.compute_metapath_adj(g, [('domain', '1', 'ip'), ('ip', '_1', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj1)
            
            # Metapath 2: domain-similar-domain
            metapath_adj2 = self.compute_metapath_adj(g, [('domain', '2', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj2)
            
            # Metapath 3: domain-apex-domain
            metapath_adj3 = self.compute_metapath_adj(g, [('domain', '3', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj3)
            
        elif benchmark == 'iochg':
            # Metapath 1: domain-ip-domain
            metapath_adj1 = self.compute_metapath_adj(g, [('domain', '8', 'ip'), ('ip', '_8', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj1)
            
            # Metapath 2: domain-domain (edge type '7')
            metapath_adj2 = self.compute_metapath_adj(g, [('domain', '7', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj2)
            
            # Metapath 3: domain-domain (edge type '9')
            metapath_adj3 = self.compute_metapath_adj(g, [('domain', '9', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj3)
        
        return metapath_adjs
    
    def compute_metapath_adj(self, g, metapath, num_nodes):
        """Compute adjacency matrix for a given metapath.
        
        Args:
            g: DGL heterogeneous graph
            metapath: List of edge types forming the metapath
            num_nodes: Number of target nodes (domains)
            
        Returns:
            DGL graph representing the metapath adjacency
        """
        if len(metapath) == 1:
            # Direct edge between same node type
            etype = metapath[0]
            if etype in g.canonical_etypes:
                src, dst = g.edges(etype=etype)
                # Create scipy sparse matrix
                adj = sp.coo_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                   shape=(num_nodes, num_nodes))
            else:
                # If edge type doesn't exist, create empty adjacency
                adj = sp.coo_matrix((num_nodes, num_nodes))
        else:
            # Multi-hop metapath
            adj = None
            for i, etype in enumerate(metapath):
                if etype not in g.canonical_etypes:
                    # Edge type doesn't exist, return empty
                    return dgl.from_scipy(sp.coo_matrix((num_nodes, num_nodes)))
                
                src, dst = g.edges(etype=etype)
                src_type = etype[0]
                dst_type = etype[2]
                
                if i == 0:
                    # First hop: domain to intermediate
                    num_intermediate = g.num_nodes(dst_type)
                    curr_adj = sp.coo_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                            shape=(num_nodes, num_intermediate))
                    adj = curr_adj
                else:
                    # Subsequent hops
                    num_dst = g.num_nodes(dst_type)
                    curr_adj = sp.coo_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                            shape=(adj.shape[1], num_dst))
                    adj = adj.dot(curr_adj)
            
            # Convert to binary adjacency
            adj = (adj > 0).astype(np.float32)
        
        # Add self-loops
        adj = adj + sp.eye(num_nodes)
        adj = (adj > 0).astype(np.float32)
        
        # Convert to DGL graph
        return dgl.from_scipy(adj.tocoo())



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()
def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0



        for i, label in enumerate(labels_np):
            if label == 0:  # benign
                binary_labels[i, 0] = 1
            elif label == 1:  # malicious
                binary_labels[i, 1] = 1
            # label == 2 (n_a) will remain [0, 0], we'll filter these out
        
        labels_tensor = t.FloatTensor(binary_labels).to(device)
        
        # Filter out unlabeled nodes (label==2) from train/val/test indices
        # Keep only nodes with binary labels (benign or malicious)
        labeled_mask = (labels_np == 0) | (labels_np == 1)
        
        def filter_indices(idx_tensor):
            """Filter indices to keep only labeled nodes"""
            idx_np = idx_tensor.cpu().numpy()
            # Keep only indices that correspond to labeled nodes
            filtered_idx = idx_np[labeled_mask[idx_np]]
            return t.LongTensor(filtered_idx)
        
        train_idx_filtered = filter_indices(train_idx)
        val_idx_filtered = filter_indices(val_idx)
        test_idx_filtered = filter_indices(test_idx)
        
        print(f"\n[{benchmark}] Binary classification data:")
        print(f"  - Original train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
        print(f"  - Filtered train/val/test: {len(train_idx_filtered)}/{len(val_idx_filtered)}/{len(test_idx_filtered)}")
        print(f"  - Benign: {np.sum(labels_np == 0)}, Malicious: {np.sum(labels_np == 1)}, Unlabeled: {np.sum(labels_np == 2)}")
        
        # Extract metapath-based adjacency matrices for domain nodes
        metapath_adjs = self.extract_metapath_adjacencies(g, benchmark)
        
        # Prepare train/val/test indices for multiple ratios
        # Replicate for consistency with DiffGraph's multiple ratio training
        train_list = [train_idx_filtered for _ in args.ratio]
        val_list = [val_idx_filtered for _ in args.ratio]
        test_list = [test_idx_filtered for _ in args.ratio]
        
        return domain_features, metapath_adjs, train_list, val_list, test_list, labels_tensor
    
    def extract_metapath_adjacencies(self, g, benchmark):
        """Extract metapath-based adjacency matrices from heterogeneous graph.
        
        For each dataset, we extract domain-domain adjacency matrices via different metapaths:
        - pdns: domain-ip-domain, domain-similar-domain, domain-apex-domain
        - minta: domain-ip-domain, domain-similar-domain, domain-apex-domain
        - iochg: domain-ip-domain, domain-domain (direct), domain-url-domain
        
        Args:
            g: DGL heterogeneous graph
            benchmark: Dataset name
            
        Returns:
            List of DGL graphs representing metapath-based adjacencies
        """
        metapath_adjs = []
        num_domains = g.num_nodes('domain')
        
        if benchmark == 'pdns':
            # Metapath 1: domain-ip-domain (via edge type '1' and '_1')
            metapath_adj1 = self.compute_metapath_adj(g, [('domain', '1', 'ip'), ('ip', '_1', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj1)
            
            # Metapath 2: domain-similar-domain (edge type '2')
            metapath_adj2 = self.compute_metapath_adj(g, [('domain', '2', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj2)
            
            # Metapath 3: domain-apex-domain (edge type '3')
            metapath_adj3 = self.compute_metapath_adj(g, [('domain', '3', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj3)
            
        elif benchmark == 'minta':
            # Metapath 1: domain-ip-domain
            metapath_adj1 = self.compute_metapath_adj(g, [('domain', '1', 'ip'), ('ip', '_1', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj1)
            
            # Metapath 2: domain-similar-domain
            metapath_adj2 = self.compute_metapath_adj(g, [('domain', '2', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj2)
            
            # Metapath 3: domain-apex-domain
            metapath_adj3 = self.compute_metapath_adj(g, [('domain', '3', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj3)
            
        elif benchmark == 'iochg':
            # Metapath 1: domain-ip-domain
            metapath_adj1 = self.compute_metapath_adj(g, [('domain', '8', 'ip'), ('ip', '_8', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj1)
            
            # Metapath 2: domain-domain (edge type '7')
            metapath_adj2 = self.compute_metapath_adj(g, [('domain', '7', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj2)
            
            # Metapath 3: domain-domain (edge type '9')
            metapath_adj3 = self.compute_metapath_adj(g, [('domain', '9', 'domain')], num_domains)
            metapath_adjs.append(metapath_adj3)
        
        return metapath_adjs
    
    def compute_metapath_adj(self, g, metapath, num_nodes):
        """Compute adjacency matrix for a given metapath.
        
        Args:
            g: DGL heterogeneous graph
            metapath: List of edge types forming the metapath
            num_nodes: Number of target nodes (domains)
            
        Returns:
            DGL graph representing the metapath adjacency
        """
        if len(metapath) == 1:
            # Direct edge between same node type
            etype = metapath[0]
            if etype in g.canonical_etypes:
                src, dst = g.edges(etype=etype)
                # Create scipy sparse matrix
                adj = sp.coo_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                   shape=(num_nodes, num_nodes))
            else:
                # If edge type doesn't exist, create empty adjacency
                adj = sp.coo_matrix((num_nodes, num_nodes))
        else:
            # Multi-hop metapath
            adj = None
            for i, etype in enumerate(metapath):
                if etype not in g.canonical_etypes:
                    # Edge type doesn't exist, return empty
                    return dgl.from_scipy(sp.coo_matrix((num_nodes, num_nodes)))
                
                src, dst = g.edges(etype=etype)
                src_type = etype[0]
                dst_type = etype[2]
                
                if i == 0:
                    # First hop: domain to intermediate
                    num_intermediate = g.num_nodes(dst_type)
                    curr_adj = sp.coo_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                            shape=(num_nodes, num_intermediate))
                    adj = curr_adj
                else:
                    # Subsequent hops
                    num_dst = g.num_nodes(dst_type)
                    curr_adj = sp.coo_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                            shape=(adj.shape[1], num_dst))
                    adj = adj.dot(curr_adj)
            
            # Convert to binary adjacency
            adj = (adj > 0).astype(np.float32)
        
        # Add self-loops
        adj = adj + sp.eye(num_nodes)
        adj = (adj > 0).astype(np.float32)
        
        # Convert to DGL graph
        return dgl.from_scipy(adj.tocoo())



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()
def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0


