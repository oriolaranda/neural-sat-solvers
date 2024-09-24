import torch
import copy
import torch_sparse
import torch.nn as nn
import numpy as np

from ..utils import sparse_blockdiag, sparse_elem_mul


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, out_dim)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.r(self.l1(x))
        x = self.l2(x)
        return x


class CircuitSAT(nn.Module):
    def __init__(self, dim=100, dim_agg=50, dim_class=30, n_rounds=20):
        super().__init__()
        self.name = "CSAT"
        self.dim = dim
        self.n_rounds = n_rounds
        self.init = nn.Linear(4, dim)
        self.forward_msg = MLP(dim, dim_agg, dim)
        self.backward_msg = MLP(dim, dim_agg, dim)
        self.forward_update = nn.GRU(dim, dim)
        self.backward_update = nn.GRU(dim, dim)

        self.classif = MLP(dim, dim_class, 1) # Classification network C_a
        
        # F_o(u_G) = C_a(P(E_b(u_G)))
        # u_G -> DAG function, in this case, we encode the type as one-hot d-dimensional encoding {And, Or, Not, Variable}
        # C_a -> Classification network from q dimensions to [0, 1] range (soft classification)
        # E_b -> Embedding function that maps from d-dimensional function to q-dimensional function.
        # P -> Non-parametric Pooling function that aggregates the q-dim functions depends on the application

        # x_v = u_G(v) node feature vector
        # h_v = d_G(v) node state vector
        # h_v = GRU(x_v, h'_v)
        # h'_v = A({h_u | u \in \pi(v)}) 
        # A -> aggregator function (deep set function)

        # S(u_G) = R(F_o(u_G))
        # F_o -> Policy network
        # R -> Evaluator network (kind of reward)
        # L = (1-S(u_G))**k/(((1-S(u_G))**k) + S(u_G)**k)
        # Minimizing L loss, we push S(u_G) -> 1 (i.e., sat result, and thus a valid assignment) 

    def forward(self, sample):
        # Allocate tensors contiguous in GPU for efficiency
        self.forward_update.flatten_parameters()
        self.backward_update.flatten_parameters()
        adj = sample['adj']
        h_state = self.init(sample['features'].cuda()).unsqueeze(0)

        for _ in range(self.n_rounds):
            f_pre_msg = self.forward_msg(h_state.squeeze(0)) # Pass through the GRU
            f_msg = torch_sparse.matmul(adj, f_pre_msg)

            _, h_state = self.forward_update(f_msg.unsqueeze(0), h_state)

            b_pre_msg = self.backward_msg(h_state.squeeze(0))
            b_msg = torch_sparse.matmul(adj.t(), b_pre_msg)

            _, h_state = self.backward_update(b_msg.unsqueeze(0), h_state)
            
        return self.classif(h_state.squeeze(0))


    @staticmethod
    def cnf_to_csat(clauses, nv):
        adj = torch.zeros(nv * 2, nv * 2)  # Adjacency matrix

        # Indices for the literals (atoms and their negations). The negation of atom i is i+nv.
        idx = torch.arange(nv * 2).view(2, -1).T
        for (i, j) in idx:  # Add direct edge between postive literal and its negation
            adj[i, j] = 1

        # Convert negation literals to indices
        t_cnf = [[abs(l) + nv if l < 0 else l for l in c] for c in clauses]

        # Make indicator for each node (literal) of the graph
        indicator = torch.zeros(1, 2 * nv)
        
        # For each clause add a new node
        for i, c in enumerate(t_cnf):
            # Add row and column to adjacency matrix
            adj = nn.functional.pad(adj, (0, 1, 0, 1))

            # Add direct edge for each literal index in the clause to the new node (clause)
            for l in c:
                adj[l - 1, -1] = 1

            # Add a new position with 1 for the new added node
            indicator = nn.functional.pad(indicator, (0, 1), value=1)

            # Save the index of the clause node (is the last element), for connecting with the root node later
            t_cnf[i] = adj.size(0)

        # Add a final node (root)
        adj = nn.functional.pad(adj, (0, 1, 0, 1))

        # Add a new postion for the root with value -1
        indicator = nn.functional.pad(indicator, (0, 1), value=-1)

        # For each clause index add a direct edge to the last (root) node
        for l in t_cnf:
            adj[l - 1, -1] = 1
        
        # Generate a CSAT instance
        csat_instance = {
            'clauses': clauses,
            'n_clauses': len(clauses),
            'n_vars': nv,
            'adj': torch_sparse.SparseTensor.from_dense(adj),
            'ind': indicator
        }
        return csat_instance
    

    @staticmethod
    def collate_fn(batch):
        # Prepare a batch of samples
        is_sat, solution, n_vars, clauses = [], [], [], []
        single_adjs, inds, fnames, fs = [], [], [], []
        n_clauses = []
        var_labels = []
        for sample in batch:
            cnf = sample['cnf']
            p = CircuitSAT.cnf_to_csat(cnf.clauses, cnf.nv)
            single_adjs.append(p['adj'])
            n_vars.append(p['n_vars'])
            clauses.append(p['clauses'])
            n_clauses.append(len(p['clauses']))
            is_sat.append(sample['is_sat'])
            inds.append(p['ind'])
            f = copy.deepcopy(p['ind'][0])
            f[f == 1] = 2
            f[f == -1] = 3
            f[p['n_vars']:p['n_vars']*2] = 1
            fs.append(f)
            fnames.append(sample['file'])
            solution.append(sample['solution'])
            c = np.array(sample['solution'])
            c = c[c > 0] - 1
            l = torch.zeros(p['n_vars'])
            l[c] = 1
            var_labels.append(l)

        adj = sparse_blockdiag(single_adjs)
        indicators = torch.cat(fs).squeeze(0)
        assert indicators.size(0) == adj.size(0) == adj.size(1)
        feats = torch.zeros((indicators.size(0), 4))
        feats = torch.scatter(feats, 1, indicators.long().unsqueeze(1), torch.ones(feats.shape))
        assert torch.all(feats.sum(-1) == 1)

        new_batch = {
            'batch_size': len(batch),
            'n_vars': torch.Tensor(n_vars).int(),
            'is_sat': torch.Tensor(is_sat).float(),
            'adj': adj,
            'ind': inds,
            'clauses': clauses,
            'solution': solution,
            'fnames': fnames,
            'features': feats,
            'varlabel': var_labels
            # 'n_clauses': torch.Tensor(n_clauses).int()
        }
        return new_batch
    

    @staticmethod
    def evaluate_circuit(sample, emb, epoch, eps=1.2, hard=False):
        # explore exploit with annealing rate
        t = epoch ** (-eps)
        inds = torch.cat(sample['ind'], dim=1).view(-1)

        # set to negative to make sure we don't accidentally use nn preds for or/and
        temporary = emb.clone()
        temporary[inds == 1] = -1
        temporary[inds == -1] = -1

        # NOT gate
        temporary[sample['features'][:, 1] == 1] = 1 - emb[sample['features'][:, 0] == 1].clone()
        emb = temporary.clone()
        
        # OR gate
        idx = torch.arange(inds.size(0))[inds==1]
        or_gates = torch_sparse.index_select(sample['adj'], 1, idx.to(emb.device))
        e_gated = torch_sparse.mul(or_gates, emb)
        row, col, vals = e_gated.coo()
        assert torch.all(vals >= 0)

        if hard:
            e_gated = e_gated.max(dim=0)
        else:
            eps = e_gated + (-e_gated.max(dim=0).unsqueeze(0))
            _, _, vals = eps.coo()
            e_temp = torch.exp(vals / t).clone()
            # no elementwise multiplication in torch_sparse
            e_temp =torch_sparse.SparseTensor(row=row, rowptr=None, col=col, value=e_temp, sparse_sizes=e_gated.sizes())
            e_gated = sparse_elem_mul(e_gated, e_temp).sum(dim=0) / e_temp.sum(dim=0)
            assert torch.all(torch.eq(e_gated, e_gated))

        # AND gate
        idx2 = torch.arange(inds.size(0))[inds==-1]
        and_gates = torch_sparse.index_select(sample['adj'], 0, idx.to(emb.device))
        and_gates = torch_sparse.index_select(and_gates, 1, idx2.to(emb.device))
        e_gated = torch_sparse.mul(and_gates, e_gated.unsqueeze(1))
        row, col, vals = e_gated.coo()
        assert torch.all(vals >= 0)
        
        if hard:
            e_gated = e_gated.min(dim=0)
        else:
            eps = e_gated + (-e_gated.min(dim=0).unsqueeze(0))
            _, _, vals = eps.coo()
            e_temp = torch.exp(-vals / t).clone()
            e_temp = torch_sparse.SparseTensor(row=row, rowptr=None, col=col, value=e_temp, sparse_sizes=e_gated.sizes())
            e_gated = sparse_elem_mul(e_gated, e_temp).sum(dim=0) / e_temp.sum(dim=0)
            assert torch.all(torch.eq(e_gated, e_gated))

        assert torch.all(e_gated >= 0)
        assert len(e_gated) == len(sample['n_vars']), f"{len(e_gated)}, {len(sample['n_vars'])}"
        return e_gated
    

    @staticmethod
    def get_representation(batch):
        all_inds = torch.cat(batch['ind'], dim=1).squeeze()
        literals = all_inds == 0
        ors = all_inds == 1
        # We are only interested in the region of the adjacency matrix between literals and clauses
        repr = batch['adj'][literals, ors].to_dense()
        return repr.cuda()

    @staticmethod
    def get_sat_mask(batch):
        all_inds = torch.cat(batch['ind'], dim=1).squeeze()
        literals = all_inds == 0
        ors = all_inds == 1
        # We are only interested in the region of the adjacency matrix between literals and clauses
        sat_mask = batch['sat_mask'][literals, ors].to_dense()
        return sat_mask.cuda()


    @staticmethod
    def reconstruct(repr, batch):
        batch_ = copy.deepcopy(batch)
        adj = batch_['adj'].to_dense().to(repr.device)
        all_inds = torch.cat(batch['ind'], dim=1).squeeze()
        literals = all_inds == 0
        ors = all_inds == 1
        temp = adj[literals]
        temp[:, ors] = repr
        adj[literals] = temp
        batch_['adj'] = torch_sparse.SparseTensor.from_dense(adj)
        return batch_


    




