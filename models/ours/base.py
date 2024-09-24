import copy
import torch
import numpy as np
import torch_sparse
import torch.nn as nn
from abc import abstractmethod

from ..utils import sparse_elem_mul, sparse_blockdiag


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


def pad(adjs, inds, n_vars, solutions):
    max_size = np.max(n_vars)
    # It is sufficient to add the padding nodes at the end of the adjacency and indicator vector
    adjs = [torch.nn.functional.pad(adj, (0, (max_size-n_var)*2, 0, (max_size-n_var)*2))
             for adj, n_var in zip(adjs, n_vars)]
    
    sparse_adjs = [torch_sparse.SparseTensor.from_dense(adj) for adj in adjs]
    inds = [torch.nn.functional.pad(ind, (0, max_size-n_var), value=0).view(-1) for ind, n_var in zip(inds, n_vars)]
    inds = [torch.nn.functional.pad(ind, (0, max_size-n_var), value=1).view(-1) for ind, n_var in zip(inds, n_vars)]

    fs = [ind.clone() for ind in inds]
    n_vars = [torch.nn.functional.pad(torch.ones(n_var), (0, max_size-n_var), value=0) for n_var in n_vars]
    if solutions[0] != -1:
        solutions = [torch.nn.functional.pad(torch.tensor(sol), (0, max_size-len(sol)), value=0) for sol in solutions]
    else:
        solutions = [torch.tensor(sol) for sol in solutions]
    return sparse_adjs, inds, fs, n_vars, solutions


class OurSATBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.name = "OSATBase"
    
    @abstractmethod
    def forward():
        return NotImplementedError
    
    
    @staticmethod
    def cnf_to_ours(clauses, nv):
        """
        Same as for CircuitSAT but with different indicators values
        (0: pos_lit, 1: neg_lit, 2: or/clauses, 3: and/root)
        """
        adj = torch.zeros(nv * 2, nv * 2)  # Adjacency matrix
        idx = torch.arange(nv * 2).view(2, -1).T
        for (i, j) in idx:  # Add direct edge between postive literal and its negation
            adj[i, j] = 1

        # Convert negation literals to indices
        t_cnf = [[abs(l) + nv if l < 0 else l for l in c] for c in clauses]

        # Make indicator for each node (literal) of the graph, mark pos with 0 and neg with 1
        indicator = torch.zeros(1, 2 * nv)
        indicator[:, nv:nv*2] = 1

        # For each clause add a new node
        for i, c in enumerate(t_cnf):
            # Add zero row and column to adjacency matrix
            adj = torch.nn.functional.pad(adj, (0, 1, 0, 1))
            # Add direct edge for each literal index in the clause to the new node (clause)
            for l in c:
                adj[l - 1, -1] = 1
            # Add a new position with 1 for the new added node
            indicator = torch.nn.functional.pad(indicator, (0, 1), value=2)

            # Save the index of the clause node (is the last element), for connecting with the root node later
            t_cnf[i] = adj.size(0)

        # Add a final node (root)
        adj = torch.nn.functional.pad(adj, (0, 1, 0, 1))

        # Add a new postion for the root with value 3
        indicator = torch.nn.functional.pad(indicator, (0, 1), value=3)

        # For each clause index add a direct edge to the last (root) node
        for l in t_cnf:
            adj[l - 1, -1] = 1
        
        # Generate an instance for our model
        csat_instance = {
            'clauses': clauses,
            'n_clauses': len(clauses),
            'n_vars': nv,
            'adj': adj,
            'ind': indicator
        }
        return csat_instance
    

    @staticmethod
    def evaluate_circuit(sample, emb, epoch, eps=1.2, hard=False):
        # explore exploit with annealing rate
        t = epoch**(-eps)
        inds = torch.cat(sample['ind'], dim=0).view(-1)

        # set to negative to make sure we don't accidentally use nn preds for or/and
        temporary = emb.clone()
        temporary[inds == 2] = -1
        temporary[inds == 3] = -1

        # NOT gate
        #temporary[sample['features'][:, 1] == 1] = 1 - emb[sample['features'][:, 0] == 1].clone()
        temporary[inds == 1] = 1 - emb[inds == 0].clone()
        emb = temporary.clone()

        # OR gate
        idx = torch.arange(inds.size(0))[inds==2]
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
            e_temp = torch_sparse.SparseTensor(row=row, rowptr=None, col=col, value=e_temp, sparse_sizes=e_gated.sizes())
            e_gated = sparse_elem_mul(e_gated, e_temp).sum(dim=0) / e_temp.sum(dim=0)
            assert torch.all(torch.eq(e_gated, e_gated)) # make sure there are no nan values

        # AND gate
        idx2 = torch.arange(inds.size(0))[inds==3]
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
        n_vars = sample['n_vars'].size(0)
        assert len(e_gated) == n_vars, f"{len(e_gated)}, {n_vars}"
        return e_gated


    @staticmethod
    def collate_fn(batch):
        is_sat, solution, n_vars, clauses = [], [], [], []
        single_adjs, inds, fnames, fs = [], [], [], []
        n_clauses = []
        var_labels = []
        for sample in batch:
            cnf = sample['cnf']
            p = OurSATBase.cnf_to_ours(cnf.clauses, cnf.nv)
            single_adjs.append(p['adj'])
            n_vars.append(p['n_vars'])
            clauses.append(p['clauses'])
            n_clauses.append(len(p['clauses']))
            is_sat.append(sample['is_sat'])
            inds.append(p['ind'])
            fnames.append(sample['file'])
            solution.append(sample['solution'])
            c = np.array(sample['solution'])
            c = c[c > 0] - 1
            l = torch.zeros(p['n_vars'])
            l[c] = 1
            var_labels.append(l)
        single_adjs, inds, fs, n_vars, solutions = pad(single_adjs, inds, n_vars, solution)
        adj = sparse_blockdiag(single_adjs)
        indicators = torch.cat(fs).squeeze(0)
        assert indicators.size(0) == adj.size(0) == adj.size(1), f"{indicators.size(0)}, {adj.size(0)} , {adj.size(1)}"
        feats = torch.zeros((indicators.size(0), 4))
        feats = torch.scatter(feats, 1, indicators.long().unsqueeze(1), torch.ones(feats.shape))
        assert torch.all(feats.sum(-1) == 1)
        decision = torch.full((feats.size(0), 1), 2).int()
        feats = torch.cat([indicators.view(-1, 1), decision], dim=1)
        new_batch = {
            'batch_size': len(batch),
            'n_vars': torch.stack(n_vars),
            'is_sat': torch.Tensor(is_sat).float(),
            'adj': adj,
            'ind': inds,
            'clauses': clauses,
            'solution': torch.stack(solutions),
            'fnames': fnames,
            'features': feats,
            'varlabel': var_labels
        }
        return new_batch