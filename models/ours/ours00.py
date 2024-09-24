import torch
import numpy as np
import torch_sparse
import torch.nn as nn

from .base import OurSATBase, MLP, pad
from ..utils import sparse_blockdiag


class CSAT(nn.Module):
    def __init__(self, in_dim=4, dim=100, dim_agg=50, dim_class=30, n_rounds=20):
        super().__init__()
        self.name = "CSAT"
        self.dim = dim
        self.n_rounds = n_rounds
        self.init = nn.Linear(in_dim, dim)
        self.forward_msg = MLP(dim, dim_agg, dim)
        self.backward_msg = MLP(dim, dim_agg, dim)
        self.forward_update = nn.GRU(dim, dim)
        self.backward_update = nn.GRU(dim, dim)


    def forward(self, sample):
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
            
        out = h_state.squeeze(0)
        return out
    

class OurSAT00(OurSATBase):

    def __init__(self, n_rounds):
        super().__init__()
        self.name = "Ours00"
        self.enc = CSAT(n_rounds=n_rounds)
        self.decide = MLP(in_dim=100*2, hidden_dim=30, out_dim=1)
        

    def forward(self, batch):
        # adj: [32*steps, 32*steps]
        # adj, step=1:
        #| [adj_1]           |
        #|      [...]        | 
        #|           [adj_32]|
        # pred: [32*steps, 1]
        # sol: [32, steps]
        # gt: [32, steps]
        max_n_steps = batch['n_vars'].size(1)
        batch_size = batch['batch_size']
        
        # Calculate embedding of the variables
        emb = self.enc(batch)
        
        # Select one embedding from the pos (0) and neg (1) variables and predict (decide: True, False)
        step_inds = torch.eye(max_n_steps).repeat(1, batch_size).view(-1).bool()

        inds = batch['ind'].view(-1) # all indicators
        var_pos = emb[inds == 0] # get positive variables emb
        var_neg = emb[inds == 1] # get negative variables emb
        var = torch.cat((var_pos[step_inds], var_neg[step_inds]), dim=1)
        pred = self.decide(var).flatten()
        
        # Get ground truth and solution
        solutions = batch['solution'].T.reshape(-1)
        gt = (solutions > 0).float()

        # Compute loss and accuracy
        preds = pred[solutions != 0]
        labels = gt[solutions != 0]
        assert pred.shape == gt.shape
        assert preds.shape == labels.shape
        # Return pred (logit) because we are using loss BCEWithLogits
        return preds, labels
    

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
        adj, inds, fs = preprocessing(adj, inds, n_vars, len(batch))
        assert fs.size(0) == adj.size(0) == adj.size(1), f"{fs.size(0)}, {adj.size(0)} , {adj.size(1)}"
        feats = torch.zeros((fs.size(0), 4))
        feats = torch.scatter(feats, 1, fs.long().unsqueeze(1), torch.ones(feats.shape))
        assert torch.all(feats.sum(-1) == 1)

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


def reduce_adj(adj, inds, max_n_steps, batch_size):
    adj = adj.to_dense()
    reduced_adjs = []

    for step in range(max_n_steps):
        step_ind = torch.nn.functional.one_hot(torch.tensor(step), max_n_steps).squeeze()
        step_ind = step_ind.repeat(batch_size).bool()

        # pos literals (0) neg literals (1)
        tmp_pos = adj[inds == 0]
        tmp_neg = adj[inds == 1]
        zeros = torch.zeros_like(tmp_pos[step_ind, :])
        tmp_pos[step_ind,:] = zeros.clone()
        tmp_neg[step_ind,:] = zeros.clone()
        adj[inds == 0] = tmp_pos
        adj[inds == 1] = tmp_neg
        new_adj = torch_sparse.SparseTensor.from_dense(adj)
        reduced_adjs.append(new_adj)
    del adj
    return sparse_blockdiag(reduced_adjs)


def preprocessing(adj, inds, n_vars, batch_size):
    inds = torch.cat(inds, dim=0).squeeze()
    max_n_steps = n_vars[0].size(0) # all formulas has indicator of n_vars (1) and pad (0), same size
    reduced_adjs = reduce_adj(adj, inds, max_n_steps, batch_size)
    reduced_inds = inds.repeat(max_n_steps)
    return reduced_adjs, reduced_inds, reduced_inds