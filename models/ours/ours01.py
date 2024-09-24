import torch
import numpy as np
import torch_sparse
import torch.nn as nn

from .base import OurSATBase, MLP, pad
from ..utils import sparse_blockdiag


class CSAT(nn.Module):
    def __init__(self, dim=100, dim_agg=50, dim_class=30, n_rounds=20):
        super().__init__()
        self.name = "CSAT"
        self.dim = dim
        self.n_rounds = n_rounds
        self.init = MLP(200, dim, dim)
        self.forward_msg = MLP(dim, dim_agg, dim)
        self.backward_msg = MLP(dim, dim_agg, dim)
        self.forward_update = nn.GRU(dim, dim)
        self.backward_update = nn.GRU(dim, dim)
        self.classif = MLP(dim, dim_class, 1)
        self.emb_var = nn.Embedding(4, 100)
        self.emb_dec = nn.Embedding(3, 100)


    def forward(self, sample):
        self.forward_update.flatten_parameters()
        self.backward_update.flatten_parameters()
        adj = sample['adj']
        # h_state = self.init(sample['features'].cuda()).unsqueeze(0)
        emb_var_type = self.emb_var(sample['features'][:, 0].int().cuda())
        emb_decision = self.emb_dec(sample['features'][:, 1].int().cuda())
        h_state = self.init(torch.cat((emb_var_type, emb_decision), dim=1)).unsqueeze(0)
        for _ in range(self.n_rounds):
            f_pre_msg = self.forward_msg(h_state.squeeze(0)) # Pass through the GRU
            f_msg = torch_sparse.matmul(adj, f_pre_msg)

            _, h_state = self.forward_update(f_msg.unsqueeze(0), h_state)

            b_pre_msg = self.backward_msg(h_state.squeeze(0))
            b_msg = torch_sparse.matmul(adj.t(), b_pre_msg)

            _, h_state = self.backward_update(b_msg.unsqueeze(0), h_state)
            
        out = h_state.squeeze(0)
        out = self.classif(out)
        return out
    

class OurSAT01(OurSATBase):

    def __init__(self, eps, n_rounds):
        super().__init__()
        self.name = "Ours01"
        self.enc = CSAT(n_rounds=n_rounds)
        self.eps = eps

    def forward(self, batch, epoch=1, test=False):
        max_n_steps = batch['n_vars'].size(1)
        batch_size = batch['batch_size']
        inds = torch.cat(batch['ind'], dim=0)
        # Make adj indicators for the max of each adj
        rows_inds = [torch.nn.functional.one_hot(torch.tensor(x), batch_size) for x in range(batch_size)]
        adj_inds = [row.repeat(ind.size(0), 1) for ind, row in zip(batch['ind'], rows_inds)]
        adj_inds = torch.cat(adj_inds, dim=0).cuda()
        
        preds = []
        decided = torch.zeros_like(inds)
        for step in range(max_n_steps):
            # Calculate embedding of the variables
            pred = torch.sigmoid(self.enc(batch)) # (pred, 1)
            
            # Split pred for each formula of the batch (columns) (pred, batch_size) 
            all_pred = pred.repeat(1, batch_size)

            # Select only non-decided positive variables and get the most likely for each formula of the batch
            pos_inds = (inds.view(-1, 1).repeat(1, batch_size) == 0).float()
            all_pred = torch.abs(all_pred - 0.5) + 0.1
            non_decided = 1 - decided.view(-1, 1).repeat(1, batch_size)
            pos_all_pred = all_pred*adj_inds*pos_inds*non_decided
            _, max_indices = torch.max(pos_all_pred, dim=0) #Across first dimension, i.e., max for each column

            # At each step batch_size decisions have to be performed
            sum_prev = torch.sum(decided)
            decided[max_indices] = 1
            assert sum_prev + batch_size == torch.sum(decided), f"{sum_prev}, {torch.sum(decided)}"

            # Update node features of the decided variables
            values = (pred[max_indices] > 0.5).float() # The gradient is lost here, i.e., f(x) = x > k non-diff 
            
            batch['features'] = batch['features'].clone()
            batch['features'][max_indices, -1] = values.view(-1)

            # Evaluating the already reduced (condicionally decided) circuit
            s = self.evaluate_circuit(batch, pred, epoch, self.eps, hard=test)
            preds.append(s)

        preds = torch.stack(preds)
        return preds