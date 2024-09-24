import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, out_dim)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.r(self.l1(x))
        x = self.r(self.l2(x))
        x = self.l3(x)
        return x


class NeuroSAT(nn.Module):
    def __init__(self, dim=128, n_rounds=26, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.name = "NSAT"
        self.dim = dim
        self.n_rounds = n_rounds

        self.init_ts = torch.ones(1)

        self.L_init = nn.Linear(1, dim)
        self.C_init = nn.Linear(1, dim)

        self.L_msg = MLP(dim, dim, dim)
        self.C_msg = MLP(dim, dim, dim)

        self.L_update = nn.LSTM(dim*2, dim)
        self.C_update = nn.LSTM(dim, dim)

        self.L_vote = MLP(dim, dim, 1)
        self.denom = torch.sqrt(torch.Tensor([dim]))

    def forward(self, batch):
        self.last_state = None
        n_vars = batch['n_vars'].sum()
        n_clauses = batch['n_clauses'].sum()
        n_probs = len(batch['n_clauses'])
        adj = batch['adj']

        init_ts = self.init_ts.cuda()
        L_init = self.L_init(init_ts).view(1, 1, -1)
        L_init = L_init.repeat(1, n_vars*2, 1)
        C_init = self.C_init(init_ts).view(1, 1, -1)
        C_init = C_init.repeat(1, n_clauses, 1)

        L_state = (L_init, torch.zeros(1, n_vars*2, self.dim).cuda())
        C_state = (C_init, torch.zeros(1, n_clauses, self.dim).cuda())

        for _ in range(self.n_rounds):
            L_hidden = L_state[0].squeeze(0)
            L_pre_msg = self.L_msg(L_hidden)
            LC_msg = torch.matmul(adj.T, L_pre_msg)

            _, C_state = self.C_update(LC_msg.unsqueeze(0), C_state)

            C_hidden = C_state[0].squeeze(0)
            C_pre_msg = self.C_msg(C_hidden)
            CL_msg = torch.matmul(adj, C_pre_msg)

            _, L_state = self.L_update(torch.cat([CL_msg, flip(L_state[0].squeeze(0), n_vars)], dim=1).unsqueeze(0),
                                       L_state)

        logits = L_state[0].squeeze(0)
        self.last_state = logits
        vote = self.L_vote(logits)
        
        # reshape such that we have a vector of length 2 for every variable (literal & complement)
        vote_join = torch.cat([vote[:n_vars, :], vote[n_vars:, :]], dim=1)
        # split tensor into votes for every batch, as they do not have the same dimensions
        vote_join = list(torch.split(vote_join, list(batch['n_vars'])))
        vote_mean = torch.stack([torch.mean(x) for x in vote_join]).to(adj.device)
        
        assert len(vote_join) == n_probs
        return vote_mean, vote.detach().cpu(), logits.detach().cpu()

    @staticmethod
    def collate_fn(batch):
        """
        Collate fn for torch.DataLoader to parse a batch of problem attributes into one batch dict
        """
        # a for-loop is currently neccessary because pytorch does not support ragged tensors
        # we have a varying number of clauses & literals
        is_sat, n_clauses, n_vars, clauses = [], [], [], []
        single_adjs, adj_pos, adj_neg, solutions = [], [], [], []
        fnames = []
        all_cells = []
        
        for sample in batch:
            cnf = sample['cnf']
            p = NeuroSAT.cnf_to_nsat(cnf.clauses, cnf.nv)
            adj = p['adj_nsat'].to_dense()
            single_adjs.append(adj)
            # re-sort adjacency such that regular literals come first and then, in the same order, the negated vars
            adj_pos.append(adj[:int(adj.shape[0]/2), :])
            adj_neg.append(adj[int(adj.shape[0]/2):, :])
            solutions.append(sample['solution'])
            is_sat.append(sample['is_sat'])
            
            all_cells.append(p['n_cells'])
            n_vars.append(p['n_vars'])
            n_clauses.append(p['n_clauses'])
            clauses.append(p['clauses'])
            fnames.append(sample['file'])

        # create disconnected graphs
        adj_pos = torch.block_diag(*adj_pos)
        adj_neg = torch.block_diag(*adj_neg)

        new_batch = {
            'batch_size': len(batch),
            'n_vars': torch.Tensor(n_vars).int(),
            'n_clauses': torch.Tensor(n_clauses).int(),
            'is_sat': torch.Tensor(is_sat).float(),
            'adj': torch.cat([adj_pos, adj_neg]),
            'single_adjs': single_adjs,
            'clauses': clauses,
            'solution': solutions,
            'fnames': fnames,
            'n_cells_per_sample': all_cells
        }
        return new_batch
    

    @staticmethod
    def reconstruct( repr, batch):
        nv = torch.cumsum(torch.cat((torch.zeros(1), batch['n_vars'])), dim=0).int()*2
        nc = torch.cumsum(torch.cat((torch.zeros(1), batch['n_clauses'])), dim=0).int()
        pos, neg = [], []
        for i in range(nv.size(0)-1):
            subm = repr[nv[i]:nv[i+1], nc[i]:nc[i+1]].clone()
            pos.append(subm.view(2, -1, subm.size(1))[0])
            neg.append(subm.view(2, -1, subm.size(1))[1])
        batch['adj'] = torch.cat([torch.block_diag(*pos), torch.block_diag(*neg)])
        return batch


    @staticmethod
    def get_representation(batch: dict):
        nv = torch.cumsum(torch.cat((torch.zeros(1), batch['n_vars'])), dim=0).int()
        nc = torch.cumsum(torch.cat((torch.zeros(1), batch['n_clauses'])), dim=0).int()
        adj = batch['adj'].view(2, -1, batch['adj'].size(1))
        chunks = []
        for i in range(nv.size(0)-1):
            subm = adj[:, nv[i]:nv[i+1], nc[i]:nc[i+1]]
            chunks.append(torch.cat((subm[0], subm[1])))
        repr = torch.block_diag(*chunks)
        assert batch['n_vars'].sum()*2 == repr.size(0)
        assert sum(batch['n_clauses']) == repr.size(1)
        block_check = torch.block_diag(*[torch.ones(x*2, y) for (x, y) in zip(batch['n_vars'], batch['n_clauses'])])
        assert torch.all(torch.eq(repr, repr*block_check.to(repr.device)))
        return repr

    @staticmethod
    def cnf_to_nsat(clauses, nv):
        n_cells = sum([len(clause) for clause in clauses])
        # construct adjacency for neurosat model
        nsat_indices = np.zeros([n_cells, 2], dtype=np.int64)
        cell = 0
        for clause_idx, iclause in enumerate(clauses):
            vlits = [ilit_to_vlit(x, nv) for x in iclause]
            for vlit in vlits:
                nsat_indices[cell, :] = [vlit, clause_idx]
                cell += 1
        assert(cell == n_cells)
        adj_nsat = torch.sparse.FloatTensor(torch.Tensor(nsat_indices).T.long(), torch.ones(n_cells), 
                                            torch.Size([nv*2, len(clauses)]))
        nsat_instance = {
            "clauses": clauses,
            "n_clauses": len(clauses),
            "n_vars": nv,
            "adj_nsat": adj_nsat,
            "n_cells": n_cells
        }
        return nsat_instance

    # WARNING: if the same clause has repeated literals then the number of cells != adj_indices from sparse adj (It doesn't count repetitions),
    # actually the repetitions get value 2 (which is WRONG)
    @staticmethod
    def find_solutions(batch, all_votes, final_lits):
        # batch is an object of lists with len = batch_size = n_samples
        # all_votes = [pos+neg, vote], all_votes.shape = (n_vars*2, 1) -> votes
        # final_lits = [pos+neg, emb], final_lits.shape = (n_vars*2, hidden_dim) -> embeddings

        def flip_vlit(vlit):
            # get the oposite literal (pos/neg)
            if vlit < n_vars: return vlit + n_vars
            else: return vlit - n_vars

        batch_size = batch['batch_size']
        n_vars = batch['n_vars'].sum()
        n_vars_ranges = torch.cumsum(torch.cat((torch.zeros(1).cuda(), batch['n_vars'])), dim=0).int()

        solutions = []
        for sample in range(batch_size):
            decode_cheap_A = (lambda vlit: all_votes[vlit, 0] > all_votes[flip_vlit(vlit), 0]) # x_i > ~x_i (Delta_1)
            decode_cheap_B = (lambda vlit: not decode_cheap_A(vlit)) # ~x_i < x_i (Delta_2) 

            def reify(phi):
                xs = list(zip([phi(vlit) for vlit in range(n_vars_ranges[sample], n_vars_ranges[sample+1])],
                            [phi(flip_vlit(vlit)) for vlit in range(n_vars_ranges[sample], n_vars_ranges[sample+1])]))
                # The assertion might fail if the predictions of neg and pos have equal value, e.g., random weights initialization.
                # During the small test of the validation loader it fails.
                # def one_of(a, b): return (a and (not b)) or (b and (not a))
                # assert(all([one_of(x[0], x[1]) for x in xs]))
                return [x[0] for x in xs]

            if solves(batch, sample, decode_cheap_A): solutions.append(reify(decode_cheap_A))
            elif solves(batch, sample, decode_cheap_B): solutions.append(reify(decode_cheap_B))
            else:
                # final_lits.shape = (n_vars*2, dim) -> (sum([n_vars for each sample])*2, dim)
                # final_lits = [pos+neg, emb]
                pos = final_lits[n_vars_ranges[sample]: n_vars_ranges[sample+1], :]
                neg = final_lits[n_vars+n_vars_ranges[sample]: n_vars+n_vars_ranges[sample+1], :]
                lits = torch.cat([pos, neg], axis=0)
                kmeans = KMeans(n_clusters=2, random_state=0).fit(lits)
                distances = kmeans.transform(lits)
                scores = distances * distances

                def proj_vlit_flit(vlit):
                    if vlit < n_vars: return vlit - n_vars_ranges[sample]
                    else: return ((vlit - n_vars) - n_vars_ranges[sample]) + batch['n_vars'][sample]

                def decode_kmeans_A(vlit):
                    return scores[proj_vlit_flit(vlit), 0] + scores[proj_vlit_flit(flip_vlit(vlit)), 1] > \
                        scores[proj_vlit_flit(vlit), 1] + scores[proj_vlit_flit(flip_vlit(vlit)), 0]

                decode_kmeans_B = (lambda vlit: not decode_kmeans_A(vlit))

                if solves(batch, sample, decode_kmeans_A): solutions.append(reify(decode_kmeans_A))
                elif solves(batch, sample, decode_kmeans_B): solutions.append(reify(decode_kmeans_B))
                else: solutions.append(None)
            
        # Transform from bools to ints. Return None if solution is None (unsat)
        solutions = [[(-1)**(not x)*i for i, x in enumerate(sol, 1)] if sol else None for sol in solutions]
        return solutions



def solves(batch, sample, phi):
    start_cell = sum(batch['n_cells_per_sample'][0:sample])
    end_cell = start_cell + batch['n_cells_per_sample'][sample]
    if start_cell == end_cell:
        # no clauses
        return True
    
    adj = batch['adj'] # x_i and ~x_i is separeted by n_vars = sum(n_vars)
    adj_indices = adj.to_sparse().indices().T
    _, sorted_indices = torch.sort(adj_indices[:, 1], dim=0)
    # The same order when constrcuted the adj is needed, sparse tensor is modifying the order
    adj_indices = adj_indices[sorted_indices]
    
    current_clause = adj_indices[start_cell, 1]
    current_clause_satisfied = False
    for cell in range(start_cell, end_cell):
        next_clause = adj_indices[cell, 1]

        # the current clause is over, so we can tell if it was unsatisfied
        if next_clause != current_clause:
            if not current_clause_satisfied:
                return False

            current_clause = next_clause
            current_clause_satisfied = False

        if not current_clause_satisfied:
            vlit = adj_indices[cell, 0]
            if phi(vlit):
                current_clause_satisfied = True

    # edge case: the very last clause has not been checked yet
    if not current_clause_satisfied: return False
    return True


def ilit_to_var_sign(x):
    assert(abs(x) > 0)
    var = abs(x) - 1
    sign = x < 0
    return var, sign

def ilit_to_vlit(x, n_vars):
    assert(x != 0)
    var, sign = ilit_to_var_sign(x)
    if sign: return var + n_vars
    else: return var

def flip(msg, n_vars):
    return torch.cat([msg[n_vars:2*n_vars, :], msg[:n_vars, :]], dim=0)


    