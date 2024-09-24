import torch
import torch_sparse


def step_loss(outputs, k=10, mean=True):
    # L loss -> smooth step function
    loss = (1 - outputs)**k / (((1 - outputs)**k) + outputs**k)
    return loss.mean() if mean else loss


def discounted_step_loss(outputs, k=10, gamma=0.9, mean=True):
    loss = step_loss(outputs, k, mean=False)
    disc = gamma**torch.arange(outputs.size(0)).float().to(loss.device)
    discounted_loss = torch.matmul(disc, loss)
    return discounted_loss.mean() if mean else discounted_loss


def norm_discounted_step_loss(outputs, k=10, gamma=0.9, mean=True):
    discounted_loss = discounted_step_loss(outputs, k, gamma, False)
    normalized_discounted_loss = (discounted_loss - discounted_loss.mean()) / (discounted_loss.std() + 1e-8)  # Normalize the loss
    return normalized_discounted_loss.mean() if mean else normalized_discounted_loss



def sparse_elem_mul(s1, s2):
    # until torch_sparse support elementwise multiplication, we have to do this
    s1 = s1.to_torch_sparse_coo_tensor()
    s2 = s2.to_torch_sparse_coo_tensor()
    return torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(s1 * s2)


def sparse_blockdiag(single_adjs, square=True):
    single_adjs = [(sa.coo(), sa.size(0), sa.size(1)) for sa in single_adjs]
    values = torch.cat([sa[0][2] for sa in single_adjs])
    indices = []
    counter = 0
    sec_counter = 0
    for sa in single_adjs:
        if square:
            indices.append(torch.stack((sa[0][0], sa[0][1])) + counter)
        else:
            indices.append(torch.stack((sa[0][0] + counter, sa[0][1] + sec_counter)))
        counter += sa[1]
        sec_counter += sa[2]
    indices = torch.cat(indices, dim=1)
    if square: assert counter == sec_counter
    adj = torch.sparse_coo_tensor(indices, values, [counter, sec_counter])
    adj = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(adj)
    return adj


def forward_mat(self, batch):
        # adj: [32*steps, 32*steps]
        # adj, step=1:
        #| [adj_1]           |
        #|      [...]        | 
        #|           [adj_32]|
        # pred: [32*steps, 1]
        # sol: [32, steps]
        # gt: [32, steps]
        max_n_steps = batch['n_vars'].size(1)
        
        # Calculate embedding of the variables
        emb = self.enc(batch)
        
        # Select one embedding from the pos (0) and neg (1) variables and predict (decide: True, False)
        step_inds = torch.eye(max_n_steps).repeat(1, self.batch_size).view(-1).bool()

        inds = batch['ind'].view(-1) # all indicators
        var_pos = emb[inds == 0] # get positive variables emb
        var_neg = emb[inds == 1] # get negative variables emb
        var = torch.cat((var_pos[step_inds], var_neg[step_inds]), dim=1)
        pred = self.decide(var).flatten()
        # print("emb:", emb.shape, emb)
        # print("var_pos:", var_pos.shape, var_pos)
        # print("var_neg:", var_neg.shape, var_neg)
        # print("var:", var.shape, var)
        # print("pred:", pred.shape, pred)
        
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

def reduce_graph(batch, step_ind, gt):
    _batch = copy.deepcopy(batch)
    adj = _batch['adj'].to_dense().cuda()
    inds = torch.cat(_batch['ind'], dim=1).squeeze()

    # pos literals (0) neg literals (1)
    tmp_pos = adj[inds == 0]
    tmp_neg = adj[inds == 1]
    zeros = torch.zeros_like(tmp_pos[step_ind, :])
    tmp_pos[step_ind,:] = zeros.clone().cuda()
    tmp_neg[step_ind,:] = zeros.clone().cuda()
    adj[inds == 0] = tmp_pos
    adj[inds == 1] = tmp_neg

    batch['adj'] = torch_sparse.SparseTensor.from_dense(adj)
    return batch