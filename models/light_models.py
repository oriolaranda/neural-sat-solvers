import datetime
import time
import torch
import torch.nn as nn
import lightning.pytorch as pl
from sklearn.metrics import accuracy_score
from pysat.solvers import Solver
from abc import abstractmethod

from .baselines.circuitsat import CircuitSAT
from .baselines.neurosat import NeuroSAT
from .utils import step_loss, discounted_step_loss, norm_discounted_step_loss
from .ours.ours00 import OurSAT00
from .ours.ours01 import OurSAT01
from .ours.ours03 import OurSAT03
from .ours.ours04 import OurSAT04
from .ours.ours05 import OurSAT05
from .ours.ours06 import OurSAT06


class BaseLight(pl.LightningModule):
    """
    Base Lightning Module, to avoid redundancies on the code. Basically for logging purposes.
    """

    def __init__(self, lr, weight_decay, batch_size, dataset_names):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.train_set = dataset_names.setdefault('train', "")
        self.valid_set = dataset_names.setdefault('valid', "")
        self.test_set = dataset_names['test'] # list of multiple names
        self.test_times = {}

    @abstractmethod
    def _training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def _shared_eval_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self._training_step(batch, batch_idx)
        logs = {f"train_loss": loss}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        acc, loss = self._shared_eval_step(batch, batch_idx)
        
        logs = {f"val_acc": acc, f"val_loss": loss}
        self.log_dict(logs, prog_bar=True, batch_size=self.batch_size)
        return acc, loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        acc, loss = self._shared_eval_step(batch, batch_idx)
        self.test_times.setdefault(dataloader_idx, []).append(time.perf_counter() - self.start_time)

        logs = {f"test_acc/{self.test_set[dataloader_idx]}": acc, f"test_loss/{self.test_set[dataloader_idx]}": loss}
        self.log_dict(logs, prog_bar=True, batch_size=self.batch_size, add_dataloader_idx=False)
        return acc, loss
    
    def on_test_start(self):
        self.start_time = time.perf_counter()


    def on_test_end(self):
        print("Times on inference")
        for k, v in self.test_times.items():
            elapsed_time = str(datetime.timedelta(seconds=max(v)-min(v)))
            print(self.test_set[k], ":", elapsed_time)
        self.elapsed_time = time.perf_counter() - self.start_time

    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optim
    

# TODO: Implementation PDP
class PDPLight(BaseLight):
    """
    PDP Lightning Implementation
    """
    
    def __init__(self, lr, weight_decay, batch_size, eps, dataset_names):
        super().__init__(dataset_names)
        self.model = ...
        self.loss = step_loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.eps = eps
        self.save_hyperparameters()
    
    # class attribute
    collate_fn = CircuitSAT.collate_fn
    
    def _training_step(self, batch, batch_idx):
        self.model.forward_update.flatten_parameters()
        self.model.backward_update.flatten_parameters()

        out = self.model(batch)
        epoch = self.current_epoch
        out = CircuitSAT.evaluate_circuit(batch, torch.sigmoid(out), epoch+1, eps=self.eps)
        loss = self.loss(out)
        return loss
    

    def _shared_eval_step(self, batch, batch_idx):
        out = self.model(batch)
        out = CircuitSAT.evaluate_circuit(batch, torch.sigmoid(out), 1, eps=self.eps, hard=True)
        label = batch['is_sat'].cpu()
        hard_pred = (out.detach().flatten() > 0.5).float().cpu()
        acc = accuracy_score(label.numpy(), hard_pred.numpy()).item()
        loss = self.loss(out)
        return acc, loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optim


class NeuroSATLight(BaseLight):
    """
    NeuroSAT Lightning Implementation
    """
    def __init__(self, eps, n_rounds, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.model = NeuroSAT(n_rounds=n_rounds)
        self.loss = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()
    
    # class attribute
    collate_fn = NeuroSAT.collate_fn

    def _training_step(self, batch, batch_idx):
        out, _, _ = self.model(batch)
        loss = self.loss(out, batch['is_sat'])
        return loss

    def _shared_eval_step(self, batch, batch_idx, test):
        batch_size = batch['batch_size']
        mean_votes, all_votes, final_lits = self.model(batch)

        target = batch['is_sat']
        assert len(target) == batch_size
        loss = self.loss(mean_votes, target) # loss take logits
        
        pred = torch.sigmoid(mean_votes)
        hard_pred = (pred.detach().flatten() > 0.5).float()
        bin = accuracy_score(target.cpu().numpy(), hard_pred.cpu().numpy()).item()
        acc = miss = None
        
        if test:
            solutions = NeuroSAT.find_solutions(batch, all_votes, final_lits)
            # If the solution is not empty (None), we are sure it's a valid solution
            solved = list(map(bool, solutions))
            acc = sum(solved)/batch_size
            miss = sum(s and not p for s, p in zip(solved, hard_pred))/batch_size
        return acc, bin, loss, miss
    
    def validation_step(self, batch, batch_idx):
        # For validation with true and false samples, doesn't make sense to count % solved,
        # makes sense to log the binary classification (sat) accuracy
        _, bin, loss, _ = self._shared_eval_step(batch, batch_idx, test=False)
        
        logs = {f"val_acc": bin, f"val_loss": loss}
        self.log_dict(logs, prog_bar=True, batch_size=self.batch_size)
        return bin, loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        acc, sat, loss, miss = self._shared_eval_step(batch, batch_idx, test=True)
        self.test_times.setdefault(dataloader_idx, []).append(time.perf_counter() - self.start_time)
        
        logs = {f"test_acc/{self.test_set[dataloader_idx]}": acc, f"test_loss/{self.test_set[dataloader_idx]}": loss,
                f"test_binary/{self.test_set[dataloader_idx]}": sat, f"test_miss/{self.test_set[dataloader_idx]}": miss}
        self.log_dict(logs, prog_bar=True, batch_size=self.batch_size, add_dataloader_idx=False)
        return acc, sat, loss
    
    
    
class CircuitSATLight(BaseLight):
    """
    CircuitSAT Lightning Implementation
    """
    
    def __init__(self, eps, n_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = CircuitSAT(n_rounds=n_rounds)
        self.loss = step_loss
        self.eps = eps
        self.save_hyperparameters()
    
    # class attribute
    collate_fn = CircuitSAT.collate_fn
    
    def _training_step(self, batch, batch_idx):
        self.model.forward_update.flatten_parameters()
        self.model.backward_update.flatten_parameters()

        out = self.model(batch)
        epoch = self.current_epoch
        out = CircuitSAT.evaluate_circuit(batch, torch.sigmoid(out), epoch+1, eps=self.eps)
        loss = self.loss(out)
        return loss
    

    def _shared_eval_step(self, batch, batch_idx):
        print("ROUNDS", self.model.n_rounds)
        out = self.model(batch)
        out = CircuitSAT.evaluate_circuit(batch, torch.sigmoid(out), 1, eps=self.eps, hard=True)
        label = batch['is_sat'].cpu()
        hard_pred = (out.detach().flatten() > 0.5).float().cpu()
        acc = accuracy_score(label.numpy(), hard_pred.numpy()).item()
        loss = self.loss(out)
        return acc, loss
    
    


class OurSAT00Light(BaseLight):
    """
    OurSAT00 Lightning Implementation
    """
    
    def __init__(self, eps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = OurSAT00()
        self.loss = nn.BCEWithLogitsLoss() # Binary cross entropy
        self.eps = eps
        self.save_hyperparameters()
        
    collate_fn = OurSAT00.collate_fn
    
    def _training_step(self, batch, batch_idx):
        self.model.enc.forward_update.flatten_parameters()
        self.model.enc.backward_update.flatten_parameters()
        preds, labels = self.model(batch)
        loss = self.loss(preds, labels)
        return loss
    

    def _shared_eval_step(self, batch, batch_idx):
        preds, labels = self.model(batch)
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        loss = self.loss(preds, labels).item()
        hard_preds = (torch.sigmoid(preds) > 0.5)
        acc = accuracy_score(labels.float(), hard_preds.float()).item()
        return acc, loss
    

    

class OurSAT01Light(BaseLight):
    """
    OurSAT01 Lightning Implementation
    """
    
    def __init__(self, eps, gamma, tau, beta, n_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = OurSAT01(eps, n_rounds)
        self.loss = step_loss
        self.eps = eps
        self.gamma = gamma
        self.save_hyperparameters()
        
    collate_fn = OurSAT01.collate_fn
    
    def _training_step(self, batch, batch_idx):
        self.model.enc.forward_update.flatten_parameters()
        self.model.enc.backward_update.flatten_parameters()
        epoch = self.current_epoch
        preds = self.model(batch, epoch+1, test=False)
        loss = self.loss(preds)
        return loss
    

    def _shared_eval_step(self, batch, batch_idx):
        preds = self.model(batch, test=True).detach().cpu()
        # preds.shape = (num_steps, batch_size)

        loss = self.loss(preds).item()
        final_preds = preds[-1, :] # we are interested in last pred assignment
        hard_preds = (final_preds.flatten() > 0.5).float()
        labels = batch['is_sat'].cpu()
        assert labels.shape == hard_preds.shape
        acc = accuracy_score(labels, hard_preds).item()
        return acc, loss
    
    
    

class OurSAT02Light(BaseLight):
    """
    OurSAT02 Lightning Implementation
    """
    
    def __init__(self, eps, gamma, tau, beta, n_rounds, *args, **kwargs):
        print(args, kwargs)
        super().__init__(*args, **kwargs)
        self.model = OurSAT01(eps, n_rounds)
        self.loss = discounted_step_loss
        self.eps = eps
        self.gamma = gamma
        self.save_hyperparameters()
        
    collate_fn = OurSAT01.collate_fn
    
    def _training_step(self, batch, batch_idx):
        self.model.enc.forward_update.flatten_parameters()
        self.model.enc.backward_update.flatten_parameters()
        epoch = self.current_epoch
        preds = self.model(batch, epoch+1, test=False)
        loss = self.loss(preds, gamma=self.gamma)
        return loss
    

    def _shared_eval_step(self, batch, batch_idx):
        preds = self.model(batch, test=True).detach().cpu()
        loss = self.loss(preds, gamma=self.gamma).item()
        final_preds = preds[-1, :] # we are interested in last pred assignment
        hard_preds = (final_preds.flatten() > 0.5).float()
        labels = batch['is_sat'].cpu()
        assert labels.shape == hard_preds.shape
        acc = accuracy_score(labels, hard_preds).item()
        return acc, loss
    
    
    

class OurSAT03Light(BaseLight):
    """
    OurSAT03 Lightning Implementation
    """
    
    def __init__(self, eps, gamma, beta, tau, n_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = OurSAT03(eps, tau, n_rounds)
        self.loss = discounted_step_loss
        self.eps = eps
        self.gamma = gamma
        self.tau = tau
        self.save_hyperparameters()
        self.v_n_backtracks = []
        self.test_n_backtracks = {}
        
    collate_fn = OurSAT03.collate_fn
    
    def _training_step(self, batch, batch_idx):
        self.model.enc.forward_update.flatten_parameters()
        self.model.enc.backward_update.flatten_parameters()
        epoch = self.current_epoch
        preds, _ = self.model(batch, epoch+1, test=False)
        loss = self.loss(preds)
        return loss
    

    def _shared_eval_step(self, batch, batch_idx, dataloader_idx=None):
        preds, n_backs = self.model(batch, test=True)
        if dataloader_idx == None:
            self.v_n_backtracks.append(n_backs)
        else:
            self.test_n_backtracks.setdefault(dataloader_idx, []).append(n_backs)
        preds = preds.detach().cpu()
        loss = self.loss(preds, gamma=self.gamma).item()
        final_preds = preds[-1, :] # we are interested in last pred assignment
        hard_preds = (final_preds.flatten() > 0.5).float()
        labels = batch['is_sat'].cpu()
        assert labels.shape == hard_preds.shape
        acc = accuracy_score(labels, hard_preds).item()
        return acc, loss
    
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        acc, loss = self._shared_eval_step(batch, batch_idx, dataloader_idx)
        self.test_times.setdefault(dataloader_idx, []).append(time.perf_counter() - self.start_time)

        logs = {f"test_acc/{self.test_set[dataloader_idx]}": acc, f"test_loss/{self.test_set[dataloader_idx]}": loss}
        self.log_dict(logs, prog_bar=True, batch_size=self.batch_size, add_dataloader_idx=False)
        return acc, loss

    
    def on_train_epoch_end(self):
        epoch = self.current_epoch
        max_v = max(map(len, self.v_n_backtracks))
        v_n_backtracks = [torch.nn.functional.pad(x, (0, max_v - len(x)), value=-1) for x in self.v_n_backtracks]
        v = torch.stack(v_n_backtracks)
        print(f"[Epoch {epoch}] num_backtracks | val: {v.mean(dim=0).tolist()}")
        self.v_n_backtracks = []


    
    def on_test_epoch_end(self):
        max_size = {k: max(map(len, v)) for k,v in self.test_n_backtracks.items()}
        test = {k: [torch.nn.functional.pad(x, (0, max_size[k] - len(x)), value=-1) for x in v] for k, v in self.test_n_backtracks.items()}
        test = {k: torch.stack(v).mean(dim=0) for k, v in test.items()}
        print(f"num_backtracks | test:")
        for k, v in test.items():
            print(f"{self.test_set[k]}: {v.tolist()}")
        self.test_n_backtracks = {}
    


class OurSAT04Light(BaseLight):
    """
    OurSAT04 Lightning Implementation
    """
    
    def __init__(self, eps, gamma, beta, tau, n_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = OurSAT04(eps, beta, n_rounds)
        self.loss = discounted_step_loss
        self.eps = eps
        self.gamma = gamma
        self.beta = beta
        self.save_hyperparameters()
        self.t_acc_ahead = []
        self.v_acc_ahead = []
        self.test_acc_ahead = {}
        
    
    collate_fn = OurSAT04.collate_fn
    

    def _training_step(self, batch, batch_idx):
        self.model.enc.forward_update.flatten_parameters()
        self.model.enc.backward_update.flatten_parameters()
        epoch = self.current_epoch
        preds, acc_ahead = self.model(batch, epoch+1, test=False)
        self.t_acc_ahead.append(acc_ahead)
        loss = self.loss(preds)
        return loss
    

    def _shared_eval_step(self, batch, batch_idx, dataloader_idx=None):
        preds, acc_ahead = self.model(batch, test=True)
        if dataloader_idx == None:
            self.v_acc_ahead.append(acc_ahead)
        else:
            self.test_acc_ahead.setdefault(dataloader_idx, []).append(acc_ahead)
        preds = preds.detach().cpu()
        loss = self.loss(preds, gamma=self.gamma).item()
        final_preds = preds[-1, :] # we are interested in last pred assignment
        hard_preds = (final_preds.flatten() > 0.5).float()
        labels = batch['is_sat'].cpu()
        assert labels.shape == hard_preds.shape
        acc = accuracy_score(labels, hard_preds).item()
        return acc, loss
    

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        acc, loss = self._shared_eval_step(batch, batch_idx, dataloader_idx)
        self.test_times.setdefault(dataloader_idx, []).append(time.perf_counter() - self.start_time)

        logs = {f"test_acc/{self.test_set[dataloader_idx]}": acc, f"test_loss/{self.test_set[dataloader_idx]}": loss}
        self.log_dict(logs, prog_bar=True, batch_size=self.batch_size, add_dataloader_idx=False)
        return acc, loss
    
    
    def on_train_epoch_end(self):
        epoch = self.current_epoch
        max_t = max(map(len, self.t_acc_ahead))
        max_v = max(map(len, self.v_acc_ahead))
        t_acc_ahead = [torch.nn.functional.pad(x, (0, max_t - len(x)), value=-1) for x in self.t_acc_ahead]
        v_acc_ahead = [torch.nn.functional.pad(x, (0, max_v - len(x)), value=-1) for x in self.v_acc_ahead]

        mean = lambda x: (x*(x > -1)).sum(dim=0)/(x > -1).sum(dim=0)
        t = torch.stack(t_acc_ahead)
        v = torch.stack(v_acc_ahead)
        print(f"[Epoch {epoch}] Look ahead acc | train_acc: {mean(t).tolist()} | val_acc: {mean(v).tolist()}")
        self.t_acc_ahead, self.v_acc_ahead = [], []
    
    def on_test_epoch_end(self):
        mean = lambda x: (x*(x > -1)).sum(dim=0)/(x > -1).sum(dim=0)
        max_size = {k: max(map(len, v)) for k,v in self.test_acc_ahead.items()}
        test = {k: [torch.nn.functional.pad(x, (0, max_size[k] - len(x)), value=-1) for x in v] for k, v in self.test_acc_ahead.items()}
        test = {k: mean(torch.stack(v)) for k, v in test.items()}
        print(f" Look ahead acc | test_acc:")
        for k, v in test.items():
            print(f"{self.test_set[k]}: {v.tolist()}")
        self.test_acc_ahead = {}


class OurSAT05Light(BaseLight):
    """
    OurSAT05 Lightning Implementation
    """
    
    def __init__(self, eps, gamma, beta, tau, n_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = OurSAT05(eps, beta, n_rounds)
        self.loss = discounted_step_loss
        self.eps = eps
        self.gamma = gamma
        self.beta = beta
        self.save_hyperparameters()
        self.t_acc_ahead = []
        self.v_acc_ahead = []
        self.test_acc_ahead = {}
        
    
    collate_fn = OurSAT05.collate_fn
    

    def _training_step(self, batch, batch_idx):
        self.model.enc.forward_update.flatten_parameters()
        self.model.enc.backward_update.flatten_parameters()
        epoch = self.current_epoch
        preds, acc_ahead = self.model(batch, epoch+1, test=False)
        self.t_acc_ahead.append(acc_ahead)
        loss = self.loss(preds)
        return loss
    

    def _shared_eval_step(self, batch, batch_idx, dataloader_idx=None):
        preds, acc_ahead = self.model(batch, test=True)
        if dataloader_idx == None:
            self.v_acc_ahead.append(acc_ahead)
        else:
            self.test_acc_ahead.setdefault(dataloader_idx, []).append(acc_ahead)
        preds = preds.detach().cpu()
        loss = self.loss(preds, gamma=self.gamma).item()
        final_preds = preds[-1, :] # we are interested in last pred assignment
        hard_preds = (final_preds.flatten() > 0.5).float()
        labels = batch['is_sat'].cpu()
        assert labels.shape == hard_preds.shape
        acc = accuracy_score(labels, hard_preds).item()
        return acc, loss
    

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        acc, loss = self._shared_eval_step(batch, batch_idx, dataloader_idx)
        self.test_times.setdefault(dataloader_idx, []).append(time.perf_counter() - self.start_time)

        logs = {f"test_acc/{self.test_set[dataloader_idx]}": acc, f"test_loss/{self.test_set[dataloader_idx]}": loss}
        self.log_dict(logs, prog_bar=True, batch_size=self.batch_size, add_dataloader_idx=False)
        return acc, loss
    
    
    def on_train_epoch_end(self):
        epoch = self.current_epoch
        max_t = max(map(len, self.t_acc_ahead))
        max_v = max(map(len, self.v_acc_ahead))
        t_acc_ahead = [torch.nn.functional.pad(x, (0, max_t - len(x)), value=-1) for x in self.t_acc_ahead]
        v_acc_ahead = [torch.nn.functional.pad(x, (0, max_v - len(x)), value=-1) for x in self.v_acc_ahead]

        mean = lambda x: (x*(x > -1)).sum(dim=0)/(x > -1).sum(dim=0)
        t = torch.stack(t_acc_ahead)
        v = torch.stack(v_acc_ahead)
        print(f"[Epoch {epoch}] Look ahead acc | train_acc: {mean(t).tolist()} | val_acc: {mean(v).tolist()}")
        self.t_acc_ahead, self.v_acc_ahead = [], []
    
    def on_test_epoch_end(self):
        mean = lambda x: (x*(x > -1)).sum(dim=0)/(x > -1).sum(dim=0)
        max_size = {k: max(map(len, v)) for k,v in self.test_acc_ahead.items()}
        test = {k: [torch.nn.functional.pad(x, (0, max_size[k] - len(x)), value=-1) for x in v] for k, v in self.test_acc_ahead.items()}
        test = {k: mean(torch.stack(v)) for k, v in test.items()}
        print(f" Look ahead acc | test_acc:")
        for k, v in test.items():
            print(f"{self.test_set[k]}: {v.tolist()}")
        self.test_acc_ahead = {}



class OurSAT06Light(BaseLight):
    """
    OurSAT06 Lightning Implementation
    """
    
    def __init__(self, eps, gamma, beta, tau, n_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = OurSAT06(eps, tau, n_rounds)
        self.loss = discounted_step_loss
        self.eps = eps
        self.gamma = gamma
        self.tau = tau
        self.save_hyperparameters()
        self.v_n_backtracks = []
        self.test_n_backtracks = {}
        
    collate_fn = OurSAT06.collate_fn
    
    def _training_step(self, batch, batch_idx):
        self.model.enc.forward_update.flatten_parameters()
        self.model.enc.backward_update.flatten_parameters()
        epoch = self.current_epoch
        preds, _ = self.model(batch, epoch+1, test=False)
        loss = self.loss(preds)
        return loss
    

    def _shared_eval_step(self, batch, batch_idx, dataloader_idx=None):
        preds, n_backs = self.model(batch, test=True)
        if dataloader_idx == None:
            self.v_n_backtracks.append(n_backs)
        else:
            self.test_n_backtracks.setdefault(dataloader_idx, []).append(n_backs)
        preds = preds.detach().cpu()
        loss = self.loss(preds, gamma=self.gamma).item()
        final_preds = preds[-1, :] # we are interested in last pred assignment
        hard_preds = (final_preds.flatten() > 0.5).float()
        labels = batch['is_sat'].cpu()
        assert labels.shape == hard_preds.shape
        acc = accuracy_score(labels, hard_preds).item()
        return acc, loss
    
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        acc, loss = self._shared_eval_step(batch, batch_idx, dataloader_idx)

        self.test_times.setdefault(dataloader_idx, []).append(time.perf_counter() - self.start_time)
        
        logs = {f"test_acc/{self.test_set[dataloader_idx]}": acc, f"test_loss/{self.test_set[dataloader_idx]}": loss}
        self.log_dict(logs, prog_bar=True, batch_size=self.batch_size, add_dataloader_idx=False)
        return acc, loss

    
    def on_train_epoch_end(self):
        epoch = self.current_epoch
        max_v = max(map(len, self.v_n_backtracks))
        v_n_backtracks = [torch.nn.functional.pad(x, (0, max_v - len(x)), value=-1) for x in self.v_n_backtracks]
        v = torch.stack(v_n_backtracks)
        print(f"[Epoch {epoch}] num_backtracks | val: {v.mean(dim=0).tolist()}")
        self.v_n_backtracks = []


    
    def on_test_epoch_end(self):
        max_size = {k: max(map(len, v)) for k,v in self.test_n_backtracks.items()}
        test = {k: [torch.nn.functional.pad(x, (0, max_size[k] - len(x)), value=-1) for x in v] for k, v in self.test_n_backtracks.items()}
        test = {k: torch.stack(v).mean(dim=0) for k, v in test.items()}
        print(f"num_backtracks | test:")
        for k, v in test.items():
            print(f"{self.test_set[k]}: {v.tolist()}")
        self.test_n_backtracks = {}
