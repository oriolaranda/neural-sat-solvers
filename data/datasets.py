import copy
import os
import json
import numpy as np
import lightning.pytorch as pl
from pysat.formula import CNF
from torch.utils.data import Dataset, DataLoader, random_split


def get_filter_dimacs(sat_only):
    """
    Filter Dimacs files optionally just SAT instances
    """
    def _filter(file):
        sat = 'sat=1' if sat_only else ''
        return file.endswith(".DIMACS") and (sat in file)
    return _filter


class SATDimacs(Dataset):
    """
    Dataset class for SAT data written in DIMACS files.
    With the purpose of unifying the datasets.

    @param path: dataset path DIMACS files
    @param sat_only: include only sat instances
    @param labeled: include the solution in the instances
    """
    def __init__(self, path: str, sat_only: bool, solution: bool, limit: int = None):
        self.path = path
        self.solution = solution
        filter_func = get_filter_dimacs(sat_only)
        self.files = list(os.path.join(path, file) for file in os.listdir(path) if filter_func(file))[:limit]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Get cnf
        file = self.files[idx]
        cnf = CNF(from_file=file)
        
        # Get solution and satisfiability
        is_sat = 'sat=1' in file
        solution = -1
        if is_sat and self.solution:
            sol_file = file.split("sat=1")[0]+"sol.json"
            with open(sol_file, 'r') as f:
                solution = json.load(f)
        
        sample = {'file': file, 'cnf': cnf, 'is_sat': is_sat, 'solution': solution}
        return sample


class SATDataLoader(pl.LightningDataModule):
    """
    LightningDataModule custom class for SAT
    """
    def __init__(self, collate_fn: callable, batch_size: int, sat_only: bool = False, 
                 solution: bool = False, overfit: int = None, rnd_g=None, num_workers: int = 4):
        super().__init__()
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.sat_only = sat_only
        self.solution = solution
        self.overfit = overfit
        self.rnd_g = rnd_g
        self.train_data_dir = "/nfs/students/aor/datasets"
        self.test_data_dir = "/nfs/students/aor/datasets"
    
    def setup(self, stage: str):
        if stage == "fit" or stage == "validate":
            # Define the training set
            dataset_path = os.path.join(self.train_data_dir, self.dataset_names['train'])
            dataset = SATDimacs(dataset_path, self.sat_only, self.solution, self.overfit)
            if self.overfit:
                # Use the same data as validation
                self.train_set = self.valid_set = dataset
            else:
                # Split the train set into two
                train_set_size = int(len(dataset) * 0.8)
                valid_set_size = len(dataset) - train_set_size
                self.train_set, self.valid_set = random_split(dataset, [train_set_size, valid_set_size], generator=self.rnd_g)

        if stage == "test":
            if self.overfit:
                # Use the same data as train and val
                self.test_sets = [self.train_set]
                self.dataset_names['test'] = self.dataset_names['train']
            else:
                # Define the test sets
                self.test_sets = []
                for dataset_name in self.dataset_names['test']:
                    dataset_path = os.path.join(self.test_data_dir, dataset_name)
                    self.test_sets.append(SATDimacs(dataset_path, sat_only=True, solution=self.solution, limit=500))

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_set, collate_fn=self.collate_fn, pin_memory=True,
                                  shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(dataset=self.valid_set, collate_fn=self.collate_fn, pin_memory=False,
                                  shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
        return valid_loader

    def test_dataloader(self):
        test_loaders = []
        for test_set in self.test_sets:
            test_loader = DataLoader(dataset=test_set, collate_fn=self.collate_fn, pin_memory=False,
                                     shuffle=False, batch_size=32, num_workers=self.num_workers, drop_last=True)
            test_loaders.append(test_loader)
        return test_loaders

    @property
    def names(self):
        return self.dataset_names


class U_4_100(SATDataLoader):
    """
    Dataset U-4-100
    - training: uniform k-SAT-4-100, with k in [2,10] and alpha in [2, 10]
    - test: 4-SAT-100 with alpha in [7, 10]
    """
    def __init__(self, collate_fn: callable, *args, **kwargs):
        super().__init__(collate_fn, *args, **kwargs)
        self.dataset_names = {
            "train": "U-4-100",
            "valid": "U-4-100",
            'test': [f"4-SAT-100_{i}_{a}" for i, a in enumerate(np.arange(7, 10.5, 0.5))]
        }
        self.test_data_dir = "/nfs/students/aor/datasets/test/uniform"



class M_4_100(SATDataLoader):
    """
    Dataset M-4-100
    - training: modular k-SAT-4-100, with k in [2,10], alpha in [2, 10], c in [10, 20]
    - test: 200 samples for alpha, 4-SAT-4-100 with alpha in [7, 10]
    """
    def __init__(self, collate_fn: callable, *args, **kwargs):
        super().__init__(collate_fn, *args, **kwargs)
        self.dataset_names = {
            "train": "M-4-100",
            "valid": "M-4-100",
            'test': [f"M-4-100_{i}_{a}" for i, a in enumerate(np.arange(7, 10.5, 0.5))]
        }
        self.test_data_dir = "/nfs/students/aor/datasets/test/modular"


class SR_3_10(SATDataLoader):
    """
    Dataset SR-3-10
    - training: SR-3-10 [ SR(U(3, 10)) (e.i. from NeuroSAT paper) ]
    - test: [SR-20, SR-40, SR-60, SR-80]
    """

    def __init__(self, collate_fn: callable, *args, **kwargs):
        super().__init__(collate_fn, *args, **kwargs)
        self.dataset_names = {
            'train': "SR-3-10",
            'valid': "SR-3-10",
            'test': ["SR-20", "SR-40", "SR-60", "SR-80"]
        }


class SR_10_40(SATDataLoader):
    """
    Dataset SR-10-40
    - training: SR-10-40 [ SR(U(10, 40)) (e.i. from NeuroSAT paper) ]
    - test: [SR-60, SR-80, SR-100]
    """

    def __init__(self, collate_fn: callable, *args, **kwargs):
        super().__init__(collate_fn, *args, **kwargs)
        self.dataset_names = {
            'train': "SR-10-40",
            'valid': "SR-10-40",
            'test': ["SR-20", "SR-40", "SR-60", "SR-80"]
        }


class OOD_Test(SATDataLoader):
    """
    Dataset SR-10-40
    - test: [SR-20, SR-40, SR-60, SR-80, SR-100, SR-200, U-4-100, M-4-100]
    """

    def __init__(self, collate_fn: callable, *args, **kwargs):
        super().__init__(collate_fn, *args, **kwargs)
        self.dataset_names = {
            'test': ["SR-20", "SR-40", "SR-60", "SR-80", "SR-100", "SR-120", "U-4-100", "M-4-100"]
        }