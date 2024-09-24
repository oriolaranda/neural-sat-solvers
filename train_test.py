import json
import os
import torch
import wandb
import random
import argparse
import warnings
import copy
import numpy as np
import torch.nn as nn
import seaborn as sns
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product
from sklearn.metrics import accuracy_score
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.profilers import PyTorchProfiler, SimpleProfiler


from data import SR_3_10, SR_10_40, U_4_100, M_4_100, OOD_Test
from models.baselines import NeuroSATLight, CircuitSATLight
from models import OurSAT00Light, OurSAT01Light, OurSAT02Light, OurSAT03Light, OurSAT04Light, OurSAT05Light, OurSAT06Light


class PrintCallback(pl.Callback):
    """ Print callback to print on each epoch during trianing """

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        print(f"[Epoch {epoch}] loss/train: {metrics['train_loss']} | loss/val: {metrics['val_loss']} | acc/val: {metrics['val_acc']}")

    
class MetricsCallback(pl.Callback):
    """Metric callback to store in disk the results"""

    def __init__(self):
        super().__init__()
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = copy.deepcopy(trainer.callback_metrics)
        for k, v in metrics.items():
            if "n_back" not in k:
                self.metrics[k].append(v.detach().item())
    



def plot_results(train_loss, val_loss, val_acc, path):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = ['Roman']
    epochs = range(1, len(train_loss) + 1)
    
    # Plotting Loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(axis='y')

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc, label='val_acc')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('% solved instances')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()

    plt.savefig(path, format='png')
    plt.show()



def run(args):
    
    # Reproducebility
    pl.seed_everything(args.seed, workers=True)
    rnd_g = torch.Generator().manual_seed(args.seed)

    # Experiment / Run ID and arguments
    overfit = f"-overfit-{args.overfit}" if args.overfit else ""
    project_name = f"{args.model}-{args.dataset}{overfit}" if not args.name else args.name
    checkpoint_path = "logs"

    hash_id = hash(v for v in vars(args).values())

    args.run_id += f"-{args.eps:.1}-{args.lr}-{hash_id}"

    print("EXPERIMENT ID:", project_name, args.run_id, "| seed:", args.seed)
    for k, v in vars(args).items():
        print(f"{k}={v}")
    
    ##########
    # MODELS #
    ##########
    models = {
        'csat': {'class': CircuitSATLight, 'args': {'sat_only': True,'solution': False}},
        'nsat': {'class': NeuroSATLight, 'args': {'sat_only': False,'solution': False}},
        'ours00': {'class': OurSAT00Light, 'args': {'sat_only': True,'solution': True}},
        'ours01': {'class': OurSAT01Light, 'args': {'sat_only': True,'solution': False}},
        'ours02': {'class': OurSAT02Light, 'args': {'sat_only': True,'solution': False}},
        'ours03': {'class': OurSAT03Light, 'args': {'sat_only': True,'solution': False}},
        'ours04': {'class': OurSAT04Light, 'args': {'sat_only': True,'solution': False}},
        'ours05': {'class': OurSAT05Light, 'args': {'sat_only': True,'solution': False}},
        'ours06': {'class': OurSAT06Light, 'args': {'sat_only': True,'solution': False}},

    }
    assert args.model in models.keys(), f'"{args.model}": Model not Implemented!'
    model_class = models[args.model]['class']
    data_args = models[args.model]['args']
    
    ################
    # DATA MODULES #
    ################
    datasets = {
        'sr-3-10': SR_3_10,
        'sr-10-40': SR_10_40,
        'u-4-100': U_4_100,
        'm-4-100': M_4_100,
        'ood-test': OOD_Test
    }
    assert args.dataset in datasets.keys(), f'"{args.dataset}": Data Module not Implemented!'
    dataset = datasets[args.dataset]
    data_module = dataset(collate_fn=model_class.collate_fn, num_workers=args.num_workers, batch_size=args.batch_size, 
                          overfit=args.overfit, rnd_g=rnd_g, **data_args)
    
    # Define Model
    if args.model in models.keys() - {'csat', 'nsat', 'ours00'}:
        model = model_class(lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, n_rounds=args.n_rounds,
                            eps=args.eps, gamma=args.gamma, beta=args.beta, tau=args.tau, dataset_names=data_module.names)
    else:
        model = model_class(lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, n_rounds=args.n_rounds, 
                            eps=args.eps, dataset_names=data_module.names)

    # Loggers and Trainer
    wandb_enabled, wandb_logger = not (args.debug or args.test), True
    

    if wandb_enabled:
        wandb_logger = WandbLogger(name=args.run_id, save_dir=checkpoint_path, project=project_name, checkpoint_name="last", id=args.run_id)
        # wandb_logger = TensorBoardLogger("tb_logs", name=args.run_id)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath="logs/trained_models", monitor=f"val_acc", mode='max', 
                                          filename=project_name+"-"+args.run_id+"-{epoch:02d}-{val_acc:.2f}")
    
    print_callback = PrintCallback()
    metrics_callback = MetricsCallback()
    callbacks = [lr_monitor, print_callback, metrics_callback]
    if not args.test:
        callbacks += [checkpoint_callback]


    profiler = SimpleProfiler(".", filename="profiler-logs-ours03")
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        trace_memory=True,
        schedule=torch.profiler.schedule(skip_first=10, wait=1,  warmup=1, active=20)
    )

    trainer = pl.Trainer(accelerator='gpu', max_epochs=args.epochs, gradient_clip_val=args.grad_clip_val, logger=wandb_logger,
                         callbacks=callbacks, deterministic=False, enable_progress_bar=args.debug, default_root_dir="logs",
                         limit_train_batches=args.limit, limit_val_batches=args.limit, limit_test_batches=args.limit,
                         precision="16-mixed",
                         # detect_anomaly=True, profiler=profiler
                         )
    
    
    # Training
    if not args.test:
        print("Train starting...")
        trainer.fit(model=model, datamodule=data_module)
        print("Testing with model val_acc:", checkpoint_callback.best_model_score.item(), "| path:", checkpoint_callback.best_model_path)
        best_model_dir = checkpoint_callback.best_model_path
    
    # Load Pre-Trained Model for only Testing
    if args.test:
        file_name = [file for file in os.listdir(f'inf/{project_name}') if file.endswith(".ckpt")][0]
        best_model_dir = f'inf/{project_name}/{file_name}'
            
    # Test
    print("Test starting...")
    test_metrics = trainer.test(model=model, ckpt_path=best_model_dir, datamodule=data_module)
    
    # Saving metrics
    if wandb_enabled:
        wandb_logger.experiment.finish()
    
    if not args.debug:
        metrics_path = f"inf/{project_name}/"
        
        if not args.test:
            train_metrics = metrics_callback.metrics
            plot_results(**train_metrics, path=os.path.join(metrics_path, f'train_metrics-{args.lr}-{args.eps}-{args.seed}.png'))

            with open(os.path.join(metrics_path, f'train_metrics-{args.lr}-{args.eps}-{args.gamma}-{args.tau}-{args.seed}.json'), 'w') as f:
                json.dump(train_metrics, f)

        with open(os.path.join(metrics_path, f'test_metrics-{args.lr}-{args.eps}-{args.gamma}-{args.tau}-{args.n_rounds}-{args.seed}.json'), 'w') as f:
            json.dump(test_metrics, f)
    

def main(args):
    # DEFAULT HYPERPARAMETERS
    args.weight_decay = 1e-10
    args.grad_clip_val = 0.65
    args.run_id = datetime.now().strftime("%d-%m--%H-%M") + "_" + str(args.seed)
    
    # DEBUG
    if args.debug:
        args.epochs = 2
        args.limit = 5
        args.run_id = "debug"

    # OVERFITTING
    if args.overfit:
        args.epochs = 5000
        args.limit = 1
        args.batch_size = args.overfit
    
    run(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training and testing on CircuitSAT/NeuroSAT')
    parser.add_argument('-n', '--name', help="Project name", default="")
    parser.add_argument('-m', '--model', help='Model to use', default="csat")
    parser.add_argument('-d', '--dataset', help="Dataset to use", default="sr-3-10")
    parser.add_argument('-s', '--seed', help='Seed to use', default=12345, type=int)
    parser.add_argument('-t', '--test', help='Only test mode', action='store_true', default=False)
    parser.add_argument('-l', '--limit', help='Use limited data', default=None, type=int)
    parser.add_argument('-nw', '--num-workers', help="Num of Workers", default=4, type=int)
    parser.add_argument('-e', '--epochs', help="Number of epochs", default=30, type=int)
    parser.add_argument('-bs', '--batch-size', help="Batch size", default=32, type=int)
    parser.add_argument('--lr', help="Learning Rate", default=0.00002, type=float)
    parser.add_argument('-nr', '--n-rounds', help="Num of Rounds", default=40, type=int)
    parser.add_argument('--eps', help="Epsilon", default=1.2, type=float)
    parser.add_argument('--beta', help="Beta", default=0.9, type=float)
    parser.add_argument('--gamma', help="Gamma (discounted loss)", default=1.0, type=float)
    parser.add_argument('--tau', help="Tau (ood detection)", default=0.1, type=float)
    parser.add_argument('--debug', help='Debug mode', action='store_true', default=False)
    parser.add_argument('--overfit', help='Overfit n samples', default=None, type=int)
    args = parser.parse_args()
    
    if not args.debug:
        warnings.filterwarnings("ignore", category=UserWarning)

    main(args)