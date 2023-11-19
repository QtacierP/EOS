import glob
import logging
import os
import torch
from torch import nn
import numpy as np
import wandb
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning import seed_everything, Trainer
import hydra
from omegaconf import DictConfig, OmegaConf, MISSING
from typing import Dict, Any, List
from torchvision import transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters         
from scipy.sparse.linalg import LinearOperator, eigsh   
import time
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Tuple

log = logging.getLogger(__name__)




def get_transform(transform_list):
    transform = []
    for transform_item in transform_list:
        transform_name = transform_item[0]
        if len(transform_item) > 1:
            transform_args = transform_item[1:]
            transform_func  = getattr(transforms, transform_name)(*transform_args)
        else:
            transform_args = None
            transform_func  = getattr(transforms, transform_name)()
        transform.append(transform_func)
    return transforms.Compose(transform)



def setup_training(config):
    if "seed" in config:
        seed_everything(config.seed)
    if config.debug:
        log.info(f"Running in debug mode! <{config.debug}>")
        config.num_workers = 0
        config.trainer.fast_dev_run = True
        if 'precision' in config.trainer:
            config.trainer.precision = 32
        os.environ['WANDB_MODE'] = 'dryrun'

def init_loggers(config, loggers=None):
    if loggers is None:
        loggers = []
    else:
        assert isinstance(loggers, list)
    if config.logger is None:
        return loggers
    for _, lg_conf in config.logger.items():
        if lg_conf is None:
            continue
        elif "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))
        else:
            continue
    return loggers


def init_callbacks(config, callbacks=None):
    if callbacks is None:
        callbacks = []
    else:
        assert isinstance(callbacks, list)
    if config.callbacks is None:
        return callbacks
    for _, cb_conf in config.callbacks.items():
        if cb_conf is None:
            continue
        elif "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
        else:
            raise ValueError(cb_conf)
    return callbacks


def get_wandb_logger(trainer: pl.Trainer) -> WandbLogger:
    logger = None
    for lg in trainer.logger:
        if isinstance(lg, WandbLogger):
            logger = lg
    if not logger:
        log.warn(
            "You are using wandb related callback,"
            "but WandbLogger was not found for some reason..."
        )
    return logger


def init_eos_experiment(config):
    from eos import EOSExperiment
    log.info(f"Instantiating EOS experiment <{config.experiment_name}>")
    experiment = EOSExperiment(config)
    return experiment


def init_trainer(model, config, callbacks=None, logger=None):
    if callbacks is None:
        callbacks = init_callbacks(config)
    else:
        assert isinstance(callbacks, list)
    logger = init_loggers(config, logger)
    log.info(f"Instantiating Trainer ")
    trainer = Trainer(callbacks=callbacks, logger=logger, **config.trainer)
    log_hyperparameters(config, model, trainer, logger)
    return  trainer


def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        trainer: pl.Trainer,
        logger: List[pl.loggers.LightningLoggerBase],
):
    hparams = OmegaConf.to_container(config)
    # save number of model parameters
    params = {}
    params["params_total"] = sum(p.numel() for p in model.parameters())
    params["params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    params["params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    hparams['params'] = params

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just to prevent trainer logging hparams of model as we manage it ourselves)
    for lg in logger:
        lg.log_hyperparams = lambda x: None


@ rank_zero_only
def finish_run(trainer):
    logger = None
    for lg in trainer.logger:
        if isinstance(lg, WandbLogger):
            logger = lg
    if logger is not None:
        path = logger.experiment.path
        dir = logger.experiment.dir
        logger.experiment.finish()
        run_api = wandb.Api().run(path)
        return run_api, dir
    else:
        return None, None


class UploadCodeToWandbAsArtifact(Callback):
    """Upload all *.py files to wandb as an artifact at the beginning of the run."""

    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    @ rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")
        for path in glob.glob(os.path.join(self.code_dir, "**/*.py"), recursive=True):
            code.add_file(path)

        experiment.use_artifact(code)


class UploadConfigToWandbAsArtifact(Callback):
    def __init__(self, config_dir: str):
        self.config_dir = config_dir

    @ rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        run_config = wandb.Artifact("run-config", type="config")
        for path in glob.glob(os.path.join(self.config_dir, "*.yaml")):
            log.info(f'Uploading config {path}')
            run_config.add_file(path)

        experiment.use_artifact(run_config)


class UploadCheckpointsToWandbAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of training."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @ rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(
                os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True
            ):
                ckpts.add_file(path)

        experiment.use_artifact(ckpts)


class SaveBestCheckpointPathToWandbSummary(Callback):
    def __init__(self, prefix: str = None):
        self.prefix = '' if prefix is None else prefix + '_'

    @ rank_zero_only
    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        logger = get_wandb_logger(trainer=trainer)
        logger.experiment.summary[self.prefix + 'best_checkpoint'] = trainer.checkpoint_callback.best_model_path


class WatchModelWithWandb(Callback):
    """Make WandbLogger watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @ rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


def iterate_dataset(dataset: Dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (batch_X, batch_y) in loader:
        yield batch_X, batch_y


def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                dataset, batch_size, device, vector):
    """Compute a Hessian-vector product."""
    p = len(parameters_to_vector(network.parameters()))

    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.to(device)
    start = time.time()

    for (x, y) in iterate_dataset(dataset, batch_size):
        x = x.to(device)
        y = y.to(device)
        loss = loss_fn(network(x), y) 
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    
    end = time.time()
    return hvp


def lanczos(matrix_vector, dim: int, neigs: int, device='cuda'):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).to(device)
        return matrix_vector(gpu_vec)
    start = time.time()
    operator = LinearOperator((dim, dim), matvec=mv)
    end = time.time()
    start = time.time()
    evals, evecs = eigsh(operator, neigs)
    end = time.time()

    return np.ascontiguousarray(evals[::-1]).copy().astype(np.float32), \
           np.ascontiguousarray(np.flip(evecs, -1)).copy().astype(np.float32)


def get_hessian_eigenvalues(network: nn.Module, loss_fn, dataset,
                            neigs=2, batch_size=128, device='cuda'):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(network, loss_fn,dataset,batch_size, device,
                                          delta).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs, device=device)
    return evals

def plot_acc(acc_history, title, save_path):
    plt.figure()
    plt.plot(acc_history)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig(save_path)
    
def plot_loss(loss_history, title, save_path):
    plt.figure()
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.savefig(save_path)



def plot_sharpness(gd_sharpness, save_path, eign_freq, gd_lr_line):
    plt.scatter(np.arange(len(gd_sharpness)) * eign_freq, gd_sharpness, s=5)
    # plot line across scatter plot
    plt.plot(np.arange(len(gd_sharpness)) * eign_freq, gd_sharpness, linestyle='-')
    #plt.axhline(2. / lr, linestyle='dotted')
    # plot a horizontal line across lying on 2/gd_lr
    if gd_lr_line is not None:
        plt.plot(np.arange(len(gd_sharpness)) * eign_freq, np.ones(len(gd_sharpness)) * gd_lr_line, linestyle='dashed', color='red') 
    plt.title("sharpness")
    plt.xlabel("iteration") 
    plt.savefig(save_path)


class FastTensorDataset(Dataset[Tuple[Tensor, ...]]):
    def __init__(self, x, y):
        assert x.size(0) == y.size(0), "x and y must have the same size of the first dimension."
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(0)
    
    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self


def compute_gradient(network: nn.Module, loss_fn: nn.Module,
                     dataset: Dataset, physical_batch_size: int = 5000, device='cuda'):
    """ Compute the gradient of the loss function at the current network parameters. """
    p = len(parameters_to_vector(network.parameters()))
    average_gradient = torch.zeros(p).to(device)
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        X = X.to(device)
        y = y.to(device)
        batch_loss = loss_fn(network(X), y) / len(dataset)
        batch_gradient = parameters_to_vector(torch.autograd.grad(batch_loss, inputs=network.parameters()))
        average_gradient += batch_gradient
    return average_gradient

class AtParams(object):
    """ Within a with block, install a new set of parameters into a network.

    Usage:

        # suppose the network has parameter vector old_params
        with AtParams(network, new_params):
            # now network has parameter vector new_params
            do_stuff()
        # now the network once again has parameter vector new_params
    """

    def __init__(self, network: nn.Module, new_params: Tensor):
        self.network = network
        self.new_params = new_params

    def __enter__(self):
        self.stash = parameters_to_vector(self.network.parameters())
        vector_to_parameters(self.new_params, self.network.parameters())

    def __exit__(self, type, value, traceback):
        vector_to_parameters(self.stash, self.network.parameters())


def rk_step(network: nn.Module, loss_fn: nn.Module, dataset: Dataset, step_size: float,
            physical_batch_size=5000, device='cuda'):
    """ Take a Runge-Kutta step with a given step size. """
    theta = parameters_to_vector(network.parameters())

    def f(x: torch.Tensor):
        with AtParams(network, x):
            fx = - compute_gradient(network, loss_fn, dataset, physical_batch_size=physical_batch_size, device=device)
        return fx

    k1 = f(theta)
    k2 = f(theta + (step_size / 2) * k1)
    k3 = f(theta + (step_size / 2) * k2)
    k4 = f(theta + step_size * k3)

    theta_next = theta + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    vector_to_parameters(theta_next, network.parameters())


def rk_advance_time(network: nn.Module, loss_fn: nn.Module, dataset: Dataset, T: float,
                    rk_step_size: float, physical_batch_size: int, device='cuda'):
    """ Using the Runge-Kutta algorithm, numerically integrate the gradient flow ODE for time T, using a given
     Runge-Kutta step size."""
    T_remaining = T
    while T_remaining > 0:
        this_step_size = min(rk_step_size, T_remaining)
        rk_step(network, loss_fn, dataset, this_step_size, physical_batch_size, device)
        T_remaining -= rk_step_size