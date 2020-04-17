import functools
import logging
import os
import random
import typing as typ
from dataclasses import dataclass
from matplotlib import pyplot as plt

import numpy as np
import torch as T
import gzip

from .datasets import StitchDataset, create_data_split, create_dataloader
from .model import (CovidModel, create_protein_model, run_model)
from .modules.chemistry import MPNEncoder
from .schedulers import LinearWarmupScheduler
from .reporting import get_performance_plots, calculate_average_loss_and_accuracy
from .utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def initialize_logger(run_name,
                      output_dir='./logs', 
                      print_lvl = logging.INFO, 
                      write_lvl = logging.DEBUG):
    if not os.path.exists("./logs"):
        os.mkdir("./logs")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
     
    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(print_lvl)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(
        output_dir, 
        f"{run_name}.log"
    ), "w")
    handler.setLevel(write_lvl)
    formatter = logging.Formatter("%(asctime)s | [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclass
class CovidTrainingConfiguration():
    # Global config
    random_seed: int = 4
    batch_size: int = 24
    training_fold: typ.Union[int, None] = 0
    max_epochs: int = 100
    validation_frequency: float = 0.2

    create_checkpoints: bool = True

    device: str = 'cuda'

    # Dataset Configuration
    synthetic_negative_rate: float = 0.2
    dataloader_num_workers: int = 1

    # Optimizer Configuration
    optim_initial_lr: float = 1e-3
    optim_adam_betas: typ.Tuple[int, int] = (0.9, 0.999)
    optim_adam_eps: float = 0.01
    optim_warmup_override: typ.Union[None, int] = None # Applies 2/(1-Beta2) by default

    # LR Scheduler Configuration
    optim_scheduler_factor: float = 0.1**0.125
    optim_scheduler_patience: int = 4
    optim_minimum_lr: float = 1e-7

    # Model hyperparameters
    chem_layers_per_message: int = 2
    chem_hidden_size: int = 300

    dropout_rate: float = 0.4

    init_normal: bool = True


def _set_random_seeds(seed = 4):
    logging.info(f"Random seeds set: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    T.manual_seed(seed)


def _create_all_data_splits():
    _set_random_seeds(4) # Always split data with consistent random seed
    logging.info("Creating data splits -- global training/holdout")
    create_data_split('./data', './data/training', './data/final_holdout')
    
    for i in range(10):
        logging.info(f"Creating data splits -- train/validation {i:02}")
        create_data_split('./data/training', f'./data/train_{i:02}', f'./data/valid_{i:02}')


def _create_model(config):
    logging.info("Creating model")
    chem_model = MPNEncoder(
        layers_per_message=config.chem_layers_per_message, 
        hidden_size=config.chem_hidden_size,
        dropout=config.dropout_rate
    )
    protein_model = create_protein_model(dropout=config.dropout_rate)
    model = CovidModel(chem_model, protein_model, dropout=config.dropout_rate)
    return model


def _create_optimizer_and_schedulers(model, config):
    logging.info("Initializing optimizers/schedulers")

    optim = T.optim.Adam(
        model.parameters(), 
        lr=config.optim_initial_lr, 
        betas=config.optim_adam_betas,
        eps=config.optim_adam_eps
    )

    warmup = LinearWarmupScheduler(
        optim, 
        config.optim_warmup_override if config.optim_warmup_override is not None else 2/(1-config.optim_adam_betas[1])
    )
    scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(
        optim, 
        factor=config.optim_scheduler_factor, 
        patience=config.optim_scheduler_patience,
        min_lr=config.optim_minimum_lr
    )

    return optim, warmup, scheduler


def _create_dataloaders(config):
    logging.info("Initializing Datasets")
    data = StitchDataset(f'./data/train_{config.training_fold:02}')
    dataloader = create_dataloader(
        data, 
        config.batch_size, 
        drop_last=True,
        neg_rate = config.synthetic_negative_rate, 
        num_workers = config.dataloader_num_workers
    )

    validation_data = StitchDataset(f'./data/valid_{config.training_fold:02}')
    validation_dataloader = create_dataloader(
        validation_data, 
        config.batch_size, 
        drop_last=True,
        neg_rate=0.0, 
        num_workers=config.dataloader_num_workers
    )

    return dataloader, validation_dataloader


def train_model(config:CovidTrainingConfiguration, 
                disable_training_resume:bool=False,
                run_name:str=None,
                check_interrupted:typ.Callable=None,
                disable_checkpointing:bool=False):
    
    if run_name is None:
        run_name = f"train_fold{config.training_fold:02}" if config.training_fold is not None else "train_global"

    initialize_logger(run_name)

    logging.info(f"Training initializing -- {run_name}")
    
    if config.training_fold is None:
        logging.warning("Note: You are training this model on the full training data!")

    if not os.path.exists('./data/training'):
        # Create data splits if they haven't been computed
        if config.training_fold is not None and config.training_fold != 0:
            raise Exception("Cannot generate trait/test splits with non-zero training fold specified")

        _create_all_data_splits()

    # Set random seeds for reproducibility
    _set_random_seeds(config.random_seed)
    
    # Create and initialize model
    model = _create_model(config)
    
    logging.info("Pushing model to device")
    model.to(config.device)

    # Create dataloaders
    dataloader, validation_dataloader = _create_dataloaders(config)

    # Create the optimizer & schedulers
    optim, warmup, scheduler = _create_optimizer_and_schedulers(model, config)

    training_state_path = f"./training_state/{run_name}__state.pkl"
    
    losses = []
    validation_stats = []

    get_validation_loss = functools.partial(
        calculate_average_loss_and_accuracy, 
        model, 
        validation_dataloader,
        config.device
    )

    epoch = 0
    losses = []
    validation_stats = []
    last_validation = 0

    if os.path.exists(training_state_path) and not disable_training_resume:
        logging.info("Loading previous training state")
        state = T.load(training_state_path, map_location=config.device)
            
        epoch = state.get('epoch', epoch-1) + 1
        losses = state.get('losses', losses)
        validation_stats = state.get('validation_stats', validation_stats)
        last_validation = state.get('last_validation', last_validation)
        model.load_state_dict(state['model'])
        optim.load_state_dict(state['optim'])
        warmup.load_state_dict(state['warmup'])
        if 'scheduler' in state:
            scheduler.load_state_dict(state['scheduler'])
    elif disable_training_resume:
        logging.info("Resuming from previous training state manually disabled")
    else:
        logging.info("No previous training state to load")

    # make required subfolders
    for folder in ['outputs', 'models', 'checkpoints', 'training_state']:
        if not os.path.exists(folder):
            os.mkdir("./" + folder)

    if epoch == 0:
        vloss, vacc, v_conf, v_outputs = get_validation_loss()
        v_outputs.to_csv(f"./outputs/{run_name}_epoch00_validation_result.csv.gz", index=False)
        validation_stats.append([0, vloss, vacc, v_conf])

    epoch_length = len(dataloader)
    interrupted = False

    for epoch in tqdm(range(epoch, 100)):
        logging.info(f"Beginning epoch {epoch}")
        
        pct_epoch = 0
        model.train()

        for idx, batch in enumerate(tqdm(dataloader, leave=False)):

            model.zero_grad()
            _, _, loss, _ = run_model(model, batch, config.device)

            loss.backward()

            optim.step()
            warmup.step()
                
            pct_epoch = min(1.0, (idx+1)/epoch_length)
            
            losses.append((epoch + pct_epoch, loss.item()))
        
            if pct_epoch == 1.0 or epoch + pct_epoch - last_validation > config.validation_frequency:
                logging.info("Generating validation stats")

                vloss, vacc, v_conf, v_output = get_validation_loss()
                model.train()
                v_outputs.to_csv(f"./outputs/{run_name}_epoch{epoch:02}_validation_result.csv.gz", index=False)

                validation_stats.append([epoch+pct_epoch, vloss, vacc, v_conf])
                
                scheduler.step(vloss)

                fig = get_performance_plots(losses, validation_stats)
                fig_path = f'./outputs/performance_{run_name}.png'
                fig.savefig(fig_path)
                plt.close(fig)
                logging.info(f'Generated validation stats -- plot saved to "{fig_path}"')
                last_validation = epoch + pct_epoch

            if check_interrupted is not None and check_interrupted():
                interrupted = True
                break

        if interrupted:
            logging.info("user interrupt received -- quitting")
            break

        state = {
            'epoch': epoch,
            'losses': losses,
            'validation_stats': validation_stats,
            'last_validation': last_validation,
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'warmup': warmup.state_dict(),
            'scheduler': scheduler.state_dict()
        }

        if not disable_checkpointing:
            logging.info('Saving checkpoint')
            with gzip.open(f'./checkpoints/model_{run_name}_{epoch:03}.pkl.gz', 'wb') as f:
                T.save(state, f)
            
        logging.info('Saving state')
        T.save(state, f"./training_state/{run_name}__state.pkl")
    
    return losses, validation_stats