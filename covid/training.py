import functools
import logging
import os
import random
import typing as typ
from dataclasses import dataclass
import dataclasses as dc
from matplotlib import pyplot as plt

import numpy as np
import torch as T
import gzip

from .datasets import StitchDataset, create_data_split, create_dataloader
from .model import (CovidModel, run_model)
from .modules.chemistry import MPNEncoder
from .schedulers import LinearWarmupScheduler
from .optim import LocallyCyclicAdam
from .reporting import get_performance_plots, calculate_average_loss_and_accuracy
from .utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = logging.getLogger(__name__)


def initialize_logger(run_name,
                      output_dir='./logs', 
                      print_lvl = logging.INFO, 
                      write_lvl = logging.DEBUG):
    if not os.path.exists("./logs"):
        os.mkdir("./logs")

    logging.getLogger('covid')
    logger.setLevel(logging.DEBUG)
     
    handlers = []

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(print_lvl)
    formatter = logging.Formatter(f"{run_name} | [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handlers.append(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(
        output_dir, 
        f"{run_name}.log"
    ), "w")
    handler.setLevel(write_lvl)
    formatter = logging.Formatter("%(asctime)s | [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handlers.append(handler)

    return handlers


@dataclass
class CovidTrainingConfiguration():
    # Global config
    root_folder: str = '.'
    random_seed: int = 4
    batch_size: int = 32
    training_fold: typ.Union[int, None] = 0
    max_epochs: int = 100
    validation_frequency: float = 0.1
    verbosity: int = logging.INFO

    device: str = 'cuda'

    # Early Stopping
    early_stop_min_epochs: int=2
    early_stop_milestones: typ.List[typ.Tuple[int,float]] = dc.field(
        default_factory=lambda: [] # [(1,0.3), (2,0.275), (3, 0.25), (5, 0.2)]
    )

    # Dataset Configuration
    synthetic_negative_rate: float = 0.25
    dataloader_num_workers: int = 1

    # Optimizer Configuration
    optim_initial_lr: float = 1e-5
    optim_type: str = 'adam'
    optim_adam_betas: typ.Tuple[int, int] = (0.98, 0.995)
    optim_adam_eps: float = 1e-5
    optim_sgd_momentum: float = 0.9
    optim_warmup_override: typ.Union[None, int] = None # Applies 2/(1-Beta2) by default

    # LR Scheduler Configuration
    optim_scheduler_factor: float = 0.1**0.125
    optim_scheduler_patience: int = 10
    optim_scheduler_cooldown: int = 10
    optim_scheduler_burn_in: float = 2
    optim_minimum_lr: float = 1e-7

    # Model hyperparameters
    chem_layers_per_message: int = 1
    chem_messages_per_pass: int = 1
    chem_hidden_size: int = 64
    chem_bias: bool = False
    chem_nonlinearity: str = 'ReLU'
    chem_undirected: bool = False
    chem_atom_messages: bool = False

    protein_base_dim: int = 16
    protein_output_dim: int = 128
    protein_nonlinearity: str = 'silu'
    protein_downscale_nonlinearity: str = 'tanh'
    protein_maxpool: bool = True
    protein_attention_layers: int = 1
    protein_attention_heads: int = 4
    protein_attention_window: int = 1
    protein_output_use_attention: bool = True
    protein_output_attention_heads: int = 4

    dropout_rate: float = 0.4
    context_dim: int = 32
    negotiation_passes: int = 2
    

def set_random_seeds(seed = 4):
    logger.debug(f"Random seeds set: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    T.manual_seed(seed)


def _create_all_data_splits(root):
    set_random_seeds(4) # Always split data with consistent random seed
    logger.info("Creating data splits -- global training/holdout")
    create_data_split(
        os.path.join(root, 'data'), 
        os.path.join(root, 'data/training'), 
        os.path.join(root, 'data/final_holdout')
    )
    
    for i in range(10):
        logger.info(f"Creating data splits -- train/validation {i:02}")
        create_data_split(
            os.path.join(root, 'data/training'),
            os.path.join(root, f'data/train_{i:02}'), 
            os.path.join(root, f'data/valid_{i:02}')
        )


def _create_model(config, debug=False):
    logger.debug("Creating model")
    model = CovidModel(
        config.dropout_rate,
        config.chem_nonlinearity,
        config.chem_hidden_size,
        config.chem_bias,
        config.chem_layers_per_message,
        config.chem_undirected,
        config.chem_atom_messages,
        config.chem_messages_per_pass,
        211,
        config.protein_base_dim,
        config.protein_output_dim,
        config.protein_nonlinearity,
        config.protein_downscale_nonlinearity,
        config.protein_maxpool,
        config.protein_attention_layers,
        config.protein_attention_heads,
        config.protein_attention_window,
        config.protein_output_use_attention,
        config.protein_output_attention_heads,
        config.negotiation_passes,
        config.context_dim,
        debug=debug
    )
    return model


def _create_optimizer_and_schedulers(model, config):
    logger.debug("Initializing optimizers/schedulers")

    if config.optim_type == 'adam':
        optim = T.optim.Adam(
            model.parameters(), 
            lr=config.optim_initial_lr, 
            betas=config.optim_adam_betas,
            eps=config.optim_adam_eps
        )
    elif config.optim_type == 'sgd':
        optim = T.optim.SGD(
            model.parameters(),
            lr=config.optim_initial_lr,
            momentum=config.optim_sgd_momentum
        )
    elif config.optim_type == 'lcadam':
        optim = LocallyCyclicAdam(
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
        cooldown=config.optim_scheduler_cooldown,
        min_lr=config.optim_minimum_lr
    )

    return optim, warmup, scheduler


def _create_dataloaders(config, validation_synth_neg_rate = 0.0):
    logger.debug("Initializing Datasets")
    data = StitchDataset(os.path.join(config.root_folder, f'data/train_{config.training_fold:02}'))
    dataloader = create_dataloader(
        data, 
        config.batch_size, 
        drop_last=True,
        neg_rate = config.synthetic_negative_rate, 
        num_workers = config.dataloader_num_workers,
    )

    validation_data = StitchDataset(os.path.join(config.root_folder, f'data/valid_{config.training_fold:02}'))
    validation_dataloader = create_dataloader(
        validation_data, 
        config.batch_size, 
        drop_last=True,
        neg_rate=validation_synth_neg_rate, 
        num_workers=config.dataloader_num_workers,
        sample_size = config.batch_size * 50
    )

    return dataloader, validation_dataloader


def train_model(config:CovidTrainingConfiguration, 
                disable_training_resume:bool=False,
                run_name:str=None,
                check_interrupted:typ.Callable=None,
                disable_checkpointing:bool=False,
                debug:bool=False):
    
    if run_name is None:
        run_name = f"train_fold{config.training_fold:02}" if config.training_fold is not None else "train_global"

    logging_handlers = initialize_logger(run_name, print_lvl=config.verbosity)

    logger.info(f"Training initializing -- {run_name}")
    
    if config.training_fold is None:
        logger.warning("Note: You are training this model on the full training data!")

    if not os.path.exists(os.path.join(config.root_folder, 'data/training')):
        # Create data splits if they haven't been computed
        if config.training_fold is not None and config.training_fold != 0:
            raise Exception("Cannot generate trait/test splits with non-zero training fold specified")

        _create_all_data_splits(config.root_folder)

    # Set random seeds for reproducibility
    set_random_seeds(config.random_seed)
    
    # Create and initialize model
    model = _create_model(config, debug)
    
    logger.debug("Pushing model to device")
    model.to(config.device)

    # Create dataloaders
    dataloader, validation_dataloader = _create_dataloaders(config)

    # Create the optimizer & schedulers
    optim, warmup, scheduler = _create_optimizer_and_schedulers(model, config)

    training_state_path = os.path.join(config.root_folder, f"training_state/{run_name}__state.pkl")
    
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
    learning_rates = []
    validation_stats = []
    last_validation = 0

    state = None

    if os.path.exists(training_state_path) and not disable_training_resume:
        logger.info("Loading previous training state")
        state = T.load(training_state_path, map_location=config.device)

    elif os.path.exists(training_state_path + '.gz') and not disable_training_resume:
        logger.info("Loading previous training state")
        try:
            with gzip.open(training_state_path + '.gz', 'rb') as f:
                state = T.load(f, map_location=config.device)
        except:
            state = None
            logger.info("Previous state corrupt -- training from scratch")
            os.remove(training_state_path + '.gz')

    if state is not None:
        epoch = state.get('epoch', epoch-1) + 1
        losses = state.get('losses', losses)
        learning_rates = state.get('learning_rates', learning_rates)
        validation_stats = state.get('validation_stats', validation_stats)
        last_validation = state.get('last_validation', last_validation)
        model.load_state_dict(state['model'])
        optim.load_state_dict(state['optim'])
        warmup.load_state_dict(state['warmup'])
        if 'scheduler' in state:
            scheduler.load_state_dict(state['scheduler'])
    elif disable_training_resume:
        logger.info("Resuming from previous training state manually disabled")
    else:
        logger.info("No previous training state to load")

    # make required subfolders
    for folder in ['outputs', 'models', 'checkpoints', 'training_state']:
        if not os.path.exists(os.path.join(config.root_folder, folder)):
            os.mkdir(os.path.join(config.root_folder, folder))

    if epoch == 0:
        vloss, vacc, v_conf, v_outputs = get_validation_loss()
        v_outputs.to_csv(os.path.join(
            config.root_folder, 
            f"outputs/{run_name}_epoch00_validation_result.csv.gz"
        ), index=False)
        logger.info(f"Validation loss at {epoch:0.1f} = {vloss:0.4f}")
        validation_stats.append([0, vloss, vacc, v_conf])
        learning_rates.append((0, optim.param_groups[0]['lr']))

    epoch_length = len(dataloader)
    interrupted = False

    for epoch in tqdm(range(epoch, config.max_epochs)):
        logger.info(f"Beginning epoch {epoch}")
        
        pct_epoch = 0
        model.train()
        early_stop = False

        for m_e, m_v in config.early_stop_milestones:
            if m_e <= epoch and all(v[1] > m_v for v in validation_stats):
                logger.info(f"Failed to meet early stopping milestone ({m_e}, {m_v}) -- stopping")
                early_stop = True
                break

        if epoch >= config.early_stop_min_epochs and (
            optim.param_groups[0]['lr'] <= config.optim_minimum_lr + 1e-9 or early_stop
        ):
            logger.info("Stopping early")
            break

        for idx, batch in enumerate(tqdm(dataloader, leave=False)):

            model.zero_grad()
            _, _, loss, _ = run_model(model, batch, config.device)

            loss.backward()

            optim.step()
            warmup.step()
                
            pct_epoch = min(1.0, (idx+1)/epoch_length)
            
            losses.append((epoch + pct_epoch, loss.item()))
        
            if pct_epoch == 1.0 or epoch + pct_epoch - last_validation > config.validation_frequency:
                logger.info("Generating validation stats")

                vloss, vacc, v_conf, v_outputs = get_validation_loss()
                model.train()
                v_outputs.to_csv(os.path.join(
                    config.root_folder, 
                    f"outputs/{run_name}_epoch{epoch:02}_validation_result.csv.gz"
                ), index=False)

                validation_stats.append([epoch+pct_epoch, vloss, vacc, v_conf])
                learning_rates.append((epoch + pct_epoch, optim.param_groups[0]['lr']))

                logger.info(f"Validation loss at {epoch+pct_epoch:0.1f} = {vloss:0.4f}")

                if epoch + pct_epoch >= config.optim_scheduler_burn_in:
                    scheduler.step(vloss)

                fig = get_performance_plots(losses, validation_stats, learning_rates)
                fig_path = os.path.join(config.root_folder, f'outputs/{run_name}_performance.png')
                fig.savefig(fig_path)
                plt.close(fig)
                logger.info(f'Generated validation stats -- plot saved to "{fig_path}"')
                last_validation = epoch + pct_epoch

            if check_interrupted is not None and check_interrupted():
                interrupted = True
                break

        if interrupted:
            logger.info("user interrupt received -- quitting")
            break

        state = {
            'epoch': epoch,
            'losses': losses,
            'learning_rates': learning_rates,
            'validation_stats': validation_stats,
            'last_validation': last_validation,
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'warmup': warmup.state_dict(),
            'scheduler': scheduler.state_dict()
        }

        if not disable_checkpointing:
            logger.info('Saving checkpoint')
            with gzip.open(os.path.join(config.root_folder, f'checkpoints/model_{run_name}_{epoch:03}.pkl.gz'), 'wb') as f:
                T.save(state, f)
            
        logger.info('Saving state')
        with gzip.open(training_state_path + ".gz", 'wb') as f:
            T.save(state, f)
    
    for handler in logging_handlers:
        logging.getLogger().removeHandler(handler)

    return losses, validation_stats