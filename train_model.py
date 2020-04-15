import functools
import os
import random

import gzip
import numpy as np
import threading
import torch as T
from tqdm import tqdm

import covid
from covid.data import *
from covid.datasets import StitchDataset, create_data_split, create_dataloader
from covid.model import (CovidModel, calculate_average_loss_and_accuracy,
                         create_protein_model, run_model, RandomModel)
from covid.modules import *
from covid.modules.chemistry import MPNEncoder
from covid.reporting import get_performance_plots
from covid.schedulers import LinearWarmupScheduler
from covid.utils import getch

import logging

DROPOUT_RATE = 0.4
BATCH_SIZE = 16
VALIDATION_FREQUENCY = 0.2
SYNTHETIC_NEGATIVE_RATE = 0.2

DEVICE = 'cuda'

TRAINING_FOLD = 1


# set random seeds
# 4 -- chosen by fair die roll.  Guaranteed to be random.  https://xkcd.com/221/
np.random.seed(4 + TRAINING_FOLD)
random.seed(4 + TRAINING_FOLD)
T.manual_seed(4 + TRAINING_FOLD)

def initialize_logger(output_dir='./logs'):
    if not os.path.exists("./logs"):
        os.mkdir("./logs")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
     
    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(output_dir, f"debug_{TRAINING_FOLD:02}.log"), "w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

if __name__ == '__main__':
    initialize_logger()
    logging.info(f"Training initializing -- fold {TRAINING_FOLD}")

    if not os.path.exists('./data/training'):
        logging.info("Creating data splits")
        create_data_split('./data', './data/training', './data/final_holdout')
        
        for i in range(10):
            create_data_split('./data/training', f'./data/train_{i:02}', f'./data/valid_{i:02}')

        # Have to reset the seeds again to match times that we don't generate the data
        np.random.seed(4)
        random.seed(4)
        T.manual_seed(4)

    logging.info("Initializing model")
    chem_model = MPNEncoder(
        layers_per_message=2, 
        hidden_size=300,
        dropout=DROPOUT_RATE
    )
    protein_model = create_protein_model(dropout=DROPOUT_RATE)
    model = CovidModel(chem_model, protein_model, dropout=DROPOUT_RATE)

    logging.info("Pushing model to device")
    model.to(DEVICE)

    logging.info("Initializing Datasets")
    data = covid.datasets.StitchDataset(f'./data/train_{TRAINING_FOLD:02}')
    dataloader = create_dataloader(
        data, BATCH_SIZE, neg_rate = SYNTHETIC_NEGATIVE_RATE, num_workers=1
    )

    validation_data = covid.datasets.StitchDataset(f'./data/valid_{TRAINING_FOLD:02}')
    validation_dataloader = create_dataloader(
        validation_data, BATCH_SIZE, neg_rate=0.2, num_workers=1
    )

    logging.info("Initializing optimizers/schedulers")
    optim = T.optim.Adam(model.parameters(), lr=1e-4, betas=(0.95, 0.99))
    warmup = LinearWarmupScheduler(optim, 2000)

    losses = []
    validation_stats = []

    get_validation_loss = functools.partial(
        calculate_average_loss_and_accuracy, 
        model, 
        validation_dataloader,
        DEVICE
    )

    epoch = 0
    last_validation = epoch

    if os.path.exists(f"./training_state_{TRAINING_FOLD:02}.pkl"):
        logging.info("Loading previous training state")
        state = T.load(f"./training_state_{TRAINING_FOLD:02}.pkl", map_location=DEVICE)
            
        epoch = state.get('epoch', epoch-1) + 1
        losses = state.get('losses', losses)
        validation_stats = state.get('validation_stats', validation_stats)
        last_validation = state.get('last_validation', last_validation)
        model.load_state_dict(state['model'])
        optim.load_state_dict(state['optim'])
        warmup.load_state_dict(state['warmup'])
    else:
        logging.info("No previous training state to load")
        
    if epoch == 0:
        vloss, vacc, v_conf = get_validation_loss()
        validation_stats.append([0, vloss, vacc, v_conf])

    if not os.path.exists("./checkpoints/"):
        os.mkdir("./checkpoints")

    print("Press 'q' or 'ctrl+c' to interrupt training loop")

    interrupted = False

    def wait_for_q():
        global interrupted
        while True:
            ch = getch()
            if ch in (b'q', 'q', b'\x03', '\x03'):
                interrupted = True
                break

    threading.Thread(target=wait_for_q, daemon=True).start()

    epoch_length = len(dataloader)

    for epoch in tqdm(range(epoch, 100)):
        logging.info(f"Beginning epoch {epoch}")
        
        model.train()
        pct_epoch = 0
        
        for idx, batch in enumerate(tqdm(dataloader, leave=False)):

            model.zero_grad()
            _, _, loss, _ = run_model(model, batch, DEVICE)

            loss.backward()

            optim.step()
            warmup.step()
                
            pct_epoch = min(1.0, idx/epoch_length)
            
            losses.append((epoch + pct_epoch, loss.item()))
        
            if pct_epoch == 1.0 or epoch + pct_epoch - last_validation > VALIDATION_FREQUENCY:
                logging.info("Generating validation stats")

                vloss, vacc, v_conf = get_validation_loss()
                validation_stats.append([epoch+pct_epoch, vloss, vacc, v_conf])
                
                get_performance_plots(losses, validation_stats).savefig(f'./performance_{TRAINING_FOLD:02}.png')
                logging.info(f'Generated validation stats -- plot saved to "./performance_{TRAINING_FOLD:02}.png"')
                last_validation = epoch + pct_epoch

            if interrupted:
                break

        if interrupted:
            logging.info("user interrupt received -- quitting")
            break
                
        logging.info('Saving checkpoint')
        state = {
            'epoch': epoch,
            'losses': losses,
            'validation_stats': validation_stats,
            'last_validation': last_validation,
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'warmup': warmup.state_dict(),
        }
        with gzip.open(f'./checkpoints/model_{TRAINING_FOLD:02}_{epoch:03}.pkl.gz', 'wb') as f:
            T.save(state, f)
        
        logging.info('Saving state')
        T.save(state, f"./training_state_{TRAINING_FOLD:02}.pkl")
