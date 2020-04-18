import functools
import os
import pickle as pkl
import threading
from datetime import datetime

import hyperopt
from hyperopt import hp

from covid.training import CovidTrainingConfiguration, train_model
from covid.utils import getch


def test_parameterization(params, check_interrupted=None):
    config = CovidTrainingConfiguration()

    config.max_epochs = 3
    config.batch_size = 24

    config.optim_adam_betas = (params['adam_beta1'], params['adam_beta2'])
    del params['adam_beta1']
    del params['adam_beta2']

    for key, val in params.items():
        if not hasattr(config, key):
            print(f"No parameter {key} -- skipping")
            continue

        if isinstance(getattr(config, key), int):
            setattr(config, key, int(val))
        else:
            setattr(config, key, val)

    started_at = datetime.now()

    losses, validation_stats = train_model(
        config, 
        disable_training_resume = True, 
        check_interrupted=check_interrupted, 
        disable_checkpointing=True
    )

    runtime = datetime.now() - started_at

    status = hyperopt.STATUS_OK

    if check_interrupted and check_interrupted():
        status = hyperopt.STATUS_FAIL
    
    _, vloss, _, _ = zip(*validation_stats)

    result = {
        'loss': min(vloss),
        'runtime': runtime,
        'status': status,
        'training_loss_hist': losses,
        'validation_stats': validation_stats
    }
    return result

def run_optimization():
    objective = functools.partial(test_parameterization, check_interrupted = check_interrupted)
    
    search_space = {
        'synthetic_negative_rate': hp.uniform('neg_rate', 0,1),
        'optim_initial_lr': 10 ** -hp.quniform('lr_exp', 2, 5, 0.25),
        'adam_beta1': 1-hp.loguniform('inv_beta1', -5, -1),
        'adam_beta2': 1-hp.loguniform('inv_beta2', -8, -2),
        'optim_adam_eps': hp.loguniform('eps', -15, 0),
        'chem_layers_per_message': hp.quniform('chem_layers_per_message', 1,4,1),
        'chem_hidden_size': hp.quniform('chem_hidden_size', 64,384,64),
        'chem_nonlinearity': hp.choice(
            'chem_nonlinearity',
            ['ReLU', 'LeakyReLU', 'tanh', 'ELU']),
        'protein_base_dim': hp.quniform('protien_base_dim', 16,80,16),
        'protein_output_dim': hp.quniform('protein_out_dim', 64, 384, 64),
        'protein_nonlinearity': hp.choice(
            'protein_nonlinearity', 
            ['relu', 'silu', 'tanh', 'leaky_relu', 'elu']),
        'protein_downscale_nonlinearity': hp.choice(
            'protein_downscale_nonlinearity', 
            ['relu', 'silu', 'tanh', 'leaky_relu', 'elu']),
    }

    if os.path.exists("./hyperopt_trials.pkl"):
        with open("./hyperopt_trials.pkl", "rb") as f:
            trials = pkl.load(f)
    else:
        trials = hyperopt.Trials()

    
    best = hyperopt.fmin(
        objective,
        space=search_space,
        algo=hyperopt.tpe.suggest,
        max_evals=10,
        trials=trials
    )

    print(best)
    with open("./hyperopt_trials.pkl", "wb") as f:
        pkl.dump(trials, f, pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    
    interrupted = False
    def check_interrupted():
        global interrupted
        return interrupted
    
    th = threading.Thread(target=run_optimization, daemon=False)
    th.start()

    print("Press 'q' or 'ctrl+c' to interrupt hyperparameter search")
        
    while True:
        ch = getch()
        if ch in (b'q', 'q', b'\x03', '\x03'):
            interrupted = True
            break
    
    print("Trying to quit....")

    th.join()
