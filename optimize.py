import functools
import os
import pickle as pkl
import threading
from datetime import datetime

import hyperopt
from hyperopt import hp
from hyperopt.mongoexp import MongoTrials

from covid.training import CovidTrainingConfiguration, train_model
from covid.utils import getch
import string
import hashlib

from scipy.stats import linregress

import numpy as np
from datetime import timedelta
import copy
import requests

from collections import Mapping, Iterable

# List of depth, budget pairs
LEVEL_DEFS = [
    (1,90),
    (2,50),
    (3,50),
    (4,40),
    (5,30),
    (7,40),
    (10,50),
    (15,100),
]

def make_json_friendly(result):
    if isinstance(result, list):
        return make_json_list_friendly(result)
    
    if len(result) == 0:
        return result
    
    output = {}
    
    for k,v in result.items():
        if isinstance(v, np.ndarray):
            v = list(v)
        elif isinstance(v, timedelta):
            v = v.total_seconds()
        
        if isinstance(v, Mapping):
            v = make_json_friendly(v)
        elif isinstance(v, Iterable):
            v = make_json_list_friendly(v)
            
        
        output[k] = v
    return output

def make_json_list_friendly(lst):
    if len(lst) == 0 or isinstance(lst, str):
        return lst
    
    output = []
    for item in lst:
        if isinstance(item, np.ndarray):
            item = list(item)
        elif isinstance(item, timedelta):
            item = item.total_seconds()
            
        if isinstance(item, Mapping):
            item = make_json_friendly(item)
        elif isinstance(item, Iterable):
            item = make_json_list_friendly(item)
            
        output.append(item)
    return output


def test_parameterization(params, num_epochs, check_interrupted=None):
    config = CovidTrainingConfiguration()

    hsh = hashlib.sha1('hyperopt'.encode())
    hsh.update(repr(sorted(params.items())).encode())

    label = 'hyperopt_' + hsh.hexdigest()[:12]

    config.max_epochs = num_epochs
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

    training_state_path = f"./training_state/{label}__state.pkl.gz"

    r = requests.get(f'http://localhost:5535/training-state/{label}', stream=True)

    if r.status_code == 200:
        with open(training_state_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)

    losses, validation_stats = train_model(
        config, 
        run_name=label,
        disable_training_resume=False, 
        check_interrupted=check_interrupted, 
        disable_checkpointing=True
    )

    with open(training_state_path, 'rb') as f:
        requests.put(f'http://localhost:5535/training-state/{label}', data=f)

    runtime = datetime.now() - started_at

    status = hyperopt.STATUS_OK

    if check_interrupted and check_interrupted():
        status = hyperopt.STATUS_FAIL
    
    v_x, vloss, _, _ = zip(*validation_stats)

    if num_epochs == 1:
        slope, intercept, _, _, _ = linregress(v_x[-3:], vloss[-3:])
    else:
        slope, intercept, _, _, _ = linregress(v_x[-5:], vloss[-5:])

    result = make_json_friendly({
        'loss': intercept + slope * (v_x[-1] + 1.0),
        'runtime': runtime.total_seconds(),
        'status': status,
        'label': label,
        'training_loss_hist': losses,
        'validation_stats': validation_stats,
    })
    return result


def configure_next_level(lvl:int, depth:int, num_suggestions:int=20):
    new_exp_key = f'covid-{lvl}'

    src_trials = MongoTrials('mongo://localhost:1234/covid/jobs', exp_key=f'covid-{lvl-1}')
    all_trials = MongoTrials('mongo://localhost:1234/covid/jobs')
    dest_trials = MongoTrials('mongo://localhost:1234/covid/jobs', exp_key=new_exp_key)

    forward_losses = []
    for trial, loss in zip(src_trials.trials, src_trials.losses()):
        if loss is None:
            forward_losses.append(None)
            continue

        v_x, vloss, _, _ = zip(*trial['result']['validation_stats'])

        slope, intercept, _, _, _ = linregress(v_x[-5:], vloss[-5:])
        forward_losses.append(0.5 * (loss + intercept + slope * depth))

    ordered_idxs = np.argsort([x if x is not None else np.inf for x in forward_losses])

    last_tid = 0 if len(all_trials.tids) == 0 else max(all_trials.tids)
    available_tids = []
    
    for idx in ordered_idxs[num_suggestions:]:
        if len(available_tids) == 0:
            available_tids = dest_trials.new_trial_ids(last_tid)
        
        tid = available_tids.pop(0)
        last_tid = tid

        # copy in the ones that aren't worth exploring further
        cpy = copy.deepcopy(src_trials.trials[idx])
        cpy['exp_key'] = new_exp_key
        cpy['tid'] = tid
        cpy['misc']['tid'] = tid
        del cpy['_id']

        dest_trials.insert_trial_doc(cpy)

    new_tids = []
    new_specs = []
    new_results = []
    new_miscs = []

    for idx in ordered_idxs[:num_suggestions]:
        if src_trials.losses()[idx] is None:
            continue

        if len(available_tids) == 0:
            available_tids = dest_trials.new_trial_ids(last_tid)

        tid = available_tids.pop(0)
        last_tid = tid
        new_tids.append(tid)

        new_specs.append(None)
        new_results.append({'status': 'new'})

        misc = copy.deepcopy(src_trials.trials[idx]['misc'])
        misc['tid'] = tid
        new_miscs.append(misc)

    dest_trials.new_trial_docs(new_tids, new_specs, new_results, new_miscs)
        

def run_optimization(level=1):
    print(f"Optimizing at level {level}")

    search_space = {
        'synthetic_negative_rate': hp.uniform('neg_rate', 0,1),
        'optim_initial_lr': 10 ** -hp.quniform('lr_exp', 2, 5, 0.25),
        'adam_beta1': 1-hp.loguniform('inv_beta1', -5, -1),
        'adam_beta2': 1-hp.loguniform('inv_beta2', -8, -2),
        'optim_adam_eps': hp.loguniform('eps', -15, 0),
        'dropout_rate': hp.uniform('dropout_rate', 0.01, 0.8),
        'chem_layers_per_message': hp.quniform('chem_layers_per_message', 1,4,1),
        'chem_hidden_size': hp.quniform('chem_hidden_size', 64,512,64),
        'chem_nonlinearity': hp.choice(
            'chem_nonlinearity',
            ['ReLU', 'LeakyReLU', 'tanh', 'ELU']),
        'protein_base_dim': hp.quniform('protien_base_dim', 16,128,16),
        'protein_output_dim': hp.quniform('protein_out_dim', 64, 384, 64),
        'protein_nonlinearity': hp.choice(
            'protein_nonlinearity', 
            ['relu', 'silu', 'tanh', 'leaky_relu', 'elu']),
        'protein_downscale_nonlinearity': hp.choice(
            'protein_downscale_nonlinearity', 
            ['relu', 'silu', 'tanh', 'leaky_relu', 'elu']),
    }

    exp_key = f'covid-{level}'

    trials = MongoTrials('mongo://localhost:1234/covid/jobs', exp_key=exp_key)

    if level == 1:
        max_evals = LEVEL_DEFS[0][1]
        depth = 1

    elif level > 1:
        depth, budget = LEVEL_DEFS[level-1]
        last_depth, _ = LEVEL_DEFS[level-2]
        
        num_to_extend = int(np.ceil(budget/2/(depth-last_depth)))
        num_new = int(np.ceil(budget/2/depth))

        if len(trials.trials) == 0:
            print("Generating estimates from previous level")
            configure_next_level(level, num_to_extend)
        
        last_level_trials = MongoTrials('mongo://localhost:1234/covid/jobs', exp_key=f'covid-{level-1}')
        prev_level_count = len([x for x in last_level_trials.losses() if x is not None])

        max_evals = prev_level_count + num_new
        trials.refresh()

    objective = functools.partial(test_parameterization, num_epochs=depth)
    
    if len(trials) >= max_evals:
        print(f"Already completed level {level} -- skipping")
    else:
        best = hyperopt.fmin(
            objective,
            space=search_space,
            algo=hyperopt.tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )

        print(best)

if __name__ == '__main__':
    
    interrupted = False
    def check_interrupted():
        global interrupted
        return interrupted

    for lvl in range(1, len(LEVEL_DEFS) + 1):
        run_optimization(lvl)

    
    # th = threading.Thread(target=run_optimization, daemon=False)
    # th.start()

    # print("Press 'q' or 'ctrl+c' to interrupt hyperparameter search")
        
    # while True:
    #     ch = getch()
    #     if ch in (b'q', 'q', b'\x03', '\x03'):
    #         interrupted = True
    #         break
    
    # print("Trying to quit....")

    # th.join()
