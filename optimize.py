import copy
import functools
import hashlib
import os
import pickle as pkl
import shutil
import string
import threading
import time
from collections import Iterable, Mapping, defaultdict
from datetime import datetime, timedelta

import hyperopt
import numpy as np
import requests
from hyperopt import hp
from hyperopt.mongoexp import MongoTrials
from scipy.stats import linregress

from covid.training import CovidTrainingConfiguration, train_model
from covid.utils import getch

# List of (depth, new_budget, extend_budget)  thruples
LEVEL_DEFS = [
    (1, 100, 0),
    (2, 20, 30),
    (3, 30, 20),
    (4, 40, 20),
    (5, 25, 20),
    (7, 14, 36),
    (10, 100, 48)
]

SEARCH_SPACE = {
        'synthetic_negative_rate': hp.uniform('neg_rate', 0, 0.5),
        'optim_initial_lr': 10 ** -hp.quniform('lr_exp', 3, 6, 0.25),
        'adam_beta1': 1-hp.loguniform('inv_beta1', -6, -1),
        'adam_beta2': 1-hp.loguniform('inv_beta2', -8, -4),
        'optim_adam_eps': hp.loguniform('eps', -15, 0),
        'dropout_rate': hp.uniform('dropout_rate', 0.05, 0.5),
        'chem_layers_per_message': hp.quniform('chem_layers_per_message', 1,4,1),
        'chem_messages_per_pass': hp.quniform('chem_messages_per_pass', 1,4,1),
        'chem_hidden_size': hp.quniform('chem_hidden_size', 64,384,64),
        'chem_nonlinearity': hp.choice(
            'chem_nonlinearity',
            ['ReLU', 'LeakyReLU', 'tanh']),
        'chem_bias': hp.choice('chem_bias', [True, False]),
        'chem_mode': hp.choice('chem_mode', ['atom', 'bond', 'bond-undirected']),
        'protein_base_dim': hp.quniform('protien_base_dim', 16,80,16),
        'protein_output_dim': hp.quniform('protein_out_dim', 64, 384, 64),
        'protein_nonlinearity': hp.choice(
            'protein_nonlinearity', 
            ['relu', 'silu', 'tanh']),
        'protein_downscale_nonlinearity': hp.choice(
            'protein_downscale_nonlinearity', 
            ['relu', 'silu', 'tanh']),
        'protein_maxpool': hp.choice('protein_maxpool', [True, False]),
        'context_dim': hp.quniform('context_dim', 64, 512, 64),
        'negotiation_passes': hp.quniform('negotiation_passes', 1, 8, 1)
    }

NUM_NODES = 6

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
    
    if params['chem_mode'] == 'atom':
        config.chem_atom_messages = True
        config.chem_undirected = False
    elif params['chem_mode'] == 'bond':
        config.chem_atom_messages = False
        config.chem_undirected = False
    elif params['chem_mode'] == 'bond-undirected':
        config.chem_atom_messages = False
        config.chem_undirected = True
        
    del params['chem_mode']

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
        with open(training_state_path + ".tmp", "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)

        if os.path.exists(training_state_path):
            os.remove(training_state_path)
        shutil.move(training_state_path + ".tmp", training_state_path)

    try:
        losses, validation_stats = train_model(
            config, 
            run_name=label,
            disable_training_resume=False, 
            check_interrupted=check_interrupted, 
            disable_checkpointing=True
        )
    except Exception as ex:
        return {'status': hyperopt.STATUS_FAIL, 'error': repr(ex)}

    try:
        with open(training_state_path, 'rb') as f:
            requests.put(f'http://localhost:5535/training-state/{label}', data=f)
    except:
        print("Failed to push model state!")

    runtime = datetime.now() - started_at

    status = hyperopt.STATUS_OK

    if check_interrupted and check_interrupted():
        status = hyperopt.STATUS_FAIL
    
    v_x, vloss, _, _ = zip(*validation_stats)

    hist_length = {2:3, 3:5, 5:8}.get(num_epochs, 10)

    slope, intercept, _, _, _ = linregress(v_x[-hist_length:], vloss[-hist_length:])

    loss = min(vloss[-1], intercept + slope * (v_x[-1] + 1))

    result = make_json_friendly({
        'loss': loss,
        'runtime': runtime.total_seconds(),
        'status': status,
        'label': label,
        'training_loss_hist': losses,
        'validation_stats': validation_stats,
    })
    return result


def configure_next_level(lvl:int, depth:int, budget:int=50):
    new_exp_key = f'covid-{lvl}'

    src_trials = MongoTrials('mongo://localhost:1234/covid/jobs', exp_key=f'covid-{lvl-1}')
    all_trials = MongoTrials('mongo://localhost:1234/covid/jobs')
    dest_trials = MongoTrials('mongo://localhost:1234/covid/jobs', exp_key=new_exp_key)

    hist_length = {2:3, 3:5, 5:8}.get(depth, 10)

    forward_losses = []
    for trial, loss in zip(src_trials.trials, src_trials.losses()):
        if loss is None:
            forward_losses.append(None)
            continue

        v_x, vloss, _, _ = zip(*trial['result']['validation_stats'])

        slope, intercept, _, _, _ = linregress(v_x[-hist_length:], vloss[-hist_length:])
        forward_losses.append(min(0.5 * (loss + intercept + slope * v_x[-1] + slope * (1-0.8**(depth-v_x[-1]))/(1-0.8)), loss))

    ordered_idxs = list(np.argsort([x if x is not None else np.inf for x in forward_losses]))

    last_tid = 0 if len(all_trials.tids) == 0 else max(all_trials.tids)
    available_tids = []
    
    result_docs = []

    while len(ordered_idxs) > 0:
        idx = ordered_idxs.pop(0)
        if src_trials.losses()[idx] is None:
            continue

        epochs_completed = src_trials.trials[idx]['result'].get('training_loss_hist', [(0,np.inf)])[-1][0]
        
        spec = None
        result = {'status': 'new'}
        misc = copy.deepcopy(src_trials.trials[idx]['misc'])
        
        result_docs.append((spec, result, misc))
        budget -= (depth - epochs_completed)
        if budget <= 0:
            break

    while len(ordered_idxs) > 0:
        idx = ordered_idxs.pop()

        if src_trials.losses()[idx] is None:
            continue

        if len(available_tids) == 0:
            available_tids = dest_trials.new_trial_ids(last_tid)
        
        tid = available_tids.pop(0)
        last_tid = tid

        # copy in the ones that aren't worth exploring further
        cpy = copy.deepcopy(src_trials.trials[idx])
        cpy['exp_key'] = new_exp_key
        cpy['tid'] = tid
        cpy['misc']['tid'] = tid

        cpy['misc']['idxs'] = {k:[tid] for k in cpy['misc']['idxs'].keys()}

        del cpy['_id']

        dest_trials.insert_trial_doc(cpy)
    
    return result_docs


def create_suggestion_box(docs):
    docs = list(docs)
    def suggest(new_ids, domain, trials, seed, *args, **kwargs):
        nonlocal docs
        if len(docs) > 0:
            num_to_take = min(len(new_ids), len(docs))
            
            selected_docs = docs[:num_to_take]
            docs = docs[num_to_take:]

            sel_spec, sel_result, sel_misc = zip(*selected_docs)

            for tid, misc in zip(new_ids[:num_to_take], sel_misc):
                misc['cmd'] = domain.cmd
                misc['workdir'] = domain.workdir
                misc['idxs'] = {k:[tid] for k in misc['idxs'].keys()}
                misc['tid'] = tid

            return trials.new_trial_docs(new_ids[:num_to_take], sel_spec, sel_result, sel_misc)

        return hyperopt.tpe.suggest(new_ids, domain, trials, seed, *args, **kwargs)

    return suggest


# def create_suggestion_box(trials_to_use):
#     def suggest_with_tpe_fallback(new_ids, domain, trails, seed):
#         nonlocal trials_to_use

#     return trials.new_trial_docs([])
def run_optimization(level=1):
    print(f"Optimizing at level {level}")

    next_lvl_trials = MongoTrials('mongo://localhost:1234/covid/jobs', exp_key=f'covid-{level+1}')
    if len(next_lvl_trials.trials) > 0:
        print(f"Already completed level {level} -- skipping")
        return

    exp_key = f'covid-{level}'

    trials = MongoTrials('mongo://localhost:1234/covid/jobs', exp_key=exp_key)

    suggestion_box = hyperopt.tpe.suggest

    if level == 1:
        max_evals = LEVEL_DEFS[0][1]
        depth = 1

    elif level > 1:
        depth, new_budget, extend_budget = LEVEL_DEFS[level-1]
        last_depth, _, _ = LEVEL_DEFS[level-2]

        # Minimum one per node for the expensive ones -- no point wasting compute time
        num_new = int(np.ceil((new_budget/depth)/NUM_NODES)*NUM_NODES)

        if len(trials.trials) == 0:
            print("Generating estimates from previous level")
            result_docs = configure_next_level(level, depth, extend_budget)
            num_to_extend = len(result_docs)

            suggestion_box = create_suggestion_box(result_docs)
        
        last_level_trials = MongoTrials('mongo://localhost:1234/covid/jobs', exp_key=f'covid-{level-1}')
        prev_level_count = len([x for x in last_level_trials.losses() if x is not None])

        max_evals = prev_level_count + num_new
        trials.refresh()

    objective = functools.partial(test_parameterization, num_epochs=depth)
    
    if len([x for x in trials.statuses() if x == 'ok']) >= max_evals:
        print(f"Already completed level {level} -- skipping")
    else:
        best = hyperopt.fmin(
            objective,
            space=SEARCH_SPACE,
            algo=suggestion_box,
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
