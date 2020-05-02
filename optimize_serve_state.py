from sanic import Sanic
from sanic.response import html, file_stream, text, redirect
from sanic.exceptions import NotFound

import sys, os
from hyperopt.mongoexp import MongoTrials
from datetime import datetime
import logging
import shutil

import base64
import io
import matplotlib.pyplot as plt
import numpy as np

from covid.reporting import get_performance_plots, get_performance_stats
from covid.constants import MODE_NAMES

import hyperopt
from hyperopt import hp
import hashlib

from collections import Counter, defaultdict

PORT = 5535

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

def fig_to_base64(fig, close=False, **save_kwargs):
    pic_bytes = io.BytesIO()
    
    if 'format' not in save_kwargs:
        save_kwargs['format'] = 'png'
        
    fig.savefig(pic_bytes, **save_kwargs)
    pic_bytes.seek(0)
    result = base64.b64encode(pic_bytes.read())
    if close:
        plt.close(fig)
    return result

app = Sanic(name='CovidProject')
TRIALS = MongoTrials('mongo://localhost:1234/covid/jobs')

@app.get("/training-state/<run_id>")
async def get_training_state(request, run_id):
    logging.info(f"Looking for \"./training_state/{run_id}__state.pkl.gz\"")
    if os.path.exists(f"./training_state/{run_id}__state.pkl.gz"):
        return await file_stream(
            f"./training_state/{run_id}__state.pkl.gz",
            filename=f"{run_id}__state.pkl.gz"
        )
    raise NotFound("No state exists for that id")
        
@app.get('/training-state/delete-all/yes-really')
async def delete_all_yes_really(request, run_id):
    TRIALS.refresh()
    TRIALS.delete_all()
    TRIALS.refresh()
    return redirect(f"/status")

@app.put("/training-state/<run_id>", stream=True)
async def put_training_state(request, run_id):
    with open(f"./training_state/{run_id}__state.pkl.gz.tmp", "wb") as f:
        while True:
            chunk = await request.stream.read()
            if chunk is None:
                break
            f.write(chunk)

    if os.path.exists(f"./training_state/{run_id}__state.pkl.gz"):
        os.remove(f"./training_state/{run_id}__state.pkl.gz")
    shutil.move(f"./training_state/{run_id}__state.pkl.gz.tmp", f"./training_state/{run_id}__state.pkl.gz")
    return text("Model State Uploaded")

def make_button(to, icon, disabled=False, text=None):
    return f"""
        <a class="btn btn-info px-3 {'disabled' if disabled else ''}" role="button" href="{to}">
            <i class="fa fa-{icon}"></i>{' ' + text if text is not None else ''}
        </a>
    """

@app.get("/status")
async def get_status(request):
    if 'refresh' in request.args:
        TRIALS.refresh()

    state_lookup = {getattr(hyperopt, k):k for k in ['JOB_STATE_NEW', 'JOB_STATE_ERROR', 'JOB_STATE_RUNNING', 'JOB_STATE_DONE']}
    state_lookup[-1] = 'JOB_STATE_Prev-Level Hints'
    counters = defaultdict(lambda: Counter())

    for t in TRIALS.trials:
        if 'training_loss_hist' in t['result'] and t['state'] == hyperopt.JOB_STATE_DONE:
            try:
                epoch = int(t['result'].get('training_loss_hist', [(-1,np.inf)])[-1][0] + 1e-8)
            except:
                epoch = -1

            if epoch == -1:
                counters[t['exp_key']][hyperopt.JOB_STATE_RUNNING] += 1
            elif t['exp_key'] != f'covid-{epoch}':
                counters[t['exp_key']][-1] += 1
            else:
                counters[t['exp_key']][t['state']] += 1
        else:
            counters[t['exp_key']][t['state']] += 1

    status_table = f"""
        <table class="table table-striped table-sm w-auto mt-4 ml-1">
        <thead class="thead-light">
        <th>Hyperopt Pass</th><th>{'</th><th>'.join(state_lookup[i][10:].title() for i in range(-1,4))}</th>
        </thead>
        <tbody>
        {''.join('<tr><th>' + k + '</th><td>' + '</td><td>'.join(
            str(counters[k][c]) for c in range(-1,4)
        ) + '</td></tr>' for k in counters.keys())}
        </tbody>
        <table>
    """

    running = [t for t in TRIALS.trials if t['state'] in (
        hyperopt.JOB_STATE_RUNNING,
        hyperopt.JOB_STATE_NEW
    ) and t['owner'] is not None]
    queued = [t for t in TRIALS.trials if t['state'] == hyperopt.JOB_STATE_NEW and t not in running]

    return html(f"""
        <html>
        <head><meta name="viewport" content="width=device-width, initial-scale=1"/></head>
        <body>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
        <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous">
        <div class="col-md-12 mb-4 mt-4 text-center">
        {make_button(f"/best-trials/0", "eye", text="Review Models")}
        <br><br>
        {make_button(f"/status?refresh=True", "refresh")}
        </div>
        <h2>Optimizer Generations</h2>
        {status_table}
        <h2>Active Jobs</h2>
        <table class="table table-striped table-sm w-auto ml-1">
        <thead class="thead-light">
        <th>Node</th><th>Generation</th><th>Id</th>
        </thead><tbody>
        <tr>{'</tr><tr>'.join(f"<td>{t['owner'][0] if t['owner'] is not None else 'n/a'}</td><td>{t['exp_key']}</td><td>{t['tid']}</td>" for t in running)}</tr>
        </tbody></table>
        <h2>{len(queued)} items queued</h2>
        </body>
        </html>
    """)


@app.get("/best-trials/<n:int>")
async def get_current_best(request, n=0):
    if 'refresh' in request.args:
        TRIALS.refresh()
        
    if len(TRIALS.trials) == 0:
        redirect(f"/status")
    
    n = int(n)

    trials = list(TRIALS.trials)
    trials.sort(key=lambda x: x['exp_key'], reverse=True)

    if 'allgens' in request.args:
        all_gens = True
    else:
        all_gens = False
        trials = [x for x in trials if x['exp_key'] == trials[0]['exp_key']]

    losses = []
    for tr in trials:
        loss = min(list(zip(*tr['result'].get('validation_stats', [(0,np.inf, 0, 0)])))[1])
        #loss = tr['result'].get('loss', np.inf)
        if tr['state'] != 2: # Not Done
            loss = np.inf

        if loss in losses:
            losses.append(np.inf)
        else:
            losses.append(loss)

    idx_list = np.argsort([x if x is not None else np.inf for x in losses])
    max_idx = np.argwhere(np.isfinite(np.array(losses)[idx_list])).flatten().max()

    if n > max_idx:
        return redirect(f"/best-trials/{max_idx}")

    tr = trials[idx_list[n]]
    fig = get_performance_plots(tr['result']['training_loss_hist'], tr['result']['validation_stats'])
    img = fig_to_base64(fig, close=True).decode('utf-8')
    
    stats = get_performance_stats(tr['result']['validation_stats'])

    params = hyperopt.space_eval(SEARCH_SPACE, {k:v[0] for k,v in tr['misc'].get('vals',{}).items()})

    hsh = hashlib.sha1('hyperopt'.encode())
    hsh.update(repr(sorted(params.items())).encode())

    label = 'hyperopt_' + hsh.hexdigest()[:12]

    return html(f"""
        <html>
        <head><meta name="viewport" content="width=device-width, initial-scale=1"/></head>
        <body>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
        <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous">
        <div>
        <h3>Run Details: #{n}/{max_idx}</h3>
        <div class="col-md-12 mb-4 mt-4 text-center">
        {make_button(f"/status", "server", text="Job Status")}
        <br><br>
        {make_button(f"/best-trials/0{'' if not all_gens else '?allgens=t'}", "fast-backward", n==0)}
        {make_button(f"/best-trials/{n-1}{'' if not all_gens else '?allgens=t'}", "chevron-left", n == 0)}
        {make_button(f"/best-trials/{n}?refresh=True{'' if not all_gens else '&allgens=t'}", "refresh")}
        {make_button(f"/best-trials/{n+1}{'' if not all_gens else '?allgens=t'}", "chevron-right", n == max_idx)}
        {make_button(f"/best-trials/{max_idx}{'' if not all_gens else '?allgens=t'}", "fast-forward", n == max_idx)}
        </div>
        <br>
        <table class="table table-striped table-sm w-auto ml-1">
        <tbody>
        <tr><th>Pass</th><td>{tr["exp_key"]}</td></tr>
        <tr><th>Label</th><td>{label}</td></tr>
        <tr><th>Task ID</th><td>{tr['tid']}</td></tr>
        <tr><th>Raw Loss</th><td>{tr['result'].get('validation_stats', [(0,np.inf, 0, 0)])[-1][1]:0.4f}</td></tr>
        <tr><th>Adj Loss</th><td>{'{0:0.4f}'.format(tr['result'].get('loss',np.inf))}
        <tr><th>Epoch</th><td>{stats['epoch'][-1]:0.0f}</td></tr>
        </tbody>
        </table>
        <table class="table table-striped table-sm w-auto ml-1">
        <thead class="thead-light"><th>stat</th><th>{'</th><th>'.join(MODE_NAMES)}</th></thead>
        <tbody>
        {''.join('<tr><th>' + k + '</th><td>' + '</td><td>'.join(
                '{0:0.3f}'.format(v) for v in vals[-1]
            ) + '</td></tr>' for k,vals in stats.items() if k != 'epoch'
        )}
        </tbody></table>
        <img src="data:image/png;base64, {img}" class="img-fluid"/>
        <h4>Parameters</h4>
        <table class="table table-sm w-auto ml-1" >
        <tr>{'</tr><tr>'.join('<th>{0}</th><td>{1}</td>'.format(k,v) for k,v in params.items())}</tr>
        </table>
        </body></html>
    """)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
