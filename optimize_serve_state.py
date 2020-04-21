from sanic import Sanic
from sanic.response import html, file_stream, text
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

from collections import Counter, defaultdict

SEARCH_SPACE = {
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

PORT = 5000

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

def make_button(to, icon, disabled=False):
    return f"""
        <a class="btn btn-info px-3 {'disabled' if disabled else ''}" role="button" href="{to}">
            <i class="fa fa-{icon}"></i>
        </a>
    """

@app.get("/best-trials/<n:int>")
async def get_current_best(request, n=0):
    if request.args.get('refresh', False):
        TRIALS.refresh()
    
    n = int(n)

    trials = list(TRIALS.trials)
    trials.sort(key=lambda x: x['exp_key'], reverse=True)

    losses = []
    for tr in trials:
        loss = tr['result'].get('validation_stats', [(0,np.inf, 0, 0)])[-1][1]
        #loss = tr['result'].get('loss', np.inf)
        if tr['state'] != 2: # Not Done
            loss = np.inf

        if loss in losses:
            losses.append(np.inf)
        else:
            losses.append(loss)

    idx_list = np.argsort([x if x is not None else np.inf for x in losses])
    
    tr = trials[idx_list[n]]
    fig = get_performance_plots(tr['result']['training_loss_hist'], tr['result']['validation_stats'])
    img = fig_to_base64(fig, close=True).decode('utf-8')
    first_btn = make_button(f"/best-trials/0", "fast-backward")
    prev_btn = make_button(f"/best-trials/{n-1}", "chevron-left", n == 0)
    refresh_btn = make_button(f"/best-trials/{n}?refresh=True", "repeat")
    next_btn = make_button(f"/best-trials/{n+1}", "chevron-right")
    
    stats = get_performance_stats(tr['result']['validation_stats'])

    params = hyperopt.space_eval(SEARCH_SPACE, {k:v[0] for k,v in tr['misc'].get('vals',{}).items()})

    state_lookup = {getattr(hyperopt, k):k for k in ['JOB_STATE_NEW', 'JOB_STATE_ERROR', 'JOB_STATE_RUNNING', 'JOB_STATE_DONE']}
    state_lookup[-1] = 'JOB_STATE_Prev-Level Hints'
    counters = defaultdict(lambda: Counter())

    for tr in TRIALS.trials:
        if 'training_loss_hist' in tr['result'] and tr['state'] == hyperopt.JOB_STATE_DONE:
            epoch = int(tr['result']['training_loss_hist'][-1][0])
            if tr['exp_key'] != f'covid-{epoch}':
                counters[tr['exp_key']][-1] += 1
            else:
                counters[tr['exp_key']][tr['state']] += 1
        else:
            counters[tr['exp_key']][tr['state']] += 1

    status_table = f"""
        <table class="table table-striped table-sm w-auto mt-4">
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

    return html(f"""
        <html>
        <head><meta name="viewport" content="width=device-width, initial-scale=1"/></head>
        <body>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
        <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous">
        <div>
        <h3>Status</h3>
        {status_table}
        </div>
        <h3>Run Details</h3>
        <div class="col-md-12 mb-4 mt-4 text-center">
        {first_btn}{prev_btn}{refresh_btn}{next_btn}<br><br>
        </div>
        <table class="table table-striped table-sm w-auto ml-1">
        <tbody>
        <tr><th>Pass/Loss</th><td>{tr["exp_key"]}</td><td>{'{0:0.3f}'.format(tr['result']['loss'])}</td></tr>
        <tr><th>Epoch</th><td>{stats['epoch'][-1]:0.3f}</td></tr>
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