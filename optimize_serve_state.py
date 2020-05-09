from sanic import Sanic
from sanic.response import html, file_stream, text, redirect, json
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
from pymongo import MongoClient
from bson import ObjectId

from collections import Counter, defaultdict

from pprint import pformat
import humanize

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
        'protein_attention_layers': hp.quniform('att_layers', 1, 5, 1),
        'protein_attention_heads': hp.quniform('att_heads', 4, 16, 4),
        'protein_attention_window': hp.quniform('att_window', 1, 7, 2),
        'context_dim': hp.quniform('context_dim', 64, 512, 64),
        'negotiation_passes': hp.quniform('negotiation_passes', 1, 8, 1),
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
TRIALS_REFRESHED = datetime.now()
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

def create_delete_prompt(desc = "Something"):
    return f"""
        <html>
        <head><meta name="viewport" content="width=device-width, initial-scale=1"/></head>
        <body>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
        <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous">
        <h2>DELETE {desc}</h2>
        <strong>Are you sure?</strong>
        <form action="" method="get">
        <input type="text" name="really" placeholder="Type 'yes' to confirm"/>
        </form>
    """
        
@app.get('/delete-failed')
async def delete_failed(request):
    global TRIALS, TRIALS_REFRESHED

    if request.args.get("really", "no") == "yes":
        jobs = MongoClient('localhost', 1234).covid.jobs
        to_delete = list(jobs.find({'result.status':'fail'}))
        for obj in to_delete:
            jobs.find_one_and_delete({'_id':obj['_id']})
        TRIALS_REFRESHED = datetime.now()
        TRIALS.refresh()
        return redirect("/status")
    return html(create_delete_prompt("FAILED JOBS"))

@app.get('/delete-gen/<gen>')
async def delete_gen(request, gen):
    global TRIALS, TRIALS_REFRESHED

    if request.args.get('really', 'no') == 'yes':
        gen_trials = MongoTrials('mongo://localhost:1234/covid/jobs', f'covid-{gen}')
        gen_trials.refresh()
        gen_trials.delete_all()
        return redirect(f"/status/?refresh=true")
    return html(create_delete_prompt(f"GENERATION 'covid-{gen}'"))

@app.get('/delete-all')
async def delete_all_yes_really(request):
    global TRIALS, TRIALS_REFRESHED

    if request.args.get('really', 'no') == 'yes':
        TRIALS.refresh()
        TRIALS_REFRESHED = datetime.now()
        TRIALS.delete_all()
        TRIALS.refresh()
        return redirect(f"/status/?refresh=true")
    return html(create_delete_prompt(f"ALL DATA AND JOBS"))


def make_button(to, icon, disabled=False, text=None, cls='btn-info px-3'):
    return f"""
        <a class="btn btn-info px-3 {'disabled' if disabled else ''}" role="button" href="{to}">
            <i class="fa fa-{icon}"></i>{' ' + text if text is not None else ''}
        </a>
    """

def make_jobs_table(jobs_list):
    return f"""<table class="table table-striped table-sm w-auto ml-1">
        <thead class="thead-light">
        <th>Gen</th><th>Node</th><th>Created</th><th>Runtime</th><th>Id</th>
        </thead><tbody>
        <tr>{'</tr><tr>'.join(
            f'<td>{t["exp_key"]}</td>'
            + f'<td>{t["owner"][0] if t["owner"] is not None else "n/a"}</td>'
            + f'<td>{"" if t["book_time"] is None else humanize.naturaltime(t["book_time"])}</td>'
            + f'<td>{"" if t["refresh_time"] is None else str(t["refresh_time"] - t["book_time"])}</td>'
            + f'<td><a href="/best-trials/?tid={t["tid"]}">{t["tid"]}</a></td>'
            for t in jobs_list
        )}</tr>
        </tbody></table>"""

@app.get("/status")
async def get_status(request):
    global TRIALS, TRIALS_REFRESHED

    if 'refresh' in request.args:
        TRIALS.refresh()
        TRIALS_REFRESHED = datetime.now()

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
    )]
    
    failed = [t for t in TRIALS.trials if t['result'].get('status') == 'fail']

    completed = [t for t in TRIALS.trials if t not in running and t not in failed]

    return html(f"""
        <html>
        <head><meta name="viewport" content="width=device-width, initial-scale=1"/></head>
        <body>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
        <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous">
        <div class="col-md-12 mb-4 mt-4 text-center">
        {make_button(f"/best-trials/?n=0", "eye", text="Review Models")}
        <br><br>
        {make_button(f"/status?refresh=True", "refresh")}
        </div>
        <i>Data refreshed {humanize.naturaltime(TRIALS_REFRESHED)}</i><br/>
        <h2>Optimizer Generations</h2>
        {status_table}
        <h2>{len(running)} Active Jobs</h2>
        {make_jobs_table(running)}
        <h2>{len(failed)} Failed</h2>
        {make_jobs_table(failed)}
        <h3>{len(completed)} Completed</h3>
        </body>
        </html>
    """)

@app.get("/best-trials/raw")
async def trials_raw(request):
    global TRIALS, TRIALS_REFRESHED

    n = request.args.get('n')
    tid = request.args.get('tid')
    label = request.args.get('label')
    
    if all(x is None for x in (n, tid, label)):
        n = 0

    if 'refresh' in request.args:
        TRIALS.refresh()
        TRIALS_REFRESHED = datetime.now()
        
    if len(TRIALS.trials) == 0:
        redirect(f"/status")
    
    tr = None
    
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
    
    if n is not None:
        n = int(n)
        if n > max_idx:
            return redirect(f"/best-trials/{max_idx}")

        tr = trials[idx_list[n]]
    elif tid is not None:
        tid = int(tid)
        for t in trials:
            if t['tid'] == tid:
                tr = t
                break
    elif label is not None:
        for t in trials:
            params = hyperopt.space_eval(SEARCH_SPACE, {k:v[0] for k,v in t['misc'].get('vals',{}).items()})
            hsh = hashlib.sha1('hyperopt'.encode())
            hsh.update(repr(sorted(params.items())).encode())

            if label == 'hyperopt_' + hsh.hexdigest()[:12]:
                tr = t
                break
                
    if tr is None:
        return NotFound()
        
    return text(pformat(tr))

@app.get("/delete-trial/<id>")
async def delete_trial(request, id):
    global TRIALS, TRIALS_REFRESHED

    jobs = MongoClient('localhost', 1234).covid.jobs
    job = jobs.find_one({'_id':ObjectId(id)})

    if job is None:
        return NotFound()

    if request.args.get("really", "no") == "yes":
        jobs.find_one_and_delete({'_id':ObjectId(id)})
        TRIALS_REFRESHED = datetime.now()
        TRIALS.refresh()
        return redirect("/status")
    return html(create_delete_prompt(f"JOB {job['tid']}"))


@app.get("/best-trials")
async def get_current_best(request):
    global TRIALS, TRIALS_REFRESHED

    n = request.args.get('n')
    tid = request.args.get('tid')
    label = request.args.get('label')
    gen = request.args.get('gen')
    
    if all(x is None for x in (n, tid, label)):
        n = 0

    if 'refresh' in request.args:
        TRIALS.refresh()
        TRIALS_REFRESHED = datetime.now()
        
    if len(TRIALS.trials) == 0:
        redirect(f"/status")
    
    tr = None
    
    trials = list(TRIALS.trials)
    trials.sort(key=lambda x: x['exp_key'], reverse=True)

    if gen is not None:
        gen = int(gen)
        trials = []
        for t in TRIALS.trials:
            try:
                epoch = int(t['result'].get('training_loss_hist', [(-1,np.inf)])[-1][0] + 1e-8)
            except:
                epoch = -1
            
            if epoch == gen and t['exp_key'] == f'covid-{gen}':
                trials.append(t)

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
    
    button_suffix = '' if gen is None else f'gen={gen}'

    if n is not None:
        n = int(n)
        if n > max_idx:
            return redirect(f"/best-trials/?n={max_idx}&{button_suffix}")

        tr = trials[idx_list[n]]
    elif tid is not None:
        tid = int(tid)
        for t in trials:
            if t['tid'] == tid:
                tr = t
                break
    elif label is not None:
        for t in trials:
            params = hyperopt.space_eval(SEARCH_SPACE, {k:v[0] for k,v in t['misc'].get('vals',{}).items()})
            hsh = hashlib.sha1('hyperopt'.encode())
            hsh.update(repr(sorted(params.items())).encode())

            if label == 'hyperopt_' + hsh.hexdigest()[:12]:
                tr = t
                break
    
    if tr is None:
        return NotFound()

    for i, idx in enumerate(idx_list):
        if trials[idx] == tr:
            n = i
            break
            
    if n is None:
        n = 0
        
    if 'training_loss_hist' in tr['result']:
        fig = get_performance_plots(tr['result']['training_loss_hist'], tr['result']['validation_stats'])
        img = fig_to_base64(fig, close=True).decode('utf-8')
        
        stats = get_performance_stats(tr['result']['validation_stats'])
    else:
        img = ''
        stats = {'epoch':[0]}
        
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
        {make_button(f"/best-trials/?n=0&{button_suffix}", "fast-backward", n==0)}
        {make_button(f"/best-trials/?n={n-1}&{button_suffix}", "chevron-left", n == 0)}
        {make_button(f"/best-trials/?tid={tr['tid']}&refresh=True&{button_suffix}", "refresh")}
        {make_button(f"/best-trials/?n={n+1}&{button_suffix}", "chevron-right", n == max_idx)}
        {make_button(f"/best-trials/?n={max_idx}&{button_suffix}", "fast-forward", n == max_idx)}
        </div>
        <i>Data refreshed {humanize.naturaltime(TRIALS_REFRESHED)}</i><br/>
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

        <div class="col-md-12 mb-4 mt-4 text-center">
        {make_button(f"/best-trials/raw?tid=" + str(tr['tid']), "code", text="Full JSON Details")}
        {make_button(f"/delete-trial/{tr['_id']}", 'trash', text='Delete Trial')}
        </div>
        <img src="data:image/png;base64, {img}" class="img-fluid"/>
        <table class="table table-striped table-sm w-auto ml-1">
        <thead class="thead-light"><th>stat</th><th>{'</th><th>'.join(MODE_NAMES)}</th></thead>
        <tbody>
        {''.join('<tr><th>' + k + '</th><td>' + '</td><td>'.join(
                '{0:0.3f}'.format(v) for v in vals[-1]
            ) + '</td></tr>' for k,vals in stats.items() if k != 'epoch'
        )}
        </tbody></table>
        
        <h4>Parameters</h4>
        <table class="table table-sm w-auto ml-1" >
        <tr>{'</tr><tr>'.join('<th>{0}</th><td>{1}</td>'.format(k,v) for k,v in params.items())}</tr>
        </table>
        </body></html>
    """)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
