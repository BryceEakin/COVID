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

from covid.reporting import get_performance_plots

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
        os.remove(f"./training_state/{run_id}__state.pkl.gz.tmp")
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
        #loss = tr['result'].get('training_loss_hist', [(0,np.inf)])[-1][1]
        loss = tr['result'].get('loss', np.inf)
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
    
    return html(f"""
        <html>
        <head><meta name="viewport" content="width=device-width, initial-scale=1"/></head>
        <body>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
        <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous">
        <div class="col-md-12 mb-4 text-center">
        {first_btn}{prev_btn}{refresh_btn}{next_btn}<br><br>
        </div>
        <div>
        <span>{tr["exp_key"]} | {tr["result"]["loss"]}</span>
        </div><br>
        <img src="data:image/png;base64, {img}" class="img-fluid"/>
        </body></html>
    """)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)