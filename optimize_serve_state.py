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

@app.get("/best-trials/<n:int>")
async def get_current_best(request, n=0):
    if request.args.get('refresh', False):
        TRIALS.refresh()
    
    n = int(n)

    losses = [tr['result'].get('training_loss_hist', [(0,np.inf)])[-1][1] for tr in TRIALS.trials]

    idx_list = np.argsort([x if x is not None else np.inf for x in TRIALS.losses()])
    
    tr = TRIALS.trials[idx_list[n]]
    fig = get_performance_plots(tr['result']['training_loss_hist'], tr['result']['validation_stats'])
    img = fig_to_base64(fig, close=True).decode('utf-8')
    prev_btn = f'<button onclick="window.location.href = \'/best-trials/{n-1}\';">&lt;</button>'
    next_btn = f'<button onclick="window.location.href = \'/best-trials/{n+1}\';">&gt;</button>'
    return html(f"""
        <html><body>
        {prev_btn if n > 0 else ''} &nbsp; {next_btn}<br><br>
        {tr["result"]["loss"]}<br>
        <img src="data:image/png;base64, {img}"  style="width: 100%; height: auto;"/>
        </body></html>
    """)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)