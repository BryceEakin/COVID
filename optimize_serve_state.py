from sanic import Sanic
from sanic.response import html, file_stream, text, json
from sanic.exceptions import NotFound

import sys, os
from hyperopt.mongoexp import MongoTrials
from datetime import datetime
import logging

PORT = 8000

app = Sanic(name='CovidProject')

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
    with open(f"./training_state/{run_id}__state.pkl.gz", "wb") as f:
        while True:
            chunk = await request.stream.read()
            if chunk is None:
                break
            f.write(chunk)
    return text("Model State Uploaded")


@app.get("/current-best")
async def get_current_best(request):
    return json(
        {
            k:(v if not isinstance(v, datetime) else v.strftime("%c")) for k,v in 
            MongoTrials('mongo://localhost:1234/covid/jobs').best_trial.items()
            if not k.startswith("_")
        }
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)