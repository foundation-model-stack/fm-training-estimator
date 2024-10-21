# Standard
from typing import Any
import json
import logging

# Third Party
from fastapi import Body, FastAPI
import fire
import uvicorn

# Local
from .core import run


def api(data_path, model_path):
    app = FastAPI()

    @app.post("/api/estimate")
    def estimate(config: Any = Body()):
        conf = json.loads(config)
        output = run(conf, data_path, model_path)
        # this default float business is needed to deal with numpy.float32
        # types present in the output json which don't serialize out of the box
        return json.dumps(output, default=float)

    return app


def run_api(data_path=None, model_path=None, port=3000):

    app = api(data_path, model_path)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(run_api)
