from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS, cross_origin
from flask_session import Session
from cachelib.file import FileSystemCache
import numpy as np
import json
import logging.config
from ARC_functions import (
    ARC_main,
    setup_agent,
)  # the main function which takes task as argument and returns final predictions matrices.


app = Flask(__name__)
# Using cachelib for session management since it seems like the simplest option in flask-session
app.config["SESSION_TYPE"] = "cachelib"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_CACHELIB"] = FileSystemCache(threshold=100, cache_dir="./sessions")
CORS(app, supports_credentials=True)
Session(app)


# logging.config.dictConfig({"disable_existing_loggers": False})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_task", methods=["POST"])
@cross_origin(supports_credentials=True)
def process_task():
    # if "agent" not in session:
        # session["agent"] = setup_agent()
    session["agent"] = setup_agent()
    print("Loading in process...")
    data = request.json
    task_name = data["task_name"]
    tasks = []
    tasks.append(task_name)
    agent = session["agent"]
    # print("AGENT CURRENTLY AT: "+str(agent.state))
    Data = ARC_main(agent, tasks)

    pred_array = Data[0]
    # pred_array = pred_array.tolist()
    pred_array = json.dumps(pred_array)
    # pred_array = []
    # for pair in test_data['test']:
    #     onp = (pair['output'])
    #     pred_array.append(onp)

    print("Done.")
    return jsonify(pred_array=pred_array)


# @app.route('/test')
# def test():
#     return "Hello, World!"

if __name__ != "__main__":
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == "__main__":
    print("Starting AO ARC-AGI app")
    app.run(debug=True, host="0.0.0.0", port=5000)
