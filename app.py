from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import json
from ARC_functions import ARC_main  # the main function which takes task as argument and returns final predictions matrices. 
 


app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_task', methods=['POST'])
def process_task():
    print('Loading in process...')
    data = request.json
    task_name = data['task_name']
    tasks = []
    tasks.append(task_name)
    Data = ARC_main(tasks)

    pred_array = Data[0]
    # pred_array = pred_array.tolist()
    pred_array = json.dumps(pred_array)
    # pred_array = []
    # for pair in test_data['test']:
    #     onp = (pair['output'])
    #     pred_array.append(onp)
       
    print('Done.')
    return jsonify(pred_array=pred_array)

# @app.route('/test')
# def test():
#     return "Hello, World!"

if __name__ == '__main__':
    print("Starting AO ARC-AGI app")
    app.run(debug=True, host='0.0.0.0', port=5000)
