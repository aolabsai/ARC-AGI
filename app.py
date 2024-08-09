from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
from ARC_aolabs2 import ARC_main



app = Flask(__name__)
CORS(app)

@app.route('/process_task', methods=['POST'])
def process_task():
    data = request.json
    task_name = data['task_name']
    tasks = []
    tasks.append(task_name)
    Data = ARC_main(tasks)


    

    pred_array = Data[0]
    
    pred_array = json.dumps(pred_array)
    print('trainin Done')
       
   
    return jsonify(pred_array=pred_array)

if __name__ == '__main__':
    app.run(debug=True)

