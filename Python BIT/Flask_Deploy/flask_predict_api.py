import pickle #untuk ambil model
import numpy as np
from flasgger import Swagger
from flask import Flask, request #to get input from user
import pandas as pd
import os

path = os.path.dirname(os.path.realpath(__file__))
path = path.replace("\\", "/")


with open(path + '/rf.pkl','rb') as model_file:
     model = pickle.load(model_file)

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict')
def predict_iris():
    """Example endpoint returning a prediction of iris
    ---
    parameters:
    - name: s_length
      in: query
      type: number
      required: true
    - name: s_width
      in: query
      type: number
      required: true
    - name: p_length
      in: query
      type: number
      required: true
    - name: p_width
      in: query
      type: number
      required: true
    responses:
      200: 
        description: ok
    """
    s_length = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_length = request.args.get("p_length")
    p_width = request.args.get("p_width")
    prediction = model.predict(np.array([[s_length,s_width,p_length,p_width]]))
    return str(prediction)

@app.route('/predict_file', methods = ["POST"])
def predict_iris_csv():
    """Example file endpoint returning a prediction of iris
    ---
    parameters:
    - name: input_file
      in: formData
      type: file
      required: true
    responses:
      200: 
        description: ok
    """
    input_data = pd.read_csv(request.files.get("input_file"))
    prediction = model.predict(input_data)
    return str(list(prediction))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7002)

#panggilnya http://localhost:7002/apidocs/