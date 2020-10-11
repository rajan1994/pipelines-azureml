import json
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from azureml.core.model import Model
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
import pandas as pd
from utils import mylib

def init():
    global model
    model_path = Model.get_model_path('bank-model')

    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # For demonstration purposes only
    print(mylib.get_alphas())

input_sample = np.array([[33,2,5,76,1,0,0,2,0,0,0,0,2]])
output_sample = np.array([0])

@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        cols_when_model_builds = model.get_booster().feature_names
        predict_data = pd.DataFrame(data,columns=cols_when_model_builds)
        result = model.predict(predict_data)
        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
