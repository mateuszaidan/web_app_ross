import os
import pickle
import requests
import pandas as pd
from flask             import Flask, request, Response
from rossmann.Rossmann import Rossmann


#loading model
model = pickle.load(open('model/xgb_result.pkl', 'rb'))
model = model['model_xgb_tuned']

#inicio API
app = Flask(__name__)


@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json()

    if test_json: #there is data
        if isinstance(test_json, dict): #unique example
            test_raw = pd.DataFrame(test_json, index=[0])

        else: #multiple examples
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
    
        #instantiate Rossmann Class
        pipeline = Rossmann()

        #data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        #feature engineering
        df1 = pipeline.feature_engineering(df1)
        
        #data preparation
        df1 = pipeline.data_preparation(df1)
    
        #predintion
        df_response = pipeline.get_prediction(model, test_raw, df1)


        return df_response

    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host = '0.0.0.0', port=port)