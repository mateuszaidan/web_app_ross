from rossmann.Rossmann import Rossmann
import pickle
from flask               import Flask, request, Response
import pandas as pd 
import os


#loading data
model = pickle.load(open('model/xgb_result.pkl', 'rb'))


app = Flask(__name__)
#end point

@app.route('/rossmann/predict', methods = ['POST'])

def rossmann_predict():
    test_json = request.get_json()
    
    if test_json: #the is data
        if isinstance(test_json, dict): #Unique Example 
            test_raw = pd.DataFrame(test_json, index = [0])
        
        else: #multiples examples
            test_raw = pd.DataFrame( test_json, columns = test_json[0].keys() )

    else:
        return Reponse( '{}', status = 200, mimetype = 'application\json')
    
    #instantiate Rossmann Class 
    pipeline = Rossmann()
    
    
    #data_cleaning
    df1 = pipeline.data_cleaning( test_raw )
    
    #feature engineering 
    df2 = pipeline.feature_engineering( test_raw )
    
    #data preparation
    df3 = pipeline.data_preparation(df2)
    
    #prediction
    df_response = pipeline.get_prediction( model, test_raw, df3)
    
    return df_response
    
    
    
if __name__ ==  '__main__':
    app.run('0.0.0.0') 