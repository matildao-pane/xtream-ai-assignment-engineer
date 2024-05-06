import pickle
import pandas as pd
from flask import Flask, request, jsonify,render_template
import os
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("./challenge_3/xgboost_.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input data from the form
        data = request.form.to_dict()
        print(data)
        depth = float(data['depth'])
        table = float(data['table'])
        cut = int(data['cut'])
        x = float(data['x'])
        color = int(data['color'])
        clarity = int(data['clarity'])
        carat = float(data['carat'])
        
        #apply preprocessing
        carat = np.log1p(carat)  
        with open('./challenge_2/datasets/clean_data/20240506_225853_scaler.pkl', 'rb') as file:
            scaler = pickle.load(file) 
        new_data = [[carat,cut,color,clarity,depth,table,x]]   
        scaled_data = scaler.transform(new_data)
        print(new_data)
        price = model.predict(scaled_data)[0]
       
        print(price)
        exp_price= np.expm1(price) #log/exp denorm
        
        return jsonify({'price': float(exp_price)})
    return render_template('index.html')
 
if __name__ == '__main__':
    app.run(debug=True)