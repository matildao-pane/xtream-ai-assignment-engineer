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
        price = model.predict([[carat,cut,color,clarity,depth,table,x]])[0]
        exp_price= np.expm1(price)
        return jsonify({'price': float(exp_price)})
    return render_template('index.html')
 
if __name__ == '__main__':
    app.run(debug=True)