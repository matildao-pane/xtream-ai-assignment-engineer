import pickle
import joblib
import pandas as pd
from flask import Flask, request, jsonify,render_template
import os

app = Flask(__name__)

# Load the trained model
#model = joblib.load(os.getcwd()+'\challenge_3\XGBoost_.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input data from the form
        data = request.form.to_dict()
        print(data)
        #depth = float(data['depth'])
        #table = float(data['table'])
        #cut = int(data['cut'])
        #color = int(data['color'])
        #clarity = int(data['clarity'])
        #carat = float(data['carat'])
        #price = model.predict([[cut]])[0]
        price = 44444
        return jsonify({'price': price})
    return render_template('index.html')
 
if __name__ == '__main__':
    app.run(debug=True)