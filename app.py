from flask import Flask, render_template, request, jsonify, render_template
import joblib
import pandas as pd
import random

app == Flask(__name__)

#Load models
knn = joblib.load('knn_model.pkl')
rf = joblib.load('rf_model.pkl')
df = pd.read_csv('Pivot_woAVG.csv')

X = df.iloc[:, 3:] #select the columns starting from the 3rd one
y = df['Location'].values

@app.route('/')
def home():
    randomSample = random.df(range(len(X)), 5)

    
    return render_template('index.html', Sample= randomSample)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sample = data['sample_id'] #get the id of the sample the user selected
    x_sample = X[sample].reshape(1,-1)
    actual_zone = y[sample]

    #predictions
    knn_pred = knn.predict(x_sample) 
    rf_pred = rf.predict(x_sample)

    #verify if predictions are correct
    correct_knn = (knn_pred == actual_zone) 
    correct_rf = (rf_pred == actual_zone)
    
    return jsonify({
    'actual': actual_zone,
    'knn_prediction': knn_pred[0],  # ← Add [0] here
    'rf_prediction': rf_pred[0],    # ← Add [0] here
    'knn_correct': correct_knn,
    'rf_correct': correct_rf
})

if __name__ == '__main__':
    app.run(debug=True, port=500)