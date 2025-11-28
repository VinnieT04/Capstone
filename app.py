from flask import Flask, render_template, request, jsonify, render_template
import joblib
import pandas as pd

app == Flask(__name__)

#Load models
knn = joblib.load('knn_model.pkl')
rf = joblib.load('rf_model.pkl')
df = pd.read_csv('Pivot_woAVG.csv')

X = df.iloc[:, 3:].values #features, the parameters
y = df['Location'].values

@app.route('/')
def home():
    request.get_json

    
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
if __name__ == '__main__':
    app.run(debug=True, port=500)