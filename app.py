from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import random

app = Flask(__name__)

#Load models
knn = joblib.load('knn_model.pkl')
rf = joblib.load('rf_model.pkl')
df = pd.read_csv('Pivot_woAVG.csv') #change to one with different values

X = df.iloc[:, 3:] #select the columns starting from the 3rd one
y = df['Location'].values

print(f"X shape on load: {X.shape}") # Should be (N, M-3) where N is rows, M is total columns
print(f"y shape on load: {y.shape}") # Should be (N,)
print(f"Example feature column name: {X.columns[0]}")

@app.route('/')
def home():
    randomSample = random.sample(range(len(X)), 5)
    print(f"Sending samples: {randomSample}")
    return render_template('index.html', Sample= randomSample)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sample = int(data['sample_id']) #get the id of the sample the user selected
    x_sample = X.iloc[sample].values.reshape(1, -1)

    actual_zone = y[sample]

    #predictions
    knn_pred = knn.predict(x_sample) 
    rf_pred = rf.predict(x_sample)

    #verify if predictions are correct
    correct_knn = bool(knn_pred[0] == actual_zone) 
    correct_rf = bool(rf_pred[0] == actual_zone)
    
    return jsonify({
    'actual': actual_zone,
    'knn_prediction': knn_pred[0],  
    'rf_prediction': rf_pred[0],    
    'knn_correct': correct_knn,
    'rf_correct': correct_rf
})

if __name__ == '__main__':
    app.run(debug=True, port=5000)