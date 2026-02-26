# Indoor Positioning using Wi-Fi Signals
A Capstone proejct that utilizes Wi-Fi signal strength data from access points across different locations in the NTUST's Library and the training of machine learnign models to classify or predict the user's position based on their WiFi envirionment.

The two models that are used are:
- Random Forest
- k-Nearest Neighbors

Flask web application is used as the front-end interface for real-time location prediction

## Prerequisites
The following dependencies are needed on Python 3.x:
- Flask
- Scikit-learn
- Matplotlib
- Pandas
- Numpy

## Run the Web App
```bash
python app.py
```
Then open your browser and go to `http://127.0.0.1:5000`

## Retrain the models
```bash
python capstone_final.py
```

This will process the fingerprint data and save updated `knn_model.pkl` and `rf_model.pkl` files

