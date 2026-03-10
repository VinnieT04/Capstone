# Indoor Positioning using Wi-Fi Signals
>Capstone Project · NTUST-UPTP

A Capstone project that utilizes Wi-Fi signal strength data from access points across different locations in the NTUST's Library and the training of machine learning models to classify or predict the user's position based on their WiFi environment.

## Project Structure
| File | Purpose| Description|
|------|--------|------------|
|`scan_woAVG.py` | Data Collection | Scans nearby APs 20 times per location, saves to 'new_Scans.csv' |
|`demoSamples.py` | Data Preparation | Filters to known, reliable SSIDs, pivots data into 'demoScans.csv' |
|`testRnadomState` | k-NN tuning | Tests k values (1-9) and random seeds via cross-validation |
|`bestRF.py` | RF tuning | GridSearchCV over random forest hyperparameters |
|`app.py` | Web App | Flask demo | Pick a scan, see both models predict live |
|`yes.py` | Evaluation | Per-zone accuracy report + confusion matrix |

## Model Performance
Both models. k-Nearest Neighbors and Random Forest, are evaluated with **5-fold cross-validation** on 500 samples (5 locations x 5 runs x 20 scans)

| Model | Accuracy | Std Dev | Key Parameters |
|-------|----------|---------|----------------|
| k-Nearest Neighbors | 82.6% | ± 8.4% | k=5, manhattan distance, distance weights |
| Random Forest | 83.0% | ± 9.4% | 500 trees, max\_depth=10, random\_state=45 |

> Cross-validation was used instead of a single train/test split because with 500 samples, a 75/25 split only gives ~125 test samples — small enough that 2–3 wrong predictions changes accuracy by ~1–2%.

## How to Use
### Prerequisites
The following dependencies are needed on Python 3.x:
```bash
pip install flask scikit-learn matplotlib pandas numpy pywifi
```

### Run the Web App
```bash
python app.py
```
Then open your browser and go to `http://127.0.0.1:5000`

### Retrain the models
```bash
python capstone_final.py
```
>This will process the fingerprint data 'Library_woAVG.csv' and save updated `knn_model.pkl` and `rf_model.pkl` files

### Collect New Scan Data
```bash
python scan_woAVG.py
```
> Run this **at each physical zone** of the library. Edit the `scan_aps()` call at the bottom to set the location name and run number before scanning.

## About the Data & Samples
The raw CSV (`Library_woAVG.csv`) has ~44,000 rows, but each row is a single network detection — not an independent sample. The pivot table collapses each scan into **one fingerprint vector** of 26 SSID signal strengths, giving 500 true samples:

```
5 locations × 5 runs × 20 scan IDs = 500 samples
```

Missing SSIDs in a scan are filled with **-100 dBm** (no signal).
