#hola mundo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict, cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#LOAD DATA
df = pd.read_csv('Library_woAVG.csv')

#DROP HIDDEN NETWORKS and HOTSPOTS
df = df[df['SSID'] != 'Hidden_Network']
hotspot = ['iPhone', 'Android', 'Galaxy', 'Xiaomi', 'Pixel', 'Redmi', 'Brian', 'æ©æ©æ©æ©ç',
            'DIRECT-', 'Chris', 'Personal', 'Skyler', 'phone', 'Benben', 'Loser', 'årrrrrrr' ,'å²',
            'èæé²', 'ð«â­ï¸â¨â¡ï¸ð«', 'XxxxxxxX', 'general_7006', 'vivo Y38 5G', 'Calvin']

#count appereances
appearances = df.groupby('SSID')['Location'].nunique()

#keep ssids that appear in 70% of locations
threshold = 0.7*df['Location'].nunique()
common_ssids = appearances[appearances >= threshold].index

filtered_df = df[df['SSID'].isin(common_ssids)].copy() 
for keyword in hotspot:
    filtered_df = filtered_df[~filtered_df['SSID'].str.contains(keyword, case=False, na=False)]

#PIVOT THE DATA
pivot_table = filtered_df.pivot_table(
    index=['Scan_ID','Location', 'Run'],      
    columns='SSID',     
    values='Signal(dBm)', 
    aggfunc='mean',        
    fill_value=-100 
)

print(f"Number of SSIDs (features): {pivot_table.shape[1]}")

#keep track of the SSIDs to see if they are reliable and save them in a new csv
# print(pivot_table.columns)
pivot_table.to_csv("Pivot_woAVG.csv")

X = pivot_table.values  #features, the parameters
y = pivot_table.index.get_level_values('Location')   #labels, what is going to be predicted
                    
                    #FINAL MODELS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=45, stratify=y)

#K-NEAREST NEIGHBORS
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan')
knn.fit(X_train, y_train)
kNN_predictions = cross_val_predict(knn, X, y, cv=5)
kNN_scores = cross_val_score(knn ,X, y, cv=5, scoring='accuracy')

print(f"k-Nearest Neighbors : {kNN_scores.mean():.2%} ± {kNN_scores.std():.2%}")
print(f"Scores: {kNN_scores}")

#RANDOM FOREST
rf = RandomForestClassifier(n_estimators=500, random_state=45, min_samples_leaf=1, max_depth=10, min_samples_split=2)
rf.fit(X_train, y_train)
RF_predictions = cross_val_predict(rf, X, y, cv=5)
RF_scores = cross_val_score(rf ,X, y, cv=5, scoring='accuracy')

print(f"Random Forest : {RF_scores.mean():.2%} ± {RF_scores.std():.2%}")
print(f"Scores: {RF_scores}")

#confusion matrices
class_labels = sorted(y.unique())
cm_knn = confusion_matrix(y, kNN_predictions, labels=class_labels, normalize='true')
cm_rf = confusion_matrix(y, RF_predictions, labels=class_labels, normalize='true')

fig, ax = plt.subplots(figsize=(8, 6))  # slightly larger figure
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=class_labels)
disp.plot(cmap='Greens', values_format=".2f", ax=ax, colorbar=True)

plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
plt.tight_layout()

plt.show()

#comparison graph
models = ['k-NN', 'Random Forest']
accuracies = [82.60, 83.00]
std_dev = [8.38, 9.40]
plt.ylabel("Accuracies (%)")
plt.bar(models, accuracies, yerr=std_dev, color='lightgreen')
plt.show()

#heatmap
location_avg = pivot_table.groupby(level='Location').mean()
plt.figure(figsize=(14, 6))
sns.heatmap(location_avg, 
            cmap='RdYlGn_r',  # Red=weak, Green=strong
            center=-70,       # Middle value for color scale
            annot=False,      # Set True if you want numbers in cells
            cbar_kws={'label': 'Signal Strength (dBm)'})
plt.xlabel('SSID')
plt.ylabel('Location')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#to save the models
import joblib

joblib.dump(knn, 'knn_model.pkl')
joblib.dump(rf, 'rf_model.pkl')

print("Models saved successfully!")
