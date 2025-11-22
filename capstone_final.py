#hola mundo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict, cross_val_score
import matplotlib.pyplot as plt
import pandas as pd

#LOAD DATA
df = pd.read_csv('Library_woAVG.csv')

#DROP HIDDEN NETWORKS and HOTSPOTS
df = df[df['SSID'] != 'Hidden_Network']
hotspot = ['iPhone', 'Android', 'Galaxy', 'Xiaomi', 'Pixel', 'Redmi', 'Brian', 'æ©æ©æ©æ©ç',
            'DIRECT-', 'Chris', 'Personal', 'Skyler', 'phone', 'Benben', 'Loser', 'årrrrrrr' ,'å²']

#count appereances
appearances = df.groupby('SSID')['Location'].nunique()

#keep ssids that appear in 70% of locations
threshold = 0.7*df['Location'].nunique()
common_ssids = appearances[appearances >= threshold].index

filtered_df = df[df['SSID'].isin(common_ssids)].copy() 
for keyword in hotspot:
    filtered_df = filtered_df[~filtered_df['SSID'].str.contains(keyword, case=False, na=False)]

#print(f"After hotspot filtering: {filtered_df['SSID'].nunique()} unique SSIDs remaining")

#PIVOT THE DATA
pivot_table = filtered_df.pivot_table(
    index=['Scan_ID','Location', 'Run'],      
    columns='SSID',     
    values='Signal(dBm)', 
    aggfunc='mean',        
    fill_value=-100 
)

#print(pivot_table.columns)

X = pivot_table.values  #features, the parameters
y = pivot_table.index.get_level_values('Location')   #labels, what is going to be predicted

# print(X.shape)
# print(type(X))
# print(y.shape)
# print(type(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43, stratify=y)

                    #FINAL MODELS
#K-NEAREST NEIGHBORS
knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn.fit(X_train, y_train)
kNN_predictions = cross_val_predict(knn, X, y, cv=5)
kNN_scores = cross_val_score(knn ,X, y, cv=5, scoring='accuracy')

print(f"Number of predictions: {len(kNN_predictions)}")
print(f"First 10 predictions: {kNN_predictions[:10]}")
print(f"Type of data: {type(kNN_predictions[0])}")

print(f"k-Nearest Neighbors : {kNN_scores.mean():.2%} ± {kNN_scores.std():.2%}")
print(f"Scores: {kNN_scores}")

#RANDOM FOREST
rf = RandomForestClassifier(n_estimators=100, random_state=43)
RF_predictions = cross_val_predict(rf, X, y, cv=5)
RF_scores = cross_val_score(rf ,X, y, cv=5, scoring='accuracy')

print(f"Number of predictions: {len(RF_predictions)}")
print(f"First 10 predictions: {RF_predictions[:10]}")
print(f"Type of data: {type(RF_predictions[0])}")

print(f"Random Forest : {RF_scores.mean():.2%} ± {RF_scores.std():.2%}")
print(f"Scores: {RF_scores}")

#confusion matrices
class_labels = ['LaptopPriority/CurrentPeriodicals', 'MainStairs/TopPicks', 'TopPicks/Audiovisual', 'Hallway/LeftEnd', 'Stairs/RightEnd']
cm_knn = confusion_matrix(y, kNN_predictions, normalize='true')
cm_rf = confusion_matrix(y, RF_predictions, normalize='true')

fig, ax = plt.subplots(figsize=(8, 6))  # slightly larger figure
disp = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=class_labels)
disp.plot(cmap='Greens', values_format=".2f", ax=ax, colorbar=True)

plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
plt.tight_layout()

plt.show()

#comparison graph
models = ['k-NN', 'Random Forest']
accuracies = [82.60, 87.00]
std_dev = [9.97, 9.63]
# plt.title("Model Comparison over 5 Fold-Cross Validation")
plt.ylabel("Accuracies (%)")
plt.bar(models, accuracies, yerr=std_dev, color='lightgreen')
plt.show()
