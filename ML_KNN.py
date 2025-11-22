from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

print(f"After hotspot filtering: {filtered_df['SSID'].nunique()} unique SSIDs remaining")

#PIVOT THE DATA
pivot_table = filtered_df.pivot_table(
    index=['Scan_ID','Location', 'Run'],      
    columns='SSID',     
    values='Signal(dBm)', 
    aggfunc='mean',        
    fill_value=-100 
)

print(pivot_table.columns)

#save pivot
pivot_table.to_csv("Pivot_woAVG.csv")

X = pivot_table.values  #features, the parameters
y = pivot_table.index.get_level_values('Location')   #labels, what is going to be predicted

# print(X.shape)
# print(type(X))
# print(y.shape)
# print(type(y))

for i in [43,44,45,46,47,48]:
    #TRY RANDOM RANDOM STATE AND NEIGBORS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i, stratify=y)
    print(f"\nUsing random state: {i}")
    for i in [1,3,5,7,9]:
        print(f"Using {i} neighbor(s):")
        model = KNeighborsClassifier(n_neighbors=i, weights='distance')
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        print(f"Accuracy: {acc:.2%}")

#      #TRAIN ACTUAL MODEL
# print("\nTraining the model...")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43)
# model = KNeighborsClassifier(n_neighbors=3, weights='distance')
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)

# for i in range(len(y_test)):
#     print(f"Actual: {y_test[i]:<35} | Predicted: {predictions[i]:<35} | {'✓' if y_test[i] == predictions[i] else '✗'}")

#      #CREATE CONFUSION MATRIX
# class_labels = ['LaptopPriority/CurrentPeriodicals', 'MainStairs/TopPicks', 'TopPicks/Audiovisual', 'Hallway/LeftEnd', 'Stairs/RightEnd']
# cm = confusion_matrix(y_test, predictions, labels=class_labels, normalize='true')

# # Create and plot the display
# fig, ax = plt.subplots(figsize=(6, 5))  # slightly larger figure
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
# disp.plot(cmap='Blues', values_format=".2f", ax=ax, colorbar=True)

# plt.setp(ax.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor')

# plt.title('Confusion Matrix')
# plt.tight_layout()
# plt.show()

     #CROSS VAILDATION to solve the randomness
print("\nCROSS VALIDATION ...")
for i in [1, 3, 5, 7, 9]:
     model = KNeighborsClassifier(n_neighbors=i, weights='distance')
     cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

     #print(f"number of neighbors {i}")
     print(f"Accuracy for each fold: {cv_scores}")
     print(f"Mean Accuracy: {cv_scores.mean():.2%}")
     print(f"Standard Deviation: {cv_scores.std():.2%}\n")

     #VISUALIZE CROSS VALIDATION
neighbors = [1, 3, 5, 7, 9]
mean_acc = [79.33, 77.00, 74.67, 73.33, 74.00]
std_dev = [10.57, 10.35, 9.74, 8.56, 9.70]

plt.ylabel("Accuracy (%)")
plt.xlabel("Number of Neighbors")
plt.plot(neighbors, mean_acc, color='lightgreen')
plt.errorbar(neighbors, mean_acc, yerr=std_dev, fmt='o', color='seagreen')
plt.show()

#      #RANDOM FOREST
# model = RandomForestClassifier(n_estimators=100, random_state=43)
# scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# print(f"\nRANDOM FOREST...")
# print(f"Mean score:  {scores.mean():.2%}" ) 
# print(f"Standard deviation: {scores.std():.2%}")
# print(f"Scores: {scores}")


#      #KNN VS RF
# models = ['k-NN', 'Random Forest']
# accuracies = [73.33, 78.67]
# std_dev = [8.56, 9.33]
# # plt.title("Model Comparison over 5 Fold-Cross Validation")
# plt.ylabel("Accuracies (%)")
# plt.bar(models, accuracies, yerr=std_dev)
# plt.show()

#FKN HEATMAP
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
