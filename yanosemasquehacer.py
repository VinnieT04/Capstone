import matplotlib.pyplot as plt
import pandas as pd

data ={
    'Random State': [43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 
                     46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48],
    'Neighbors': [1, 3, 5, 7, 9] * 6,
    'Accuracy': [89.33, 86.67, 82.67, 82.67, 80.00,
                86.67, 82.67, 84.00, 84.00, 81.33,
                88.00, 85.33, 86.67, 81.33, 80.00,
                85.33, 81.33, 80.00, 81.33, 80.00,
                92.00, 90.67, 86.67, 89.33, 86.67,
                88.00, 86.67, 86.67, 86.67, 85.33]
}
df = pd.DataFrame(data)

#group by k
df = df.pivot(index='Neighbors', columns='Random State', values='Accuracy')
df.plot.bar()

#group by random state
# df = df.pivot(index='Random State', columns='Neighbors', values='Accuracy')
# df.plot.bar()

#plt.title('KNN Accuracy by Random State and Number of Neighbors')
plt.xlabel('Neighbors')
# plt.xlabel('Number of Neigbors')
plt.ylabel('Accuracy(%)')
plt.show()