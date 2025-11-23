import matplotlib.pyplot as plt
import pandas as pd

data ={
    'Random State': [43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 
                     46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48],
    'Neighbors': [1, 3, 5, 7, 9] * 6,
    'Accuracy': [86.40, 88.00, 86.40, 86.80, 84.80,
                89.60, 89.60, 87.20, 87.20, 84.80,
                88.00, 88.00, 90.00, 89.60, 87.20,
                88.80, 84.00, 82.40, 82.40, 81.60,
                86.40, 84.80, 84.00, 80.00, 80.80,
                82.40, 81.60, 83.20, 81.60, 82.40]
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