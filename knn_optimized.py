import numpy as np
import pandas as pd

dataset = pd.read_csv("breast_cancer.csv")

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.neighbors import KNeighborsClassifier

err_rate = []

for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    err_rate.append(np.mean(y_pred!= y_test))

optimal_k = k = err_rate.index(min(err_rate)) + 1

print("Minimun error  = ",min(err_rate)*100,"%")
print("Value of k for minimum error = ",optimal_k)

from sklearn.metrics import accuracy_score

knn1 = KNeighborsClassifier(n_neighbors=optimal_k)
knn1.fit(x_train,y_train)
y_pred1 = knn1.predict(x_test)

print("accuracy at k = ",optimal_k," is ", accuracy_score(y_pred1,y_test)*100,"%")