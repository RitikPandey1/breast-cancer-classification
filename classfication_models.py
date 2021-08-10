import numpy as np
import pandas as pd

dataset = pd.read_csv("breast_cancer.csv")

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.metrics import confusion_matrix, accuracy_score
def accuCheck(y_pred):
    cm = confusion_matrix(y_test,y_pred)
    print(cm)
    print(accuracy_score(y_test,y_pred)*100)

# Default Logistic regression model
from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression(random_state=1)
lgr.fit(x_train,y_train)
y1_pred = lgr.predict(x_test)

print("--- Logistic regression ---")
accuCheck(y1_pred)

# Default KNN model
from  sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
y2_pred = knn.predict(x_test)
print("--- KNN ---")
accuCheck(y2_pred)

# linear svm model
from sklearn.svm import SVC
svm = SVC(kernel='linear',random_state=0)
svm.fit(x_train,y_train)
y3_pred = svm.predict(x_test)
print("--- Linear SVM ---")
accuCheck(y3_pred)

# non linear svm
from sklearn.svm import SVC
svm = SVC(kernel='rbf' , random_state=0)
svm.fit(x_train,y_train)
y4_pred = svm.predict(x_test)
print("--- Non Linear SVM ---")
accuCheck(y4_pred)

# naive bayes 
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
y5_pred = nb.predict(x_test)
print("--- Naive bayes ---")
accuCheck(y5_pred)

# decision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=0)
dt.fit(x_train,y_train)
y6_pred = dt.predict(x_test)
print("--- Decision Tree ---")
accuCheck(y6_pred)

# random forest 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10,random_state=0)
rf.fit(x_train,y_train)
y7_pred = rf.predict(x_test)
print("--- Random forest ---")
accuCheck(y7_pred)


