import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def plot_confusion_matrix(y,y_predict): # Plots confusion matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); 
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show() 

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")
X = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv")

Y = df['Class'].to_numpy()

transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


#Logistic Regression
parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}
parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()
logreg_cv = GridSearchCV(estimator=lr, param_grid=parameters, cv=10)
logreg_cv.fit(X_train, Y_train)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

accuracy = logreg_cv.score(X_test, Y_test)
print(accuracy)
yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

#SVM
parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()
svm_cv = GridSearchCV(estimator=svm, param_grid=parameters, cv=10)
svm_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)

accuracy = svm_cv.score(X_test, Y_test)
print(accuracy)
yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

#Decision Tree
parameters = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [2**n for n in range(1, 10)],
    'max_features': ['sqrt', 'log2'],  # Use 'sqrt' or 'log2' as valid options
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]
}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(estimator=tree, param_grid=parameters, cv=10)

tree_cv.fit(X_train, Y_train)
print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)
accuracy = tree_cv.score(X_test, Y_test)
print(accuracy)
yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

#K Nearest Neighbours
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()
KNN_cv = GridSearchCV(estimator=KNN, param_grid=parameters, cv=10)

KNN_cv.fit(X_train, Y_train)
accuracy = KNN_cv.score(X_test, Y_test)
print(accuracy)
print("tuned hpyerparameters :(best parameters) ",KNN_cv.best_params_)
print("accuracy :",KNN_cv.best_score_)
yhat = KNN_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

model_scores = {
    'Logistic Regression': logreg_cv.best_score_,
    'Decision Tree': tree_cv.best_score_,
    'SVM': svm_cv.best_score_,
    'KNN': KNN_cv.best_score_
}

# Extract model names and their best scores
model_names = list(model_scores.keys())
scores = list(model_scores.values())

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(model_names, scores, color='skyblue')
plt.title('Model Accuracy Comparison')
plt.xlabel('Classification Model')
plt.ylabel('Best Accuracy Score')
plt.ylim(0, 1)  # Since accuracy ranges from 0 to 1
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()