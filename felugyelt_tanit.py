# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 20:42:55 2020

@author: IfritR
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, plot_confusion_matrix,plot_roc_curve
from sklearn import  svm
from sklearn.preprocessing import LabelEncoder, StandardScaler,Binarizer
from urllib.request import urlopen
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate,cross_val_score
#pip install scikit-plot!!!!!
import scikitplot as skplt

#Load datas from internet, read values and attribute names with np
url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"


#Load datas from internet, read values and attribute names with pandas
wine = pd.read_csv(url, sep = ';')
wine_ranges = (0, 6, 10)       
group_of_names = ['rossz', 'jó']
wine['quality'] = pd.cut(wine['quality'], bins = wine_ranges, labels = group_of_names) 
wine['quality'].unique()



#print the bad and good qualities number
print(wine['quality'].value_counts())
label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])
sns.countplot(wine['quality'])



#Drop the output variable from input variables
X = wine.drop('quality', axis=1)
X2=X.to_numpy(copy=True)
#Target variable
y = wine['quality']
y2=y.to_numpy(copy=True)
# Partitioning into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X2,y2, test_size=0.3, 
                                shuffle = True, random_state=2020)


# Applying Standard scaling to get optimized result
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Support vector classifier
clf = svm.SVC()
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)



# Fitting logistic regression
logreg_classifier = LogisticRegression(solver='liblinear',max_iter=500,random_state=2020)
logreg_classifier.fit(X_train,y_train)
score_train_logreg = logreg_classifier.score(X_train,y_train)
score_test_logreg = logreg_classifier.score(X_test,y_test)
ypred_logreg = logreg_classifier.predict(X_test)
yprobab_logreg = logreg_classifier.predict_proba(X_test)
scores = cross_val_score(logreg_classifier, X, y, cv=5)
print("Logistic regression scores:")
print(scores)

# Fitting neural network classifier
neural_classifier = MLPClassifier(hidden_layer_sizes=(100,100,100),activation='relu',max_iter=5000)
neural_classifier.fit(X_train,y_train)
score_train_neural = neural_classifier.score(X_train,y_train)
score_test_neural = neural_classifier.score(X_test,y_test)
ypred_neural = neural_classifier.predict(X_test)
yprobab_neural = neural_classifier.predict_proba(X_test)
scores = cross_val_score(neural_classifier, X, y, cv=5)
print("MLP classifier scores:")
print(scores)

#Fitting random forest classifier
rfc = RandomForestClassifier(n_estimators=200,random_state=2020) 
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
probab_rfc=rfc.predict_proba(X_test)
score_train_rfc = rfc.score(X_train,y_train)
score_test_rfc = rfc.score(X_test,y_test)
scores = cross_val_score(rfc, X, y, cv=5)
print("Random Forest Classifier")
print(scores)

#Logreg plot confusion matrix, roc_curve | print classification report and confusion matrix
print("Logreg")
print(classification_report(y_test, ypred_logreg))
print(confusion_matrix(y_test, ypred_logreg))
plot_confusion_matrix(logreg_classifier, X_train, y_train, display_labels =group_of_names[::-1] );
plot_roc_curve(logreg_classifier, X_test, y_test);

#MLP classifier plot confusion matrix, roc_curve | print classification report and confusion matrix
print("MLP classifier")
print(classification_report(y_test, ypred_neural))
print(confusion_matrix(y_test, ypred_neural))
plot_confusion_matrix(neural_classifier, X_train, y_train, display_labels =group_of_names[::-1] );
plot_roc_curve(neural_classifier, X_test, y_test);

#Random Forest classifier
print("Random Forest classifier")
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc)) 
plot_confusion_matrix(rfc, X_train, y_train, display_labels =group_of_names[::-1] );
plot_roc_curve(rfc, X_test, y_test);

#Decision tree
depth =4
class_tree = DecisionTreeClassifier(max_depth=depth)
class_tree.fit(X_train, y_train)
score_train = class_tree.score(X_train, y_train)
score_test = class_tree.score(X_test, y_test)
y_pred_gini = class_tree.predict(X_test)
fig = plt.figure(8,figsize = (16,10),dpi=100);
plot_tree(class_tree, feature_names = wine.columns, 
               class_names = group_of_names,
               filled = True, fontsize = 6)

plot_roc_curve(clf, X_test, y_test)


# Plotting ROC curve
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, yprobab_logreg[:,1]);
roc_auc_logreg = auc(fpr_logreg, tpr_logreg);

fpr_neural, tpr_neural, _ = roc_curve(y_test, yprobab_neural[:,1]);
roc_auc_neural = auc(fpr_neural, tpr_neural);

fpr_rfc, tpr_rfc, _ = roc_curve(y_test, probab_rfc[:,1]);
roc_auc_rfc = auc(fpr_rfc, tpr_rfc);

plt.figure(10);
lw = 2;
plt.plot(fpr_logreg, tpr_logreg, color='red',
         lw=lw, label='Logistic regression (AUC = %0.2f)' % roc_auc_logreg);
plt.plot(fpr_neural, tpr_neural, color='blue',
         lw=lw, label='MLPClassifier (AUC = %0.2f)' % roc_auc_neural);
plt.plot(fpr_rfc, tpr_rfc, color='green',
         lw=lw, label='Rfc (AUC = %0.2f)' % roc_auc_rfc);
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic curve');
plt.legend(loc="lower right");

# Plot alcohol boxplot
plt.figure(11);
sns.boxplot(y=wine['alcohol'],x=wine['quality'])
plt.show()


#Multiple classes

#Read the data
raw_data = urlopen(url);
attribute_names=raw_data.readline().decode('utf-8').split(';')
data = np.loadtxt(raw_data, delimiter=";")
del raw_data


#Split data to input,output variables
X3 = data[:,0:11]
y3 = data[:,11]
X3_train, X3_test, y3_train, y3_test = train_test_split(X3,y3, test_size=0.3, 
                                shuffle = True, random_state=2020)
#normalizing datas
sc = StandardScaler()
X3_train = sc.fit_transform(X3_train)
X3_test = sc.fit_transform(X3_test)

#Logreg for multiclass
logreg_multiclassifier = LogisticRegression(solver='newton-cg',penalty='l2',max_iter=500)
logreg_multiclassifier.fit(X3_train,y3_train)
score_train_logregmulti = logreg_multiclassifier.score(X3_train,y3_train)
score_test_logregmulti = logreg_multiclassifier.score(X3_test,y3_test)
ypred_logregmulti = logreg_multiclassifier.predict(X3_test)
yprobab_logregmulti = logreg_multiclassifier.predict_proba(X3_test)
scoresmulti = cross_val_score(logreg_multiclassifier, X3, y3, cv=5)
print(scoresmulti)

#Plot confusion matrix and print details
print("Logregmulti")
print(classification_report(y3_test, ypred_logregmulti,zero_division=0))
print(confusion_matrix(y3_test, ypred_logregmulti))
plot_confusion_matrix(logreg_multiclassifier, X3_train, y3_train)

#pip install scikit-plot!!!!
skplt.metrics.plot_roc(y3_test,yprobab_logregmulti,figsize=(16,10))


