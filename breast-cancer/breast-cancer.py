import pandas as pd
import numpy as np

column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
                'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names = column_names)

data = data.replace(to_replace = '?', value = np.nan)

data = data.dropna(how = 'any')

print data.shape


from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data[column_names[1:10]],data[column_names[10]], test_size = 0.25, random_state = 33)

print Y_train.value_counts()
print Y_test.value_counts()



from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

from sklearn.metrics import classification_report

#ʹ��lr����Ԥ��
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,Y_train)
lr_y_predict = lr.predict(X_test)
print "lr accuracy: ", lr.score(X_test, Y_test)

#ʹ��sgdClassifier����Ԥ��
from sklearn.linear_model import SGDClassifier
sgdc = SGDClassifier()
sgdc.fit(X_train, Y_train)
sgdc_y_predict = sgdc.predict(X_test)
print "sgdc accuracy: ", sgdc.score(X_test, Y_test)

from sklearn.metrics import classification_report
print classification_report(Y_test, lr_y_predict, target_names = ["benign", "Malignant"])
print classification_report(Y_test, sgdc_y_predict, target_names = ["benign", "Malignant"])



