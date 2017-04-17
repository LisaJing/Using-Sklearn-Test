from sklearn.datasets import load_iris
iris = load_iris()
print iris.data.shape
print iris.DESCR

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size = 0.25, random_state = 33)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#using kNN
print "KNN"
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier()
knc.fit(X_train, Y_train)
knc_y_predict = knc.predict(X_test)
print "Accuracy: " , knc.score(X_test, Y_test)
from sklearn.metrics import classification_report
print classification_report(Y_test, knc_y_predict, target_names = iris.target_names)



#using LR
print "LR"
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
lr_y_predict = lr.predict(X_test)
print "Accuracy: " , lr.score(X_test, Y_test)
print classification_report(Y_test, lr_y_predict, target_names = iris.target_names)


#using SDG
print "SDG"
from sklearn.linear_model import SGDClassifier
sgdc = SGDClassifier()
sgdc.fit(X_train, Y_train)
sgdc_y_predict = sgdc.predict(X_test)
print "Accuracy: ", sgdc.score(X_test, Y_test)
print classification_report(Y_test, sgdc_y_predict, target_names = iris.target_names)


#using SVM
print "SVM"
from sklearn.svm import LinearSVC
lsvc = LinearSVC()
lsvc.fit(X_train, Y_train)
lsvc_y_predict = lsvc.predict(X_test)
print "Accuracy: ", lsvc.score(X_test, Y_test)
print classification_report(Y_test, lsvc_y_predict, target_names = iris.target_names)


