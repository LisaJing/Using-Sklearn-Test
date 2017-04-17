import pandas as pd

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

X = titanic[["pclass", "sex", "age"]]
Y = titanic["survived"]

X['age'].fillna(X['age'].mean(), inplace = True)


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 33)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False)
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
X_test = vec.transform(X_test.to_dict(orient = 'record'))

from sklearn.metrics import classification_report

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
dtc_y_predict = dtc.predict(X_test)
print "DTC Accuracy:", dtc.score(X_test, Y_test)
print classification_report(Y_test, dtc_y_predict, target_names = ["died", "survived"])


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
rfc_y_predict = rfc.predict(X_test)
print "RFC Accuracy:", rfc.score(X_test, Y_test)
print classification_report(Y_test, rfc_y_predict, target_names = ["died", "survived"])

#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, Y_train)
gbc_y_predict = gbc.predict(X_test)
print "GBC Accuracy:", gbc.score(X_test, Y_test)
print classification_report(Y_test, gbc_y_predict, target_names = ["died", "survived"])

