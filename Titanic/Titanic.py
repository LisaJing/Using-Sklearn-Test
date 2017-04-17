import pandas as pd

#读取Titanic 乘船数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#print titanic.head()
#print titanic.info()

#pre-processing
#feature-selection
X = titanic[['pclass', "sex", "age"]]
Y = titanic['survived']
print X.info()
print "-----------------------"
#print Y

#填充缺失的数据
X['age'].fillna(X['age'].mean(), inplace = True)
print X.info()

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 33)


#特征转换
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False)
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
print vec.feature_names_

X_test = vec.transform(X_test.to_dict(orient = 'record'))


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
dtc_y_predict = dtc.predict(X_test)


print "Decision Tree Classifier Accuracy: " , dtc.score(X_test, Y_test)
from sklearn.metrics import classification_report
print classification_report(  Y_test, dtc_y_predict, target_names = ['died','survived'])
