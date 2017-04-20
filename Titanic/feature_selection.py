import pandas as pd

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

print titanic.shape


#分离数据特征与预测目标
Y = titanic['survived']
X = titanic.drop(['row.names','name','survived'], axis = 1)

X['age'].fillna(X['age'].mean(), inplace = True)
X.fillna('UNKNOWN', inplace = True)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 33)

#对类别性质进行向量化
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
X_test = vec.transform(X_test.to_dict(orient = 'record'))

#print len(vec.feature_names_)
#print vec.feature_names_

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy')
dt.fit(X_train, Y_train)
dt_y_predict = dt.predict(X_test)
print "Accuracy: ", dt.score(X_test, Y_test)

from sklearn.metrics import classification_report
print classification_report(Y_test, dt_y_predict, target_names = ['died','survived'])


#feature_selection
from sklearn import feature_selection
#筛选前20%的特征，使用相同配置的决策树模型进行预测
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile = 20)
X_train_fs = fs.fit_transform(X_train, Y_train)
X_test_fs = fs.transform(X_test)
dt.fit(X_train_fs, Y_train)
dt_fs_y_predict = dt.predict(X_test_fs)
print "Accuracy: ", dt.score(X_test_fs, Y_test)

from sklearn.metrics import classification_report
print classification_report(Y_test, dt_fs_y_predict, target_names = ['died','survived'])


#交叉验证
from sklearn.cross_validation import cross_val_score
import numpy as np
percentiles = range(1, 100, 2)
results = []

for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile = i)
    X_train_fs = fs.fit_transform(X_train, Y_train)
    scores = cross_val_score(dt, X_train_fs, Y_train, cv = 5)
    results = np.append(results, scores.mean())
print results


opt = np.where(results == results.max())[0]
print "Optimal number of features %d " %(percentiles[opt])


#绘图
import pylab as pl
pl.plot(percentiles, results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

from sklearn import feature_selection 
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile = 7)
X_train_fs = fs.fit_transform(X_train, Y_train)
dt.fit(X_train_fs, Y_train)
X_test_fs = fs.transform(X_test)
print "Accuracy:", dt.score(X_test_fs, Y_test)

