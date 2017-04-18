from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset = 'all')
print len(news.data)
print "\n"
print news.data[0]


#分割新闻文本
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(news.data, news.target, test_size = 0.25, random_state = 33)

#使用naive Bayes进行预测

#特征抽取
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

#导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, Y_train)
mnb_y_predict = mnb.predict(X_test)

from sklearn.metrics import classification_report
print "naive Bayes Accuracy" ,(mnb.score(X_test, Y_test))
print classification_report(Y_test, mnb_y_predict, target_names = news.target_names)


