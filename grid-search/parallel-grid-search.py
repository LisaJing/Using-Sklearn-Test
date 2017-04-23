from sklearn.datasets import fetch_20newsgroups

#抓取训练的新闻数据集
news = fetch_20newsgroups(subset = 'all')

import numpy as np
import pandas as pd


from sklearn.cross_validation import train_test_split
#针对前3000条数据进行训练和测试
X_train, X_test, Y_train, Y_test = train_test_split(news.data[:3000], news.target[:3000], test_size = 0.25, random_state = 33)

#使用支持向量机（分类）的模型
from sklearn.svm import SVC


#使用tfidf值来标记特征向量
from sklearn.feature_extraction.text import TfidfVectorizer


#导入Pipeline,，使用Pipeline可以简化搭建机器学习模型的代码
from sklearn.pipeline import Pipeline

#第一个参数为文本抽取器，第二个参数为分类器模型
clf = Pipeline([('vect', TfidfVectorizer(stop_words = 'english', analyzer = 'word')) , ('svc', SVC())])
#处理的两个超参数：svc_gamma 和 svc_5C
parameters = {'svc__gamma':np.logspace(-2, 1, 4), 'svc__C':np.logspace(-1, 1, 3)}


#导入网格搜索模块
from sklearn.grid_search import GridSearchCV

gs = GridSearchCV(clf, parameters, verbose = 2, refit = True, cv = 3, n_jobs = -1)

#执行单线程网格搜索
gs.fit(X_train, Y_train)
print gs.best_params_, gs.best_score_

print gs.score(X_test, Y_test)

