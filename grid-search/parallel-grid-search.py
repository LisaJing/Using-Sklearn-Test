from sklearn.datasets import fetch_20newsgroups

#ץȡѵ�����������ݼ�
news = fetch_20newsgroups(subset = 'all')

import numpy as np
import pandas as pd


from sklearn.cross_validation import train_test_split
#���ǰ3000�����ݽ���ѵ���Ͳ���
X_train, X_test, Y_train, Y_test = train_test_split(news.data[:3000], news.target[:3000], test_size = 0.25, random_state = 33)

#ʹ��֧�������������ࣩ��ģ��
from sklearn.svm import SVC


#ʹ��tfidfֵ�������������
from sklearn.feature_extraction.text import TfidfVectorizer


#����Pipeline,��ʹ��Pipeline���Լ򻯴����ѧϰģ�͵Ĵ���
from sklearn.pipeline import Pipeline

#��һ������Ϊ�ı���ȡ�����ڶ�������Ϊ������ģ��
clf = Pipeline([('vect', TfidfVectorizer(stop_words = 'english', analyzer = 'word')) , ('svc', SVC())])
#�����������������svc_gamma �� svc_5C
parameters = {'svc__gamma':np.logspace(-2, 1, 4), 'svc__C':np.logspace(-1, 1, 3)}


#������������ģ��
from sklearn.grid_search import GridSearchCV

gs = GridSearchCV(clf, parameters, verbose = 2, refit = True, cv = 3, n_jobs = -1)

#ִ�е��߳���������
gs.fit(X_train, Y_train)
print gs.best_params_, gs.best_score_

print gs.score(X_test, Y_test)

