import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header = None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header = None)


X_train = digits_train[np.arange(64)]
Y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
Y_test = digits_test[64]

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 10)
km.fit(X_train)
km_y_predict = km.predict(X_test)

from sklearn.metrics import adjusted_rand_score
print "adjusted_rand_score: " ,adjusted_rand_score(Y_test, km_y_predict)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

plt.subplot(3, 3, 1)
x1 = np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
x2 = np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])
X = np.array(zip(x1, x2)).reshape(len(x1), 2)

#1号子图中是原始点的分布图
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Instances')
plt.scatter(x1, x2)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'r','m']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+', 'v', '*']


clusters = [2, 3, 4, 5, 6, 8, 10]
subplot_counter = 1
sc_scores = []
for t in clusters:
    subplot_counter += 1
    plt.subplot(3,3,subplot_counter)
    kmeans_model = KMeans(n_clusters = t).fit(X)

    print kmeans_model.labels_

    for i, l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i], x2[i], color = colors[l], marker = markers[l], ls = 'None')

    plt.xlim([0, 10])
    plt.ylim([0, 10])
    sc_score = silhouette_score(X, kmeans_model.labels_, metric = 'euclidean')
    sc_scores.append(sc_score)
    plt.title('K = %s, silhouette_score = %0.03f' %(t, sc_score))

plt.figure()
plt.plot(clusters, sc_scores, '*-')
plt.xlabel('Numbers of clusters')
plt.ylabel('Silhoutte Coefficient Score')
plt.show()





