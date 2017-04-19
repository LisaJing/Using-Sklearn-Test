import pandas as pd
import numpy as np
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header = None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header = None)

X_digits = digits_train[np.arange(64)]
Y_digits = digits_train[64]

from sklearn.decomposition import PCA
#将64维向量降成2维
estimator = PCA(n_components = 2)
X_pca = estimator.fit_transform(X_digits)

from matplotlib import pyplot as plt

def plot_pca_scatter():
    colors = ['black','blue','purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in xrange(len(colors)):
        px = X_pca[:,0][Y_digits.as_matrix() == i]
        py = X_pca[:,1][Y_digits.as_matrix() == i]
        plt.scatter(px, py,c = colors[i])

    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Componet')
    plt.show()

plot_pca_scatter()


X_train = digits_train[np.arange(64)]
Y_train = digits_train[64]
X_test = digits_test[np.arange(64)]
Y_test = digits_test[64]


#使用SVM分类器训练
from sklearn.svm import LinearSVC
lsvc = LinearSVC()
lsvc.fit(X_train, Y_train)
lsvc_y_predict = lsvc.predict(X_test)


#PCA降维至20维
from sklearn.decomposition import PCA
estimator = PCA(n_components = 20)
pca_X_train = estimator.fit_transform(X_train)
pca_X_test = estimator.transform(X_test)

#SVM分类器进行训练
pca_svc = LinearSVC()
pca_svc.fit(pca_X_train , Y_train)
pca_svc_y_predict = pca_svc.predict(pca_X_test)


#评估
from sklearn.metrics import classification_report
print lsvc.score(X_test, Y_test)
print classification_report(Y_test,lsvc_y_predict, target_names = np.arange(10).astype(str))
print pca_svc.score(pca_X_test, Y_test)
print classification_report(Y_test,pca_svc_y_predict, target_names = np.arange(10).astype(str))
