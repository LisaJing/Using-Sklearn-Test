import pandas as pd
import numpy as np

df_train = pd.read_csv('D:\\python_Learining\\Datasets\\Breast-Cancer\\breast-cancer-train.csv')
df_test = pd.read_csv('D:\\python_Learining\\Datasets\\Breast-Cancer\\breast-cancer-test.csv')

df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

print df_test.shape

#df_train_negative = df_train.loc[df_train['Type'] == 0][['Clump Thickness', 'Cell Size']]
#df_train_positive = df_train.loc[df_train['Type'] == 1][['Clump Thickness', 'Cell Size']]

import matplotlib.pyplot as plt

plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker = 'o',s = 20, c = 'red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker = 'x',s = 15, c = 'black')

plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')




intercept = np.random.random([1])
coef = np.random.random([2])
lx = np.arange(0, 12)
ly = (-intercept - lx * coef[0])/coef[1]
plt.plot(lx, ly, c = 'yellow')

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10],df_train['Type'][:10])

intercept = lr.intercept_
coef = lr.coef_[0,:]
ly = (-intercept - lx * coef[0])/coef[1]
print 'Testing accuracy(10 training samples):',lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type'])
plt.plot(lx, ly, c = 'green')

lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:],df_train['Type'][:])

intercept = lr.intercept_
coef = lr.coef_[0,:]
ly = (-intercept - lx * coef[0])/coef[1]
plt.plot(lx, ly, c = 'blue')

print 'Testing accuracy(all training samples):',lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type'])

plt.show()
