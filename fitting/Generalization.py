X_train = [[6], [8], [10], [14], [18]]
Y_train = [[7], [9], [13], [17.5], [18]]

#线性：一次多项式
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)

import numpy as np
xx = np.linspace(0, 26, 100)
xx = xx.reshape(xx.shape[0], 1)
yy = lr.predict(xx)


import matplotlib.pyplot as plt
plt.scatter(X_train, Y_train)

plt1,= plt.plot(xx, yy, label = 'Degree=1')

print "R-squared:" , lr.score(X_train, Y_train)

#二次多项式
from sklearn.preprocessing import PolynomialFeatures
poly2 = PolynomialFeatures(degree = 2)
X_train_poly2 = poly2.fit_transform(X_train)
regressor_poly2 = LinearRegression()
regressor_poly2.fit(X_train_poly2, Y_train)

xx_poly2 = poly2.transform(xx)
yy_poly2 = regressor_poly2.predict(xx_poly2)

plt.scatter(X_train, Y_train)

plt2, = plt.plot(xx, yy_poly2, label = 'Degree=2')

print "R-squared:" , regressor_poly2.score(X_train_poly2, Y_train)

#四次多项式
from sklearn.preprocessing import PolynomialFeatures
poly4 = PolynomialFeatures(degree = 4)
X_train_poly4 = poly4.fit_transform(X_train)
regressor_poly4 = LinearRegression()
regressor_poly4.fit(X_train_poly4, Y_train)

xx_poly4 = poly4.transform(xx)
yy_poly4 = regressor_poly4.predict(xx_poly4)

plt.scatter(X_train, Y_train)

plt4, = plt.plot(xx, yy_poly4, label = 'Degree=4')

print "R-squared:" , regressor_poly4.score(X_train_poly4, Y_train)



plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles = [plt1,plt2, plt4])
plt.show()


X_test = [[6], [8], [11], [16]]
Y_test = [[8], [12], [15], [18]]

print "\n ##################"
print "R-squared:" , lr.score(X_test, Y_test)
xx_poly2 = poly2.transform(X_test)
print "R-squared:" , regressor_poly2.score(xx_poly2, Y_test)

xx_poly4 = poly4.transform(X_test)
print "R-squared:" , regressor_poly4.score(xx_poly4, Y_test)
