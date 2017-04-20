X_train = [[6], [8], [10], [14], [18]]
Y_train = [[7], [9], [13], [17.5], [18]]

X_test = [[6], [8], [11], [16]]
Y_test = [[8], [12], [15], [18]]


#四次多项式
from sklearn.preprocessing import PolynomialFeatures
poly4 = PolynomialFeatures(degree = 4)
X_train_poly4 = poly4.fit_transform(X_train)

X_test_poly4 = poly4.transform(X_test)



#L1 norm
from sklearn.linear_model import Lasso
lasso_poly4 = Lasso()
lasso_poly4.fit(X_train_poly4, Y_train)

print lasso_poly4.score(X_test_poly4, Y_test)

print lasso_poly4.coef_

#L2 norm
from sklearn.linear_model import Ridge
ridge_poly4 = Ridge()
ridge_poly4.fit(X_train_poly4, Y_train)

print ridge_poly4.score(X_test_poly4, Y_test)

print ridge_poly4.coef_
