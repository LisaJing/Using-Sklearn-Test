from sklearn.datasets import load_boston
boston = load_boston()
#print boston.DESCR


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target,test_size = 0.25, random_state = 33)

import numpy as np

print "max value is : " , np.max(boston.target)
print "min value is : " , np.min(boston.target)
print "average value is : " , np.mean(boston.target)


from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_Y = StandardScaler()

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
Y_train = ss_Y.fit_transform(Y_train)
Y_test = ss_Y.transform(Y_test)

#���Իع�
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)
lr_y_predict = lr.predict(X_test)


#�ݶ��½�
from sklearn.linear_model import SGDRegressor
sgdr = SGDRegressor()
sgdr.fit(X_train, Y_train)
sgdr_y_predict = sgdr.predict(X_test)

#����ָ��
print "the Value of default measurement of LinearRegression is ",lr.score(X_test, Y_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print "Linear Regression r2_score:" , r2_score(Y_test, lr_y_predict)
print "Linear Regression Mean squared error:" , mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(lr_y_predict))
print "Linear Regression Mean absolute error:" , mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(lr_y_predict))


#����ָ��
print "the Value of default measurement of SGDRegressor is ",sgdr.score(X_test, Y_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print "SGDRegressor r2_score:" , r2_score(Y_test, sgdr_y_predict)
print "SGDRegressor Mean squared error:" , mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(sgdr_y_predict))
print "SGDRegressor Mean absolute error:" , mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(sgdr_y_predict))

#########################################################
#֧�����������ع飩
from sklearn.svm import SVR

#���Կɷ�
linearsvr = SVR(kernel = 'linear')
linearsvr.fit(X_train, Y_train)
linearsvr_y_predict = linearsvr.predict(X_test)


#����ʽ�˺���
poly_svr = SVR(kernel = 'poly')
poly_svr .fit(X_train, Y_train)
poly_y_predict = poly_svr.predict(X_test)

#������˺������õ�֧�����������лع�ѵ��
rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, Y_train)
rbf_y_predict = rbf_svr.predict(X_test)

print "\nSVM:"

#��֧���������ĸ���ģ�ͽ�������
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print "linear SVR  R-squared: " , r2_score(Y_test, linearsvr_y_predict)
print "linear SVR  MAE: " , mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(linearsvr_y_predict))
print "linear SVR  MSE: " , mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(linearsvr_y_predict))


print "poly SVR  R-squared: " , r2_score(Y_test, poly_y_predict)
print "poly SVR  MAE: " , mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(poly_y_predict))
print "poly SVR  MSE: " , mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(poly_y_predict))


print "rbf SVR  R-squared: " , r2_score(Y_test, rbf_y_predict)
print "rbf SVR  MAE: " , mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(rbf_y_predict))
print "rbf SVR  MSE: " , mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(rbf_y_predict))


#########################################################
print "\nKNeighbors:"
#K���ڣ��ع飩

#ƽ���ع鷽ʽ weights = 'uniform'
from sklearn.neighbors import KNeighborsRegressor
uni_knr = KNeighborsRegressor(weights = 'uniform')
uni_knr.fit(X_train, Y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

#�����Ȩ�ع� weights = 'distance'
dis_knr = KNeighborsRegressor(weights = 'distance')
dis_knr.fit(X_train, Y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

#��K���ڣ��ع飩������ģ�ͽ�������
print "uniform KNR R-squared : " , r2_score(Y_test, uni_knr_y_predict)
print "uniform KNR MAE : " , mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(uni_knr_y_predict))
print "uniform KNR MSE : " , mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(uni_knr_y_predict))

print "Distance KNR R-squared : " , r2_score(Y_test, dis_knr_y_predict)
print "Distance KNR  MAE : " , mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(dis_knr_y_predict))
print "Distance KNR MSE : " , mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(dis_knr_y_predict))



#########################################################
print "\nDecisionTreeRegressor:"
#�������ع飬�ع���
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train, Y_train)
dtr_y_predict = dtr.predict(X_test)
#����
print "Decision Tree Regressor R-squared : " , r2_score(Y_test, dtr_y_predict)
print "Decision Tree Regressor MAE : " , mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(dtr_y_predict))
print "Decision Tree Regressor MSE : " , mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(dtr_y_predict)) 




###########################################################
#����ģ��

print "ensemble:"
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

#���ɭ��
rfr = RandomForestRegressor()
rfr.fit(X_train, Y_train)
rfr_y_predict = rfr.predict(X_test)
print "ERandomForestRegressor: \n ", np.sort(zip(rfr.feature_importances_, boston.feature_names),axis = 0)

#�������ɭ��
etr = ExtraTreesRegressor()
etr.fit(X_train, Y_train)
etr_y_predict = etr.predict(X_test)
#����������ɭ����ÿ��������Ԥ��Ŀ��Ĺ��׶�
print "ExtraTreesRegressor: \n ", np.sort(zip(etr.feature_importances_, boston.feature_names),axis = 0)


#������
gbt = GradientBoostingRegressor()
gbt.fit(X_train, Y_train)
gbt_y_predict = gbt.predict(X_test)
print "GradientBoostingRegressor: \n ",np.sort(zip(gbt.feature_importances_, boston.feature_names),axis = 0)

####����
print "###########################################################"
print "RandomForestRegressor R-squared: " , r2_score(Y_test, rfr_y_predict)
print "RandomForestRegressor MAE: " , mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(rfr_y_predict))
print "RandomForestRegressor MSE: " , mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(rfr_y_predict))

print "ExtraTreeRegressor R-squared: " , r2_score(Y_test, etr_y_predict)
print "ExtraTreeRegressor MAE: " , mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(etr_y_predict))
print "ExtraTreeRegressor MSE: " , mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(etr_y_predict))

print "GradientBoostingTree R-squared: " , r2_score(Y_test, gbt_y_predict)
print "GradientBoostingTree MAE: " , mean_absolute_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(gbt_y_predict))
print "GradientBoostingTree MSE: " , mean_squared_error(ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(gbt_y_predict))


#RandomForestRegressor
