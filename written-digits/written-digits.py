from sklearn.datasets import load_digits
digits = load_digits()
print digits.data.shape
print digits.data[0]


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size = 0.25, random_state = 33)
print X_train.shape, X_test.shape
print Y_train.shape, Y_test.shape


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

from sklearn.metrics import classification_report

#使用lr进行预测
print "LogisticRegression: "
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
lr_y_predict = lr.predict(X_test)


print "lr Accuracy: ", lr.score(X_test, Y_test)
print classification_report(Y_test, lr_y_predict, target_names = ["0","1","2","3","4","5","6","7","8","9"])


#使用SGDClassifier进行预测
print "SGDClassifier: "
from sklearn.linear_model import SGDClassifier
sgdc = SGDClassifier()
sgdc.fit(X_train, Y_train)
sgdc_y_predict = sgdc.predict(X_test)

print "sgdc Accuracy: ", sgdc.score(X_test, Y_test)
print classification_report(Y_test, sgdc_y_predict, target_names = ["0","1","2","3","4","5","6","7","8","9"])


#使用SVM进行预测
print "Linear SVMClassifier:"
from sklearn.svm import LinearSVC
lsvc = LinearSVC()
lsvc.fit(X_train, Y_train)
lsvc_y_predict = lsvc.predict(X_test)

print "Linear SVM Accuracy: ",lsvc.score(X_test, Y_test)

print classification_report(Y_test, lsvc_y_predict, target_names = digits.target_names.astype(str))


