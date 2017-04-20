from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset = 'all')

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(news.data, news.target, test_size = 0.25, random_state = 33)


#using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb_count = MultinomialNB()
mnb_count.fit(X_count_train, Y_train)


print "Accuracy:" , mnb_count.score(X_count_test, Y_test)


mnb_count_y_predict = mnb_count.predict(X_count_test)
from sklearn.metrics import classification_report
print classification_report(Y_test, mnb_count_y_predict, target_names = news.target_names)



#using TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
X_tfidf_train = tfidf_vec.fit_transform(X_train)
X_tfidf_test = tfidf_vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(X_tfidf_train, Y_train)
print "Accuracy:" , mnb_tfidf.score(X_tfidf_test, Y_test)
mnb_tfidf_y_predict = mnb_tfidf.predict(X_tfidf_test)
print classification_report(Y_test, mnb_tfidf_y_predict, target_names = news.target_names)



#filter the stop-word

count_filter_vec, tfidf_filter_vec = CountVectorizer(analyzer = 'word', stop_words = 'english'), TfidfVectorizer(analyzer = 'word', stop_words = 'english')
X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_filter_train, Y_train)
mnb_count_filter_y_predict = mnb_count_filter.predict(X_count_filter_test)
print "Accuracy:", mnb_count_filter.score(X_count_filter_test, Y_test)
print classification_report(Y_test, mnb_count_filter_y_predict, target_names = news.target_names)


X_tfidf_filter_train = tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test = tfidf_filter_vec.transform(X_test)


mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_filter_train, Y_train)
mnb_tfidf_filter_y_predict = mnb_tfidf_filter.predict(X_tfidf_filter_test)
print "Accuracy:" , mnb_tfidf_filter.score(X_tfidf_filter_test,Y_test)
print classification_report(Y_test, mnb_tfidf_filter_y_predict,target_names = news.target_names)









