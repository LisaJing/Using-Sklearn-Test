from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()

datas = [{'city':'Beijing','temperature':26.}, {'city':'Dalian','temperature':23.}, {'city':'Shanghai','temperature':28.}]
print vec.fit_transform(datas).toarray()
print vec.get_feature_names()
